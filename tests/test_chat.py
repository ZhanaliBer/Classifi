import io
import json
import threading
import unittest
from http.server import ThreadingHTTPServer
from urllib.request import Request, urlopen
from unittest.mock import MagicMock, patch

import driver
import requests
import tutor_graph


class ChatAnswerTests(unittest.TestCase):
    def test_parses_structured_latex_answer(self):
        payload = {
            "intent": "solve",
            "summary": "Применим правило.",
            "steps": [{"explanation": "Правило произведения.", "formula": "(fg)'=f'g+fg'"}],
            "answer": "Готово.",
        }
        answer = tutor_graph.parse_chat_answer(f"```json\n{json.dumps(payload)}\n```")
        self.assertEqual(answer["intent"], "solve")
        self.assertEqual(answer["steps"][0]["formula"], "(fg)'=f'g+fg'")

    def test_rejects_malformed_answer(self):
        with self.assertRaisesRegex(RuntimeError, "valid JSON"):
            tutor_graph.parse_chat_answer("not an answer")

    def test_completion_uses_documented_url_headers_payload_and_timeout(self):
        response = MagicMock()
        response.json.return_value = {
            "choices": [{"message": {"content": "formula"}}],
        }
        messages = [{"role": "user", "content": "test"}]
        with (
            patch.object(tutor_graph, "load_config", return_value={
                "OPENAI_API_KEY": "test-secret",
                "OPENAI_BASE_URL": "https://example.test/v1/",
                "GEMMA_MODEL": "test-model",
            }),
            patch.object(tutor_graph.requests, "post", return_value=response) as post,
        ):
            content = tutor_graph.request_completion(messages)

        self.assertEqual(content, "formula")
        post.assert_called_once_with(
            "https://example.test/v1/chat/completions",
            headers={
                "Authorization": "Bearer test-secret",
                "Content-Type": "application/json",
            },
            json={"model": "test-model", "messages": messages},
            timeout=60,
        )
        response.raise_for_status.assert_called_once_with()

    def test_completion_reports_http_errors_with_response_body(self):
        response = MagicMock()
        response.text = "invalid API key"
        response.status_code = 401
        error = requests.HTTPError("unauthorized", response=response)
        with (
            patch.object(tutor_graph, "load_config", return_value={
                "OPENAI_API_KEY": "test-secret",
                "OPENAI_BASE_URL": "https://example.test/v1",
                "GEMMA_MODEL": "test-model",
            }),
            patch.object(
                tutor_graph.requests,
                "post",
                return_value=MagicMock(raise_for_status=MagicMock(side_effect=error)),
            ),
        ):
            with self.assertRaisesRegex(tutor_graph.LLMRequestError, "invalid API key"):
                tutor_graph.request_completion([{"role": "user", "content": "test"}])

    def test_completion_reports_malformed_api_response(self):
        response = MagicMock()
        response.json.return_value = {"choices": []}
        with (
            patch.object(tutor_graph, "load_config", return_value={
                "OPENAI_API_KEY": "test-secret",
                "OPENAI_BASE_URL": "https://example.test/v1",
                "GEMMA_MODEL": "test-model",
            }),
            patch.object(tutor_graph.requests, "post", return_value=response),
        ):
            with self.assertRaisesRegex(tutor_graph.LLMRequestError, "invalid response"):
                tutor_graph.request_completion([{"role": "user", "content": "test"}])

    def test_langgraph_routes_formula_and_example_requests(self):
        for intent in ("formula", "solve"):
            with self.subTest(intent=intent):
                generated = json.dumps({
                    "intent": intent,
                    "summary": "Пошаговое объяснение",
                    "steps": [{"explanation": "Применим правило.", "formula": "(fg)'=f'g+fg'"}],
                    "answer": "Готово.",
                })
                with patch.object(
                    tutor_graph,
                    "request_completion",
                    side_effect=[intent, generated],
                ) as completion:
                    result = driver.CHAT_GRAPH.invoke({
                        "message": "Объясни правило" if intent == "formula" else "Реши пример",
                        "history": [],
                    })
                self.assertEqual(result["intent"], intent)
                self.assertEqual(result["answer"]["intent"], intent)
                self.assertEqual(completion.call_count, 2)

    def test_chat_input_is_validated_and_history_is_bounded(self):
        history = [
            {"role": "user", "content": "предыдущий вопрос " * 100},
            {"role": "assistant", "content": "предыдущий ответ"},
        ]
        body = json.dumps({"message": "  Реши пример  ", "history": history}).encode()
        handler = type("RequestHandler", (), {
            "headers": {"Content-Length": str(len(body))},
            "rfile": io.BytesIO(body),
        })()
        message, safe_history = driver.read_chat_request(handler)
        self.assertEqual(message, "Реши пример")
        self.assertEqual(len(safe_history[0]["content"]), driver.MAX_HISTORY_MESSAGE_CHARS)
        self.assertEqual(len(safe_history), 2)

    def test_chat_input_rejects_empty_message(self):
        body = json.dumps({"message": " "}).encode()
        handler = type("RequestHandler", (), {
            "headers": {"Content-Length": str(len(body))},
            "rfile": io.BytesIO(body),
        })()
        with self.assertRaisesRegex(ValueError, "Enter a question"):
            driver.read_chat_request(handler)

    def test_ping_uses_shared_completion_client(self):
        with (
            patch.object(driver, "load_config", return_value={
                "OPENAI_API_KEY": "test-secret",
                "OPENAI_BASE_URL": "https://example.test/v1",
                "GEMMA_MODEL": "test-model",
            }),
            patch.object(driver, "request_completion", return_value="OK") as request,
        ):
            status, message = driver.ping_llm()
        self.assertEqual((status, message), (200, "Connected to test-model."))
        request.assert_called_once_with([{"role": "user", "content": "Reply with OK."}])

    def test_chat_http_endpoint_returns_graph_answer(self):
        answer = {
            "intent": "solve",
            "summary": "Применим правило произведения.",
            "steps": [{"explanation": "Дифференцируем оба множителя.", "formula": "(fg)'=f'g+fg'"}],
            "answer": "Готово.",
        }
        server = ThreadingHTTPServer(("127.0.0.1", 0), driver.PageHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            payload = json.dumps({"message": "Реши пример", "history": []}).encode()
            request = Request(
                f"http://127.0.0.1:{server.server_port}/api/chat",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with patch.object(driver.CHAT_GRAPH, "invoke", return_value={"answer": answer}):
                with urlopen(request, timeout=5) as response:
                    result = json.loads(response.read())
            self.assertEqual(result, {"ok": True, "answer": answer})
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    def test_static_page_and_topics_endpoints_still_respond(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), driver.PageHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            for path, expected_type in (
                ("/", "text/html"),
                ("/materials/topics.json", "application/json"),
            ):
                with urlopen(f"http://127.0.0.1:{server.server_port}{path}", timeout=5) as response:
                    self.assertEqual(response.status, 200)
                    self.assertIn(expected_type, response.headers["Content-Type"])
                    self.assertTrue(response.read())
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
