from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import json
import os
import re
import webbrowser

from tutor_graph import CHAT_GRAPH, LLMRequestError, load_config, request_completion


HOST = "0.0.0.0" if "PORT" in os.environ else "127.0.0.1"
PORT = int(os.environ.get("PORT", "8000"))
MAX_CHAT_BODY_BYTES = 16_384
MAX_CHAT_MESSAGE_CHARS = 3_000
MAX_HISTORY_MESSAGES = 8
MAX_HISTORY_MESSAGE_CHARS = 1_200
PAGE = Path(__file__).resolve().with_name("driver.html")
MATERIALS_DIR = Path(__file__).resolve().parent / "materials"
TOPICS_FILE = MATERIALS_DIR / "topics.json"


def ping_llm() -> tuple[int, str]:
    config = load_config()
    if not config["OPENAI_API_KEY"]:
        return 503, "API key is missing. Add OPENAI_API_KEY to .env."
    try:
        request_completion([{"role": "user", "content": "Reply with OK."}])
    except LLMRequestError as error:
        return 502, str(error)
    return 200, f'Connected to {config["GEMMA_MODEL"]}.'


def send_json(handler: BaseHTTPRequestHandler, status: int, message: str) -> None:
    content = json.dumps(
        {"ok": status == 200, "message": message},
        ensure_ascii=False,
    ).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(content)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    handler.wfile.write(content)


def send_chat_json(handler: BaseHTTPRequestHandler, status: int, payload: dict) -> None:
    content = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(content)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    handler.wfile.write(content)


def read_chat_request(handler: BaseHTTPRequestHandler) -> tuple[str, list[dict[str, str]]]:
    try:
        content_length = int(handler.headers.get("Content-Length", "0"))
    except ValueError as error:
        raise ValueError("Invalid Content-Length header.") from error
    if content_length <= 0:
        raise ValueError("A chat message is required.")
    if content_length > MAX_CHAT_BODY_BYTES:
        raise OverflowError("The chat request is too large.")
    try:
        body = json.loads(handler.rfile.read(content_length))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError("The chat request must contain valid JSON.") from error
    if not isinstance(body, dict):
        raise ValueError("The chat request has an invalid format.")

    message = body.get("message")
    history = body.get("history", [])
    if not isinstance(message, str) or not message.strip():
        raise ValueError("Enter a question or example to solve.")
    if len(message) > MAX_CHAT_MESSAGE_CHARS:
        raise ValueError(f"Keep the question under {MAX_CHAT_MESSAGE_CHARS} characters.")
    if not isinstance(history, list):
        raise ValueError("Chat history has an invalid format.")

    safe_history: list[dict[str, str]] = []
    for entry in history[-MAX_HISTORY_MESSAGES:]:
        if (
            not isinstance(entry, dict)
            or entry.get("role") not in ("user", "assistant")
            or not isinstance(entry.get("content"), str)
        ):
            raise ValueError("Chat history has an invalid format.")
        content = entry["content"].strip()
        if content:
            safe_history.append({
                "role": entry["role"],
                "content": content[:MAX_HISTORY_MESSAGE_CHARS],
            })
    return message.strip(), safe_history


class PageHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path == "/materials/topics.json":
            content = TOPICS_FILE.read_bytes()
            content_type = "application/json; charset=utf-8"
        elif re.fullmatch(r"/materials/[a-z0-9-]+\.json", self.path):
            topic_file = MATERIALS_DIR / self.path.rsplit("/", 1)[1]
            if not topic_file.is_file():
                self.send_error(404, "Topic not found")
                return
            content = topic_file.read_bytes()
            content_type = "application/json; charset=utf-8"
        elif self.path in ("/", "/driver.html"):
            content = PAGE.read_bytes()
            content_type = "text/html; charset=utf-8"
        else:
            self.send_error(404, "Page not found")
            return

        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        if self.path.startswith("/materials/"):
            self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(content)

    def do_POST(self) -> None:
        if self.path == "/api/ping":
            status, message = ping_llm()
            send_json(self, status, message)
            return
        if self.path != "/api/chat":
            self.send_error(404, "Page not found")
            return

        try:
            message, history = read_chat_request(self)
            result = CHAT_GRAPH.invoke({"message": message, "history": history})
        except OverflowError as error:
            send_chat_json(self, 413, {"ok": False, "message": str(error)})
            return
        except ValueError as error:
            send_chat_json(self, 400, {"ok": False, "message": str(error)})
            return
        except LLMRequestError as error:
            send_chat_json(self, 502, {"ok": False, "message": str(error)})
            return

        send_chat_json(self, 200, {"ok": True, "answer": result["answer"]})


def main() -> None:
    if not PAGE.is_file():
        raise FileNotFoundError(f"HTML page not found: {PAGE}")

    server = ThreadingHTTPServer((HOST, PORT), PageHandler)
    url = f"http://127.0.0.1:{PORT}/"
    print(f"Serving Midterm Calculus at {url}")
    if "PORT" not in os.environ:
        print("Press Ctrl+C to stop the server.")
        webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
