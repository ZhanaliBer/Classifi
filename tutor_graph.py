import json
import os
import re
from pathlib import Path
from typing import Literal, TypedDict

import requests
from langgraph.graph import END, START, StateGraph


ENV_FILE = Path(__file__).resolve().with_name(".env")
MAX_HISTORY_MESSAGES = 8

INTENT_SYSTEM_PROMPT = """Определи тип математического запроса для курса Calculus I.
Верни ровно одно слово: formula, если пользователь просит правило/формулу или объяснение теории;
solve, если просит решить, вычислить или проверить конкретный пример.
При сомнении используй solve."""
CHAT_SYSTEM_PROMPT = r"""Вы — терпеливый преподаватель Calculus I. Отвечайте по-русски.
Автоматически определите, просит ли пользователь формулу/объяснение формулы или пошаговое решение примера.
Решайте точно и не пропускайте преобразования. В шагах сначала назовите правило, затем покажите его применение.
Формулы записывайте в LaTeX, пригодном для MathJax; поле formula каждого шага должно содержать только LaTeX без разделителей $.
Не используйте Markdown, HTML или блочные окружения в JSON-строках.
Если в условии не хватает данных, не выдумывайте их: сформулируйте один короткий уточняющий вопрос в summary, steps оставьте пустым.
Верните только валидный JSON такой формы:
{"intent":"formula|solve","summary":"краткий план или уточняющий вопрос","steps":[{"explanation":"название правила и пояснение","formula":"f'(x)=..."}],"answer":"итоговый ответ обычным текстом с LaTeX внутри \\(...\\), если он нужен"}
Для запроса формулы intent=formula; кратко объясните смысл и условия её применения.
Для решения примера intent=solve; перечислите шаги в порядке решения и явно выделите ответ.
"""


class LLMRequestError(RuntimeError):
    pass


class ChatStep(TypedDict):
    explanation: str
    formula: str


class ChatAnswer(TypedDict):
    intent: Literal["formula", "solve"]
    summary: str
    steps: list[ChatStep]
    answer: str


class ChatState(TypedDict, total=False):
    message: str
    history: list[dict[str, str]]
    intent: Literal["formula", "solve"]
    answer: ChatAnswer


def load_config() -> dict[str, str]:
    config = {
        "OPENAI_API_KEY": "",
        "OPENAI_BASE_URL": "https://llm.alem.ai/v1",
        "GEMMA_MODEL": "gemma4",
    }
    if ENV_FILE.is_file():
        for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
            key, separator, value = line.partition("=")
            if separator and key.strip() in config:
                config[key.strip()] = value.strip().strip("\"'")

    for key in config:
        if key in os.environ:
            config[key] = os.environ[key]
    return config


def request_completion(messages: list[dict[str, str]]) -> str:
    config = load_config()
    api_key = config["OPENAI_API_KEY"]
    if not api_key:
        raise LLMRequestError("API key is missing. Add OPENAI_API_KEY to .env.")

    endpoint = f'{config["OPENAI_BASE_URL"].rstrip("/")}/chat/completions'
    payload = {
        "model": config["GEMMA_MODEL"],
        "messages": messages,
    }
    try:
        response = requests.post(
            endpoint,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()
        content = data["choices"][0]["message"]["content"]
    except requests.RequestException as error:
        response_body = error.response.text[:1_000] if error.response is not None else ""
        if error.response is not None and error.response.status_code in (401, 403):
            message = "The API rejected the key or account access."
        elif error.response is not None and error.response.status_code == 429:
            message = "The API is rate-limited or the account has no available quota."
        elif error.response is not None:
            message = f"The API returned HTTP {error.response.status_code}."
        else:
            message = f"Could not connect to the LLM API: {error}"
        if response_body:
            message = f"{message} Response: {response_body}"
        raise LLMRequestError(message) from error
    except (KeyError, IndexError, TypeError, ValueError) as error:
        raise LLMRequestError(
            "The API returned an invalid response or no completion."
        ) from error

    if not isinstance(content, str) or not content.strip():
        raise LLMRequestError("The API returned an empty completion.")
    return content.strip()


def parse_chat_answer(content: str) -> ChatAnswer:
    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE)
    try:
        raw = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start < 0 or end <= start:
            raise LLMRequestError("The AI response was not valid JSON. Please try again.")
        try:
            raw = json.loads(cleaned[start:end + 1])
        except json.JSONDecodeError as error:
            raise LLMRequestError(
                "The AI response was not valid JSON. Please try again."
            ) from error

    if not isinstance(raw, dict):
        raise LLMRequestError("The AI response has an invalid format. Please try again.")
    intent = raw.get("intent")
    summary = raw.get("summary")
    steps = raw.get("steps")
    answer = raw.get("answer")
    if (
        intent not in ("formula", "solve")
        or not isinstance(summary, str)
        or not isinstance(steps, list)
        or not isinstance(answer, str)
        or len(steps) > 16
        or any(
            not isinstance(step, dict)
            or not isinstance(step.get("explanation"), str)
            or not isinstance(step.get("formula"), str)
            for step in steps
        )
    ):
        raise LLMRequestError("The AI response has an invalid format. Please try again.")
    return {
        "intent": intent,
        "summary": summary[:4_000],
        "steps": [
            {
                "explanation": step["explanation"][:2_000],
                "formula": step["formula"][:2_000],
            }
            for step in steps
        ],
        "answer": answer[:4_000],
    }


def classify_chat_intent(state: ChatState) -> dict[str, str]:
    intent = request_completion(
        [
            {"role": "system", "content": INTENT_SYSTEM_PROMPT},
            *state.get("history", [])[-MAX_HISTORY_MESSAGES:],
            {"role": "user", "content": state["message"]},
        ]
    ).strip().lower().strip(".")
    if intent not in ("formula", "solve"):
        raise LLMRequestError("The AI could not classify this request. Please rephrase it.")
    return {"intent": intent}


def route_chat_intent(state: ChatState) -> Literal["formula", "solve"]:
    return state["intent"]


def generate_chat_answer(state: ChatState) -> dict[str, ChatAnswer]:
    selected_intent = state["intent"]
    intent_label = "формулу или объяснение правила" if selected_intent == "formula" else "пошаговое решение примера"
    messages = [
        {
            "role": "system",
            "content": (
                f"{CHAT_SYSTEM_PROMPT}\n"
                f"Запрос классифицирован как {intent_label}; сохрани intent=\"{selected_intent}\"."
            ),
        },
        *state.get("history", [])[-MAX_HISTORY_MESSAGES:],
        {"role": "user", "content": state["message"]},
    ]
    answer = parse_chat_answer(request_completion(messages))
    answer["intent"] = selected_intent
    return {"answer": answer}


def build_chat_graph():
    graph = StateGraph(ChatState)
    graph.add_node("classify_request", classify_chat_intent)
    graph.add_node("explain_formula", generate_chat_answer)
    graph.add_node("solve_example", generate_chat_answer)
    graph.add_edge(START, "classify_request")
    graph.add_conditional_edges(
        "classify_request",
        route_chat_intent,
        {
            "formula": "explain_formula",
            "solve": "solve_example",
        },
    )
    graph.add_edge("explain_formula", END)
    graph.add_edge("solve_example", END)
    return graph.compile()


CHAT_GRAPH = build_chat_graph()
