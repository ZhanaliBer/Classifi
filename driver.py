from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from pathlib import Path
import json
import os
import webbrowser


HOST = "0.0.0.0"
PORT = int(os.environ.get("PORT", "8000"))
PAGE = Path(__file__).resolve().with_name("driver.html")
ENV_FILE = Path(__file__).resolve().with_name(".env")


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


def ping_llm() -> tuple[int, str]:
    config = load_config()
    api_key = config["OPENAI_API_KEY"]
    if not api_key:
        return 503, "API key is missing. Add OPENAI_API_KEY to .env."

    endpoint = f'{config["OPENAI_BASE_URL"].rstrip("/")}/chat/completions'
    payload = json.dumps({
        "model": config["GEMMA_MODEL"],
        "messages": [{"role": "user", "content": "Reply with OK."}],
        "max_tokens": 8,
    }).encode("utf-8")
    request = Request(
        endpoint,
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urlopen(request, timeout=20) as response:
            result = json.loads(response.read())
    except HTTPError as error:
        if error.code in (401, 403):
            return 502, "The API rejected the key or account access."
        if error.code == 429:
            return 502, "The API is rate-limited or the account has no available quota."
        return 502, f"The API returned HTTP {error.code}."
    except (json.JSONDecodeError, UnicodeDecodeError):
        return 502, "The API returned an invalid response."
    except (URLError, TimeoutError, OSError):
        return 502, "Could not connect to the LLM API."

    if (
        not isinstance(result, dict)
        or not isinstance(result.get("choices"), list)
        or not result["choices"]
    ):
        return 502, "The API returned no completion."

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


class PageHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path not in ("/", "/driver.html"):
            self.send_error(404, "Page not found")
            return

        content = PAGE.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def do_POST(self) -> None:
        if self.path != "/api/ping":
            self.send_error(404, "Page not found")
            return

        status, message = ping_llm()
        send_json(self, status, message)


def main() -> None:
    if not PAGE.is_file():
        raise FileNotFoundError(f"HTML page not found: {PAGE}")

    server = ThreadingHTTPServer((HOST, PORT), PageHandler)
    url = f"http://127.0.0.1:{PORT}/"
    print(f"Serving Midterm Calculus at {url}")
    if not os.environ.get("RENDER"):
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
