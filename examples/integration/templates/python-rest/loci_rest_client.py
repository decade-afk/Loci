import json
import os
import sys
import urllib.request


def safe_print(prefix: str, value: str) -> None:
    line = f"{prefix}{value}\n"
    try:
        sys.stdout.buffer.write(line.encode("utf-8", errors="replace"))
    except Exception:
        print(f"{prefix}{value.encode('ascii', errors='replace').decode('ascii')}")


def get(url: str) -> str:
    with urllib.request.urlopen(url, timeout=15) as resp:
        return resp.read().decode("utf-8")


def post_json(url: str, payload: dict) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main() -> None:
    base = os.getenv("LOCI_BASE_URL", "http://127.0.0.1:8080")

    safe_print("health: ", get(f"{base}/v1/health"))
    safe_print("info: ", get(f"{base}/v1/info"))

    result = post_json(
        f"{base}/v1/generate",
        {
            "prompt": "Hello from Python template",
            "max_tokens": 8,
            "temperature": 0.7,
        },
    )
    safe_print("response: ", result.get("response", ""))


if __name__ == "__main__":
    main()
