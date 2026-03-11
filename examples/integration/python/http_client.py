import json
import urllib.request


def post_json(url: str, payload: dict) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main() -> None:
    health = urllib.request.urlopen("http://127.0.0.1:8080/health", timeout=5).read().decode()
    print("health:", health)

    result = post_json(
        "http://127.0.0.1:8080/generate",
        {"prompt": "Hello from HTTP", "max_tokens": 32},
    )
    print("response:", result.get("response", ""))


if __name__ == "__main__":
    main()
