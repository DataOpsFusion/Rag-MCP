"""
Quick script to ping an LM Studio instance (OpenAI-compatible API) and print the reply.

Defaults assume LM Studio is on 192.168.0.242:1234 serving model "gpt-oss-20B".
Override via CLI flags or environment variables:
  LMSTUDIO_HOST, LMSTUDIO_PORT, LMSTUDIO_MODEL, LMSTUDIO_API_KEY (if your server requires one).

Usage:
  python test_lmstudio.py --prompt "Hello from MCP"
"""

import argparse
import json
import os
import sys
from typing import Any, Dict

import httpx


def call_lmstudio(
    prompt: str,
    host: str,
    port: int,
    model: str,
    temperature: float,
    api_key: str | None = None,
) -> Dict[str, Any]:
    url = f"http://{host}:{port}/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "stream": False,
    }

    resp = httpx.post(url, headers=headers, json=payload, timeout=60.0)
    resp.raise_for_status()
    return resp.json()


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Ping LM Studio model and print the reply.")
    parser.add_argument("--prompt", required=True, help="Prompt text to send.")
    parser.add_argument("--host", default=os.getenv("LMSTUDIO_HOST", "192.168.0.242"), help="LM Studio host.")
    parser.add_argument("--port", type=int, default=int(os.getenv("LMSTUDIO_PORT", "1234")), help="LM Studio port.")
    parser.add_argument(
        "--model",
        default=os.getenv("LMSTUDIO_MODEL", "gpt-oss-20B"),
        help="Model name exposed by LM Studio.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=float(os.getenv("LMSTUDIO_TEMPERATURE", "0.2")),
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("LMSTUDIO_API_KEY"),
        help="Optional API key if LM Studio requires one.",
    )

    args = parser.parse_args(argv)

    try:
        data = call_lmstudio(
            prompt=args.prompt,
            host=args.host,
            port=args.port,
            model=args.model,
            temperature=args.temperature,
            api_key=args.api_key,
        )
    except Exception as exc:  # pragma: no cover - simple harness
        print(f"[error] request failed: {exc}", file=sys.stderr)
        return 1

    choice = (data.get("choices") or [{}])[0]
    message = (choice.get("message") or {}).get("content")
    print("--- LM Studio reply ---")
    print(message or json.dumps(data, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
