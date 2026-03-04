#!/bin/env python3
"""
Summarize extracted lecture text using a free LLM.

Supported backends:
  - gemini  : Google Gemini API (free tier, requires GEMINI_API_KEY env var)
  - ollama  : Local Ollama server (completely free, no API key)

Usage:
  # Using Gemini (set your API key first)
  export GEMINI_API_KEY="your-key-here"
  python summarize_lecture.py Lecture-6_extracted.txt

  # Using Ollama (make sure ollama is running locally)
  python summarize_lecture.py Lecture-6_extracted.txt --backend ollama --model llama3

  # Specify output file
  python summarize_lecture.py Lecture-6_extracted.txt -o Lecture-6-summary.md
"""

import argparse
import json
import os
import sys
import urllib.request
import urllib.error


SYSTEM_PROMPT = """You are an expert academic note-taker. Given raw text extracted from
a university lecture PDF, produce a well-structured summary in Markdown format.

Follow these rules:
- Start with a top-level heading (#) using the lecture topic.
- Include metadata (Lecturer, Course, Topic) at the top.
- Organize content into logical numbered sections (## headings).
- Under each section, use bullet points for key concepts, definitions, and formulas.
- Preserve important mathematical notation using LaTeX-style inline ($...$) formatting.
- Highlight key terms in bold.
- Keep the summary concise but comprehensive — capture all important ideas.
- End with a brief "Key Takeaways" section summarizing the most important points.
"""


def read_input(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def summarize_gemini(text: str, model: str, api_key: str) -> str:
    """Call Google Gemini API (free tier)."""
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}"
        f":generateContent?key={api_key}"
    )
    payload = {
        "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": [
            {
                "parts": [
                    {
                        "text": (
                            "Summarize the following lecture notes:\n\n"
                            + text
                        )
                    }
                ]
            }
        ],
        "generationConfig": {"temperature": 0.3, "maxOutputTokens": 4096},
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req) as resp:
            result = json.loads(resp.read().decode("utf-8"))
        return result["candidates"][0]["content"]["parts"][0]["text"]
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        print(f"Gemini API error {e.code}: {body}", file=sys.stderr)
        sys.exit(1)


def summarize_ollama(text: str, model: str) -> str:
    """Call local Ollama API."""
    url = "http://localhost:11434/api/chat"
    payload = {
        "model": model,
        "stream": False,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Summarize the following lecture notes:\n\n" + text
                ),
            },
        ],
        "options": {"temperature": 0.3},
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            result = json.loads(resp.read().decode("utf-8"))
        return result["message"]["content"]
    except urllib.error.URLError as e:
        print(
            f"Ollama error: {e}\n"
            "Make sure Ollama is running (ollama serve) and the model is pulled.",
            file=sys.stderr,
        )
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Summarize lecture text using a free LLM."
    )
    parser.add_argument(
        "input",
        help="Path to the extracted lecture text file (.txt)"
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output Markdown file (default: <input>-summary.md)",
    )
    parser.add_argument(
        "--backend",
        choices=["gemini", "ollama"],
        default="gemini",
        help="LLM backend to use (default: gemini)",
    )
    parser.add_argument(
        "--model",
        help="Model name (default: gemini-2.0-flash for gemini, llama3 for ollama)",
    )

    args = parser.parse_args()

    # Resolve default model
    if args.model is None:
        args.model = {
            "gemini": "gemini-2.0-flash",
            "ollama": "llama3",
        }[args.backend]

    # Resolve default output path: strip _extracted/_extraced suffix, add -summary.md
    if args.output is None:
        base = os.path.splitext(args.input)[0]
        for suffix in ("_extraced", "_extracted"):
            if base.endswith(suffix):
                base = base[: -len(suffix)]
                break
        args.output = base + "-summary.md"

    # Read input
    text = read_input(args.input)
    print(f"Read {len(text)} characters from {args.input}")
    print(f"Using backend: {args.backend} / model: {args.model}")
    print(f"Output will be written to: {args.output}")

    # Summarize
    if args.backend == "gemini":
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            print(
                "Error: Set GEMINI_API_KEY environment variable.\n"
                "Get a free key at: https://aistudio.google.com/apikey",
                file=sys.stderr,
            )
            sys.exit(1)
        summary = summarize_gemini(text, args.model, api_key)
    else:
        summary = summarize_ollama(text, args.model)

    # Write output
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(summary)
    print(f"Summary written to {args.output}")


if __name__ == "__main__":
    main()