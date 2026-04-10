"""Stream and interleave datasets without loading into memory.

Demonstrates: StreamingDataset, stream_jsonl(), stream_interleave().
"""

import json
import tempfile
from pathlib import Path

from datamix import StreamingDataset, stream_jsonl, stream_interleave

if __name__ == "__main__":
    # Create sample JSONL files for demonstration
    tmpdir = Path(tempfile.mkdtemp())
    chat_path = tmpdir / "chat.jsonl"
    code_path = tmpdir / "code.jsonl"

    chat_data = [
        {"instruction": "Explain photosynthesis.", "response": "Plants convert light to energy..."},
        {"instruction": "What is gravity?", "response": "Gravity is a fundamental force..."},
        {"instruction": "Define democracy.", "response": "Democracy is a system of government..."},
        {"instruction": "How do vaccines work?", "response": "Vaccines train the immune system..."},
    ]
    code_data = [
        {"instruction": "Write a Python fibonacci function.", "response": "def fib(n): ..."},
        {"instruction": "Sort a list in-place.", "response": "Use list.sort() or sorted()..."},
        {"instruction": "Reverse a string.", "response": "return s[::-1]"},
    ]

    chat_path.write_text("\n".join(json.dumps(d) for d in chat_data))
    code_path.write_text("\n".join(json.dumps(d) for d in code_data))

    # Stream from a JSONL file
    chat_ds = stream_jsonl(chat_path)
    print("=== Streaming Chat Dataset ===")
    for example in chat_ds.take(3):
        print(f"  Q: {example['instruction'][:50]}")

    # Use filter and map transforms (lazy, no extra memory)
    long_chat = chat_ds.filter(lambda x: len(x["response"]) > 30)
    print(f"\n=== Filtered (response > 30 chars) ===")
    for example in long_chat:
        print(f"  Q: {example['instruction'][:50]}  (resp len: {len(example['response'])})")

    # Interleave multiple datasets with weights
    code_ds = stream_jsonl(code_path)
    interleaved = stream_interleave(
        [stream_jsonl(chat_path), stream_jsonl(code_path)],
        weights=[0.7, 0.3],
        seed=42,
    )

    print(f"\n=== Interleaved Stream (70% chat, 30% code) ===")
    for i, example in enumerate(interleaved):
        print(f"  [{i}] {example['instruction'][:60]}")

    # Cleanup
    chat_path.unlink()
    code_path.unlink()
    tmpdir.rmdir()
