from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any


def build_examples_from_rows(rows: list[sqlite3.Row] | list[dict[str, Any]], channel_filter: str | None = None) -> list[dict[str, Any]]:
    """Build simple training examples grouped by channel.

    Each example is represented as a chat-like structure with a system instruction
    that scopes the reply style to the selected channel.
    """
    by_channel: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        channel = row["channel"] if isinstance(row, dict) else row["channel"]
        if channel_filter and channel != channel_filter:
            continue
        by_channel.setdefault(channel, []).append(
            {
                "role": row["role"] if isinstance(row, dict) else row["role"],
                "content": row["text"] if isinstance(row, dict) else row["text"],
                "nick": row["nick"] if isinstance(row, dict) else row["nick"],
            }
        )

    examples: list[dict[str, Any]] = []
    for channel, entries in sorted(by_channel.items()):
        for idx in range(0, len(entries) - 1):
            user_msg = entries[idx]
            assistant_msg = entries[idx + 1]
            if user_msg["role"] != "user" or assistant_msg["role"] != "assistant":
                continue
            examples.append(
                {
                    "channel": channel,
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                f"Odpowiadaj jak bot IRC na kanale {channel}. "
                                "Bądź krótki, naturalny, lekko ironiczny i dostosuj się do klimatu kanału."
                            ),
                        },
                        {"role": "user", "content": user_msg["content"]},
                        {"role": "assistant", "content": assistant_msg["content"]},
                    ],
                }
            )
    return examples


def export_examples_to_jsonl(examples: list[dict[str, Any]], output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example, ensure_ascii=False) + "\n")
    return output_path


def export_from_sqlite(db_path: str | Path, output_path: str | Path, channel_filter: str | None = None) -> Path:
    db_path = Path(db_path)
    output_path = Path(output_path)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT channel, role, text, nick FROM messages ORDER BY created_at, id"
    ).fetchall()
    conn.close()
    examples = build_examples_from_rows(rows, channel_filter=channel_filter)
    return export_examples_to_jsonl(examples, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Eksportuj wiadomości z SQLite do datasetu dla fine-tuningu")
    parser.add_argument("--db", default="./context.sqlite3", help="Ścieżka do bazy SQLite")
    parser.add_argument("--output", default="./fine_tune_dataset.jsonl", help="Plik wyjściowy JSONL")
    parser.add_argument("--channel", default=None, help="Opcjonalny filtr kanału, np. #bialystok")
    args = parser.parse_args()

    output_path = export_from_sqlite(args.db, args.output, channel_filter=args.channel)
    print(f"Zapisano {output_path}")


if __name__ == "__main__":
    main()
