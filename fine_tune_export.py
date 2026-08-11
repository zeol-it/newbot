from __future__ import annotations

import argparse
import json
import re
import sqlite3
from pathlib import Path
from typing import Any


def _normalize_row(row: sqlite3.Row | dict[str, Any]) -> dict[str, Any]:
    return {
        "id": row["id"],
        "channel": row["channel"],
        "scope": row["scope"],
        "conversation_key": row["conversation_key"],
        "nick": row["nick"],
        "role": row["role"],
        "text": row["text"],
        "created_at": row["created_at"],
    }


def _default_channel_system_prompt(channel: str) -> str:
    return (
        f"Jesteś botem IRC na kanale {channel}. "
        "Odpowiadasz krótko, naturalnie, po polsku i w stylu dopasowanym do klimatu kanału. "
        "Uwzględniaj lokalny kontekst rozmowy i nicki uczestników, jeśli są istotne dla odpowiedzi."
    )


def _load_prompt_template(config_path: str | Path | None) -> str | None:
    if not config_path:
        return None
    config_file = Path(config_path)
    if not config_file.exists():
        return None
    config = json.loads(config_file.read_text(encoding="utf-8"))
    return config.get("fine_tune_prompt")


def _channel_system_prompt(channel: str, prompt_template: str | None = None) -> str:
    if prompt_template:
        return prompt_template.format(channel=channel)
    return _default_channel_system_prompt(channel)


def _format_channel_line(row: dict[str, Any]) -> str:
    return f"{row['nick']}: {row['text']}"


def _format_conversation_message(row: dict[str, Any]) -> dict[str, str]:
    if row["role"] == "assistant":
        return {"role": "assistant", "content": row["text"]}
    return {"role": "user", "content": f"{row['nick']}: {row['text']}"}


def _canonicalize_text(text: str) -> str:
    normalized = text.strip()
    normalized = re.sub(r"^[^\s:]{1,32}\s*[:,\-]\s*", "", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip().lower()


def _dedupe_rows(rows: list[dict[str, Any]], *, prefer: str) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    for row in rows:
        if deduped:
            previous = deduped[-1]
            same_event = (
                previous["nick"] == row["nick"]
                and previous["role"] == row["role"]
                and abs(float(previous["created_at"]) - float(row["created_at"])) <= 2.0
                and _canonicalize_text(previous["text"]) == _canonicalize_text(row["text"])
            )
            if same_event:
                previous_len = len(previous["text"].strip())
                row_len = len(row["text"].strip())
                if prefer == "shorter" and row_len < previous_len:
                    deduped[-1] = row
                elif prefer == "longer" and row_len > previous_len:
                    deduped[-1] = row
                continue
        deduped.append(row)
    return deduped


def build_examples_from_rows(
    rows: list[sqlite3.Row] | list[dict[str, Any]],
    channel_filter: str | None = None,
    channel_context_limit: int = 6,
    conversation_context_limit: int = 4,
    prompt_template: str | None = None,
) -> list[dict[str, Any]]:
    """Build training examples anchored on assistant replies.

    Each example uses one assistant message as the target, includes recent same-
    conversation turns for that reply, and adds nearby channel context from other
    participants to preserve per-channel tone.
    """
    normalized_rows = [_normalize_row(row) for row in rows]
    filtered_rows = [
        row for row in normalized_rows
        if (not channel_filter or row["channel"] == channel_filter)
    ]

    examples: list[dict[str, Any]] = []
    for idx, row in enumerate(filtered_rows):
        if row["role"] != "assistant":
            continue

        conversation_rows = [
            previous for previous in filtered_rows[:idx]
            if previous["channel"] == row["channel"]
            and previous["conversation_key"] == row["conversation_key"]
        ]
        if not conversation_rows:
            continue

        conversation_rows = conversation_rows[-conversation_context_limit:]
        conversation_rows = _dedupe_rows(conversation_rows, prefer="shorter")
        if not any(previous["role"] == "user" for previous in conversation_rows):
            continue

        if conversation_rows[-1]["role"] != "user":
            continue

        channel_rows = [
            previous for previous in filtered_rows[:idx]
            if previous["channel"] == row["channel"]
            and previous["conversation_key"] != row["conversation_key"]
            and not (
                previous["role"] == row["role"]
                and previous["nick"] == row["nick"]
                and abs(float(previous["created_at"]) - float(row["created_at"])) <= 2.0
                and _canonicalize_text(previous["text"]) == _canonicalize_text(row["text"])
            )
        ]
        channel_rows = channel_rows[-channel_context_limit:]
        channel_rows = _dedupe_rows(channel_rows, prefer="longer")

        messages: list[dict[str, str]] = [
            {"role": "system", "content": _channel_system_prompt(row["channel"], prompt_template=prompt_template)}
        ]
        if channel_rows:
            messages.append(
                {
                    "role": "system",
                    "content": (
                        f"Ostatni kontekst kanału {row['channel']} od innych osób:\n"
                        + "\n".join(_format_channel_line(previous) for previous in channel_rows)
                    ),
                }
            )
        messages.extend(_format_conversation_message(previous) for previous in conversation_rows)
        messages.append({"role": "assistant", "content": row["text"]})

        examples.append(
            {
                "channel": row["channel"],
                "conversation_key": row["conversation_key"],
                "messages": messages,
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


def export_from_sqlite(
    db_path: str | Path,
    output_path: str | Path,
    channel_filter: str | None = None,
    channel_context_limit: int = 6,
    conversation_context_limit: int = 4,
    config_path: str | Path | None = None,
) -> Path:
    db_path = Path(db_path)
    output_path = Path(output_path)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT id, channel, scope, conversation_key, role, text, nick, created_at "
        "FROM messages ORDER BY created_at, id"
    ).fetchall()
    conn.close()
    prompt_template = _load_prompt_template(config_path)
    examples = build_examples_from_rows(
        rows,
        channel_filter=channel_filter,
        channel_context_limit=channel_context_limit,
        conversation_context_limit=conversation_context_limit,
        prompt_template=prompt_template,
    )
    return export_examples_to_jsonl(examples, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Eksportuj wiadomości z SQLite do datasetu dla fine-tuningu")
    parser.add_argument("--db", default="./context.sqlite3", help="Ścieżka do bazy SQLite")
    parser.add_argument("--config", default="./bot_config.json", help="Plik konfiguracyjny z fine_tune_prompt")
    parser.add_argument("--output", default="./fine_tune_dataset.jsonl", help="Plik wyjściowy JSONL")
    parser.add_argument("--channel", default=None, help="Opcjonalny filtr kanału, np. #bialystok")
    parser.add_argument("--channel-context-limit", type=int, default=6, help="Liczba wcześniejszych wiadomości kanałowych od innych osób")
    parser.add_argument("--conversation-context-limit", type=int, default=4, help="Liczba wcześniejszych wiadomości z tej samej rozmowy")
    args = parser.parse_args()

    output_path = export_from_sqlite(
        args.db,
        args.output,
        channel_filter=args.channel,
        channel_context_limit=args.channel_context_limit,
        conversation_context_limit=args.conversation_context_limit,
        config_path=args.config,
    )
    print(f"Zapisano {output_path}")


if __name__ == "__main__":
    main()
