import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from fine_tune_export import build_examples_from_rows, export_examples_to_jsonl


class FineTuneExportTests(unittest.TestCase):
    def _make_db(self, path: Path) -> None:
        conn = sqlite3.connect(path)
        conn.execute(
            """
            CREATE TABLE messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                channel TEXT NOT NULL,
                scope TEXT NOT NULL,
                conversation_key TEXT NOT NULL,
                nick TEXT NOT NULL,
                role TEXT NOT NULL,
                text TEXT NOT NULL,
                created_at REAL NOT NULL
            )
            """
        )
        conn.executemany(
            "INSERT INTO messages (channel, scope, conversation_key, nick, role, text, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                ("#test", "channel", "c1", "alice", "user", "hello", 1.0),
                ("#test", "channel", "c1", "brzydalek", "assistant", "hi there", 2.0),
                ("#test", "channel", "c1", "bob", "user", "ok", 3.0),
                ("#test", "channel", "c1", "brzydalek", "assistant", "fine", 4.0),
                ("#other", "channel", "c2", "alice", "user", "x", 5.0),
            ],
        )
        conn.commit()
        conn.close()

    def test_build_examples_from_rows_creates_channel_specific_examples(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.sqlite3"
            self._make_db(db_path)
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT channel, role, text, nick FROM messages ORDER BY created_at, id"
            ).fetchall()
            conn.close()

            examples = build_examples_from_rows(rows, channel_filter="#test")

            self.assertEqual(len(examples), 2)
            self.assertEqual(examples[0]["messages"][0]["role"], "system")
            self.assertIn("#test", examples[0]["messages"][0]["content"])
            self.assertEqual(examples[0]["messages"][1]["content"], "hello")
            self.assertEqual(examples[0]["messages"][2]["content"], "hi there")

    def test_export_examples_to_jsonl_writes_json_lines(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "dataset.jsonl"
            examples = [{"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]}]
            export_examples_to_jsonl(examples, output_path)

            data = output_path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(data), 1)
            self.assertEqual(json.loads(data[0])["messages"][1]["content"], "hello")


if __name__ == "__main__":
    unittest.main()
