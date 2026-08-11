import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from fine_tune_export import build_examples_from_rows, export_examples_to_jsonl, export_from_sqlite


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
                ("#test", "channel", "ambient-1", "marek", "user", "ale tu duszno", 1.0),
                ("#test", "channel", "c1", "alice", "user", "hello", 2.0),
                ("#test", "channel", "ambient-2", "bob", "user", "wyjdz na balkon", 3.0),
                ("#test", "channel", "c1", "brzydalek", "assistant", "hi there", 4.0),
                ("#test", "channel", "c1", "alice", "user", "still there?", 5.0),
                ("#test", "channel", "c1", "brzydalek", "assistant", "fine", 6.0),
                ("#other", "channel", "c2", "alice", "user", "x", 7.0),
            ],
        )
        conn.commit()
        conn.close()

    def test_build_examples_from_rows_creates_examples_with_channel_and_conversation_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.sqlite3"
            self._make_db(db_path)
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT id, channel, scope, conversation_key, role, text, nick, created_at FROM messages ORDER BY created_at, id"
            ).fetchall()
            conn.close()

            examples = build_examples_from_rows(rows, channel_filter="#test")

            self.assertEqual(len(examples), 2)
            self.assertEqual(examples[0]["messages"][0]["role"], "system")
            self.assertIn("#test", examples[0]["messages"][0]["content"])
            self.assertEqual(examples[0]["messages"][1]["role"], "system")
            self.assertIn("marek: ale tu duszno", examples[0]["messages"][1]["content"])
            self.assertIn("bob: wyjdz na balkon", examples[0]["messages"][1]["content"])
            self.assertEqual(examples[0]["messages"][2], {"role": "user", "content": "alice: hello"})
            self.assertEqual(examples[0]["messages"][3], {"role": "assistant", "content": "hi there"})

    def test_build_examples_from_rows_uses_fine_tune_prompt_template(self) -> None:
        rows = [
            {
                "id": 1,
                "channel": "#test",
                "scope": "channel",
                "conversation_key": "c1",
                "nick": "alice",
                "role": "user",
                "text": "hello",
                "created_at": 1.0,
            },
            {
                "id": 2,
                "channel": "#test",
                "scope": "channel",
                "conversation_key": "c1",
                "nick": "brzydalek",
                "role": "assistant",
                "text": "hi there",
                "created_at": 2.0,
            },
        ]

        examples = build_examples_from_rows(
            rows,
            channel_filter="#test",
            prompt_template="Styl kanału {channel}. Odpowiadaj krótko.",
        )

        self.assertEqual(examples[0]["messages"][0]["content"], "Styl kanału #test. Odpowiadaj krótko.")

    def test_build_examples_from_rows_skips_assistant_without_direct_user_prompt(self) -> None:
        rows = [
            {
                "id": 1,
                "channel": "#test",
                "scope": "channel",
                "conversation_key": "ambient",
                "nick": "brzydalek",
                "role": "assistant",
                "text": "sam z siebie",
                "created_at": 1.0,
            }
        ]

        self.assertEqual(build_examples_from_rows(rows, channel_filter="#test"), [])

    def test_export_examples_to_jsonl_writes_json_lines(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "dataset.jsonl"
            examples = [{"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]}]
            export_examples_to_jsonl(examples, output_path)

            data = output_path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(data), 1)
            self.assertEqual(json.loads(data[0])["messages"][1]["content"], "hello")

    def test_export_from_sqlite_passes_context_limits(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.sqlite3"
            output_path = Path(tmpdir) / "dataset.jsonl"
            self._make_db(db_path)

            export_from_sqlite(
                db_path,
                output_path,
                channel_filter="#test",
                channel_context_limit=1,
                conversation_context_limit=1,
            )

            rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(len(rows), 2)
            self.assertIn("bob: wyjdz na balkon", rows[0]["messages"][1]["content"])
            self.assertEqual(rows[1]["messages"][2], {"role": "user", "content": "alice: still there?"})

    def test_export_from_sqlite_reads_fine_tune_prompt_from_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.sqlite3"
            output_path = Path(tmpdir) / "dataset.jsonl"
            config_path = Path(tmpdir) / "bot_config.json"
            self._make_db(db_path)
            config_path.write_text(
                json.dumps({"fine_tune_prompt": "Kanał {channel}. Mów krótko."}),
                encoding="utf-8",
            )

            export_from_sqlite(
                db_path,
                output_path,
                channel_filter="#test",
                config_path=config_path,
            )

            rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(rows[0]["messages"][0]["content"], "Kanał #test. Mów krótko.")


if __name__ == "__main__":
    unittest.main()
