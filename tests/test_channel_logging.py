import sqlite3
import tempfile
import threading
import unittest
from datetime import datetime
from pathlib import Path

from brzydalek import IRCBot, SQLiteContextStore


class ChannelLoggingTests(unittest.TestCase):
    def _make_bot(self, tmpdir: str) -> tuple[IRCBot, SQLiteContextStore, Path]:
        db_path = Path(tmpdir) / "context.sqlite3"
        log_dir = Path(tmpdir) / "channel_logs"
        store = SQLiteContextStore(str(db_path), "brzydalek")
        bot = object.__new__(IRCBot)
        bot.nickname = "brzydalek"
        bot.context_store = store
        bot._channel_log_lock = threading.RLock()
        bot._channel_log_cfg = {"enabled": True, "directory": str(log_dir)}
        return bot, store, log_dir

    def test_store_channel_message_writes_full_channel_log_entry(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            bot, store, log_dir = self._make_bot(tmpdir)
            created_at = datetime(2026, 8, 11, 12, 34, 56).timestamp()

            bot._store_channel_message("#antysmuty", "alice", "hej wszystkim", created_at=created_at)

            log_path = log_dir / "antysmuty" / "2026-08-11.log"
            self.assertTrue(log_path.exists())
            self.assertEqual(
                log_path.read_text(encoding="utf-8"),
                "[12:34:56] <alice> hej wszystkim\n",
            )

            conn = sqlite3.connect(store.db_path)
            row = conn.execute(
                "SELECT channel, nick, role, text FROM messages ORDER BY id DESC LIMIT 1"
            ).fetchone()
            conn.close()
            self.assertEqual(row, ("#antysmuty", "alice", "user", "hej wszystkim"))
            store.close()

    def test_channel_log_name_is_sanitized(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            bot, store, log_dir = self._make_bot(tmpdir)
            created_at = datetime(2026, 8, 11, 7, 8, 9).timestamp()

            bot._store_channel_message("#anty/smuty test", "brzydalek", "siema", created_at=created_at)

            log_path = log_dir / "anty_smuty_test" / "2026-08-11.log"
            self.assertTrue(log_path.exists())
            self.assertEqual(log_path.read_text(encoding="utf-8"), "[07:08:09] <brzydalek> siema\n")
            store.close()


if __name__ == "__main__":
    unittest.main()