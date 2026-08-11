import sqlite3
import tempfile
import threading
import unittest
from datetime import date, datetime
from pathlib import Path
from typing import Optional

from brzydalek import IRCBot, SQLiteContextStore


class _FakeSpontaneousStore:
    def __init__(self, already_sent: bool = False) -> None:
        self.already_sent = already_sent
        self.recorded: list[tuple[str, str]] = []
        self.messages: list[dict[str, object]] = []

    def has_spontaneous_message_since(self, channel: str, text: str, since_timestamp: float) -> bool:
        return self.already_sent

    def add_spontaneous_message(self, channel: str, text: str, created_at: Optional[float] = None) -> None:
        self.recorded.append((channel, text))

    def add_message(self, **kwargs) -> None:
        self.messages.append(kwargs)


class MidnightAnnouncementTests(unittest.TestCase):
    def _make_bot(self, store: _FakeSpontaneousStore) -> IRCBot:
        bot = object.__new__(IRCBot)
        bot._midnight_cfg = {
            "enabled": True,
            "channel": "#antysmuty",
            "text": "1st",
        }
        bot._midnight_last_sent_day = None
        bot.context_store = store
        bot.sent_messages: list[str] = []
        bot.nickname = "brzydalek"
        bot._channel_log_lock = threading.RLock()
        bot._channel_log_cfg = {"enabled": False, "directory": "./channel_logs"}
        bot.send = lambda message: bot.sent_messages.append(message)
        return bot

    def test_next_local_midnight_advances_to_next_day(self) -> None:
        current = datetime(2026, 8, 8, 23, 59, 59, 123456)

        self.assertEqual(
            IRCBot._next_local_midnight(current),
            datetime(2026, 8, 9, 0, 0, 0),
        )

    def test_should_send_midnight_announcement_requires_exact_midnight(self) -> None:
        bot = self._make_bot(_FakeSpontaneousStore())
        target_day = date(2026, 8, 9)

        self.assertFalse(
            bot._should_send_midnight_announcement(target_day, now=datetime(2026, 8, 9, 0, 0, 1))
        )
        self.assertTrue(
            bot._should_send_midnight_announcement(target_day, now=datetime(2026, 8, 9, 0, 0, 0))
        )

    def test_should_send_midnight_announcement_skips_when_already_recorded(self) -> None:
        bot = self._make_bot(_FakeSpontaneousStore(already_sent=True))
        target_day = date(2026, 8, 9)

        self.assertFalse(
            bot._should_send_midnight_announcement(target_day, now=datetime(2026, 8, 9, 0, 0, 0))
        )
        self.assertEqual(bot._midnight_last_sent_day, target_day)

    def test_send_midnight_announcement_sends_and_records_message(self) -> None:
        store = _FakeSpontaneousStore()
        bot = self._make_bot(store)
        target_day = date(2026, 8, 9)

        bot._send_midnight_announcement(target_day)

        self.assertEqual(bot.sent_messages, ["PRIVMSG #antysmuty :1st"])
        self.assertEqual(store.recorded, [("#antysmuty", "1st")])
        self.assertEqual(len(store.messages), 1)
        self.assertEqual(store.messages[0]["channel"], "#antysmuty")
        self.assertEqual(store.messages[0]["nick"], "brzydalek")
        self.assertEqual(store.messages[0]["text"], "1st")
        self.assertEqual(bot._midnight_last_sent_day, target_day)


    def test_context_store_marks_message_as_recent(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "context.sqlite3"
            store = SQLiteContextStore(str(db_path), "brzydalek")
            start_of_day = datetime(2026, 8, 9, 0, 0, 0).timestamp()
            store.add_spontaneous_message("#antysmuty", "1st", created_at=start_of_day + 10)

            self.assertTrue(store.has_spontaneous_message_since("#antysmuty", "1st", start_of_day))
            self.assertFalse(store.has_spontaneous_message_since("#antysmuty", "1st", start_of_day + 20))
            store.close()


if __name__ == "__main__":
    unittest.main()
