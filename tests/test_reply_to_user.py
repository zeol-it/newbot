import unittest

from brzydalek import IRCBot


class ReplyToUserTests(unittest.TestCase):
    def _make_bot(self, reply_to_user: bool) -> IRCBot:
        bot = object.__new__(IRCBot)
        bot.nickname = "brzydalek"
        bot.reply_to_user = reply_to_user
        return bot

    def test_prefix_is_added_when_reply_to_user_enabled(self) -> None:
        bot = self._make_bot(True)

        self.assertEqual(bot._format_reply_chunk("alice", "hello", is_private=False, is_first_chunk=True), "alice: hello")

    def test_prefix_is_omitted_when_reply_to_user_disabled(self) -> None:
        bot = self._make_bot(False)

        self.assertEqual(bot._format_reply_chunk("alice", "hello", is_private=False, is_first_chunk=True), "hello")


if __name__ == "__main__":
    unittest.main()
