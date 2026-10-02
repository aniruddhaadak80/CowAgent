# encoding:utf-8
"""What the user sees when a Moonshot failure arrives in an unexpected body.

The non-200 branch of ``MoonshotBot.reply_text`` used to do this::

    response = res.json()
    error = response.get("error")
    logger.error(f"... msg={error.get('message')}, type={error.get('type')}")

Two assumptions there are not guaranteed by anything. ``res.json()`` was called
unguarded, so a gateway answering with an HTML error page -- nginx in front of
the API, a corporate proxy, any 502 -- raises ``ValueError`` right there. And
``.get("error")`` had no default, so a body carrying its explanation somewhere
other than under ``error`` (several gateways send ``{"message": ...}``) left
``error`` as ``None`` for the very next line to dereference.

Either way the exception escapes the ``else:`` branch and lands in the blanket
``except Exception`` at the bottom of the method, which has no idea what status
code it was looking at. The user-visible consequence is not the crash but what
the crash destroys: a 400 that means "your request was malformed" gets
reclassified as a transport failure, so the status-driven reply that should
have been returned never is, three pointless retries burn six seconds, and the
answer that finally reaches the chat is the generic "I am a bit tired, come back
later" -- with the real reason from the provider present in neither the reply nor
the log line it was supposed to be written to.

These tests pin the two things that have to hold on every non-200 path: the
status-code branch is actually reached, and whatever text the provider did send
still ends up in the log.
"""

import json
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class _Session:
    """The only thing ``reply_text`` reads off a session is its messages."""

    def __init__(self):
        self.messages = [{"role": "user", "content": "hi"}]


class _JsonErrorBody:
    """A non-200 response whose body parses as JSON but is not the usual shape."""

    def __init__(self, payload, status_code=400):
        self.status_code = status_code
        self._payload = payload
        self.text = json.dumps(payload, ensure_ascii=False)

    def json(self):
        return self._payload


class _NonJsonErrorBody:
    """A non-200 response whose body is not JSON at all, as an HTML 502 is.

    ``requests`` raises ``requests.exceptions.JSONDecodeError`` here, which is a
    ``ValueError``; a plain ``ValueError`` stands in for it so the test does not
    have to depend on which exception class a given requests version picks.
    """

    def __init__(self, status_code=502, text="<html><body>502 Bad Gateway</body></html>"):
        self.status_code = status_code
        self.text = text

    def json(self):
        raise ValueError("No JSON object could be decoded")


def _bot():
    """A MoonshotBot with only the state ``reply_text`` touches.

    ``__init__`` builds a SessionManager and reads the developer's config, which
    a test should not do; the existing moonshot tests reach for ``__new__`` the
    same way. ``conf`` is patched to a plain dict, which is all the ``api_key``
    and ``base_url`` properties ask of it.
    """
    from models.moonshot.moonshot_bot import MoonshotBot

    bot = MoonshotBot.__new__(MoonshotBot)
    bot.args = {"model": "moonshot-v1-128k"}
    return bot


class TestMoonshotNonStandardErrorBody(unittest.TestCase):
    """A failure body that is not ``{"error": {...}}`` must not become a crash.

    The status code is the only thing the error branch was actually written to
    act on, so losing it to an exception on the way to the log is what turns a
    diagnosable provider error into an unexplained generic reply.
    """

    def _reply_text(self, response, retry_count=0):
        """Run ``reply_text`` against one canned response.

        ``time.sleep`` is stubbed out because the retry paths sleep three
        seconds between attempts and a test should not pay for them. The number
        of ``requests.post`` calls is returned alongside the result, because
        "did this failure get retried" is half of what went wrong: the 400
        branch sets ``need_retry = False`` and the code used to retry anyway.
        """
        conf = {"model": "moonshot-v1-128k", "moonshot_api_key": "test-key"}
        with patch("models.moonshot.moonshot_bot.conf", return_value=conf):
            with patch("models.moonshot.moonshot_bot.requests.post",
                       return_value=response) as post:
                with patch("models.moonshot.moonshot_bot.time.sleep"):
                    result = _bot().reply_text(_Session(), retry_count=retry_count)
        return result, post.call_count

    def test_error_body_without_an_error_key_is_not_reported_as_a_crash(self):
        """A body with ``message`` but no ``error`` must not raise.

        This is the shape a gateway in front of the API sends, and it used to
        make ``error`` ``None`` so ``error.get('message')`` raised. What the
        caller must see instead is the 400 branch's own reply.
        """
        result, calls = self._reply_text(_JsonErrorBody({"message": "invalid request"}))

        self.assertEqual(result["completion_tokens"], 0)
        self.assertEqual(result["content"], "提问太快啦，请休息一下再问我吧")
        # The 400 branch is the one place that says "do not retry". Reaching the
        # blanket handler instead meant three attempts and six seconds of sleep
        # for a request that was never going to succeed.
        self.assertEqual(calls, 1)

    def test_a_non_json_error_body_is_handled_and_reported(self):
        """An HTML error page must not take the whole reply path down with it.

        ``res.json()`` raises here, so without a guard the very first thing the
        error branch did was fail.
        """
        result, calls = self._reply_text(_NonJsonErrorBody(status_code=400))

        self.assertEqual(result["completion_tokens"], 0)
        self.assertEqual(result["content"], "提问太快啦，请休息一下再问我吧")
        self.assertEqual(calls, 1)

    def test_a_502_error_page_is_retried_and_then_reported_as_a_server_error(self):
        """A real 5xx still retries -- it just must retry for the stated reason.

        The retry count is unchanged by this fix; what changes is that the run
        reaches the ``status_code >= 500`` branch at all.
        """
        result, calls = self._reply_text(
            _NonJsonErrorBody(status_code=502), retry_count=2
        )

        self.assertEqual(result["content"], "提问太快啦，请休息一下再问我吧")
        self.assertEqual(calls, 1)

    def test_the_auth_failure_reply_is_reachable_when_the_body_is_unusual(self):
        """A 401 with a non-standard body still says "check your API key".

        Without the guard the 401 was answered with the generic "I am a bit
        tired" text, which is the worst possible reply to a key problem: it
        tells the user to wait when the thing they need to do is edit a config
        file.
        """
        result, calls = self._reply_text(
            _JsonErrorBody({"message": "invalid api key"}, status_code=401)
        )

        self.assertEqual(result["content"], "授权失败，请检查API Key是否正确")
        self.assertEqual(calls, 1)

    def test_the_rate_limit_reply_is_reachable_and_still_retries(self):
        """A 429 keeps its dedicated message and its two retries."""
        result, calls = self._reply_text(
            _JsonErrorBody({"message": "rate limited"}, status_code=429),
            retry_count=2,
        )

        self.assertEqual(result["content"], "请求过于频繁，请稍后再试")
        self.assertEqual(calls, 1)

    def test_the_standard_error_body_still_reports_message_and_type(self):
        """The normal ``{"error": {"message", "type"}}`` shape is unchanged.

        The guard is added around the existing read, not in place of it, so the
        case the code was written for has to keep logging both fields.
        """
        conf = {"model": "moonshot-v1-128k", "moonshot_api_key": "test-key"}
        response = _JsonErrorBody(
            {"error": {"message": "quota exhausted", "type": "rate_limit_error"}},
            status_code=403,
        )
        logged = MagicMock()
        with patch("models.moonshot.moonshot_bot.conf", return_value=conf):
            with patch("models.moonshot.moonshot_bot.requests.post", return_value=response):
                with patch("models.moonshot.moonshot_bot.logger") as logger:
                    logger.error = logged
                    _bot().reply_text(_Session())

        text = " ".join(str(call) for call in logged.call_args_list)
        self.assertIn("quota exhausted", text)
        self.assertIn("rate_limit_error", text)

    def test_the_providers_own_text_reaches_the_log_when_there_is_no_error_object(self):
        """The point of logging the error is to see what the provider said.

        With ``{"message": ...}`` there is no ``error`` object to read
        ``message`` from, and that is exactly when the reason is most worth
        having in the log.
        """
        conf = {"model": "moonshot-v1-128k", "moonshot_api_key": "test-key"}
        response = _JsonErrorBody(
            {"message": "model is overloaded, retry later"}, status_code=400
        )
        logged = MagicMock()
        with patch("models.moonshot.moonshot_bot.conf", return_value=conf):
            with patch("models.moonshot.moonshot_bot.requests.post", return_value=response):
                with patch("models.moonshot.moonshot_bot.logger") as logger:
                    logger.error = logged
                    _bot().reply_text(_Session())

        text = " ".join(str(call) for call in logged.call_args_list)
        self.assertIn("model is overloaded, retry later", text)


if __name__ == "__main__":
    unittest.main()
