# encoding:utf-8
"""The wechatmp active-reply callback must acknowledge WeChat with "success".

``Query.POST`` ends in ``except Exception as exc: return exc``, so a failing
handler handed the ``Exception`` *object* back as the WSGI response body.
web.py serialises anything non-``bytes`` with ``str(r).encode("utf-8")``
(web/application.py:323-328), so WeChat did not even get an error status: it
received HTTP 200 carrying the exception text (``b"boom"``).

The WeChat server-config callback contract is that the body must be exactly
``success`` (or empty); anything else is read as "delivery failed", so WeChat
re-sends the identical callback. One inbound message is therefore handed to the
agent up to three times and three replies go out to the user.

Answering ``"success"`` on the error path is the acknowledgement WeChat asked
for; the failure itself is already recorded by ``logger.exception`` on the line
above, so nothing is lost by not shipping its text back over the wire.
"""

import io
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from channel.wechatmp import active_reply


class _FakeWebInput:
    """Stand-in for the ``web`` module that yields a signature check failure.

    The callback reads ``web.input()`` first and hands it to ``verify_server``,
    which is patched to raise, so nothing past that point has to be modelled.
    """

    class ctx:
        class env:
            @staticmethod
            def get(_key, _default=None):
                return None

    @staticmethod
    def input():
        return {"signature": "bad", "timestamp": "1", "nonce": "n"}

    @staticmethod
    def data():
        return b"<xml></xml>"


def _patched_query(reason):
    """Point the callback at a signature check that always fails.

    Returns the :class:`Query` handler ready to be driven through ``POST``.
    """
    real_web = active_reply.web
    real_verify = active_reply.verify_server

    def boom(_args):
        raise RuntimeError(reason)

    active_reply.web = _FakeWebInput()
    active_reply.verify_server = boom
    active_reply._restore = (real_web, real_verify)
    return active_reply.Query()


class WechatMPActiveCallbackErrorAckTest(unittest.TestCase):
    """A failed callback must still acknowledge WeChat with ``success``."""

    def tearDown(self):
        real_web, real_verify = getattr(active_reply, "_restore", (None, None))
        if real_web is not None:
            active_reply.web = real_web
            active_reply.verify_server = real_verify
            del active_reply._restore

    def test_error_path_acknowledges_with_success(self):
        handler = _patched_query("signature mismatch")

        body = handler.POST()

        self.assertEqual(
            "success", body,
            "the error path must answer 'success'; anything else makes WeChat "
            "resend the same callback and the message is processed repeatedly",
        )

    def test_error_path_does_not_return_the_exception_object(self):
        handler = _patched_query("signature mismatch")

        body = handler.POST()

        self.assertNotIsInstance(
            body, Exception,
            "returning the Exception object made web.py stringify it into the "
            "response body, so the exception text was shipped to WeChat",
        )

    def test_error_path_body_is_what_wechat_receives(self):
        """The bytes on the wire are the whole point, so assert them.

        Drives the real ``Query`` handler through a real web.py WSGI call and
        checks the exact body WeChat reads back.
        """
        import web

        _patched_query("signature mismatch")
        app = web.application()
        app.mapping = [("/active", active_reply.Query)]

        env = {
            "REQUEST_METHOD": "POST",
            "PATH_INFO": "/active",
            "SERVER_NAME": "example.invalid",
            "SERVER_PORT": "80",
            "wsgi.input": io.BytesIO(b"<xml></xml>"),
            "wsgi.url_scheme": "http",
            "QUERY_STRING": "",
            "HTTP_HOST": "example.invalid",
        }
        captured = {}

        def start_response(status, headers):
            captured["status"] = status

        payload = b"".join(app.wsgifunc()(env, start_response))

        self.assertEqual(b"success", payload)


if __name__ == "__main__":
    unittest.main()
