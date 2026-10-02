# encoding:utf-8
"""Whether the Xunfei TTS stream is willing to talk to a server it cannot identify.

``xunfei_tts`` dials ``wss://tts-api.xfyun.cn/v2/tts`` and then hands
``run_forever`` an ``sslopt`` of ``{"cert_reqs": ssl.CERT_NONE}``. websocket-client
builds its TLS context out of that dict, so the flag switches certificate checking
off for the entire stream. The session is still encrypted, but nothing proves the
far end is Xunfei: a certificate for any name at all completes the handshake and
the client accepts it.

That matters here because the credential is derived from the account secret. The
``authorization`` query parameter is an HMAC-SHA256 signature made with APISecret,
so an interceptor on the path -- a hostile hotspot, a hijacked router -- reads the
signed URL out of the request line and can replay it, and the base64 audio chunks
coming back the other way are just as editable. The mp3 the bot then writes to
disk and plays is whatever the interceptor chose to send, so a reply the user
hears can be a recording of somebody else's voice rather than the words that were
typed.

The fix is to stop asking for that. ``ssl.create_default_context()`` verifies the
chain and the hostname, and it is the same context every other TLS caller here
relies on -- ``app.run`` calls ``ensure_ca_bundle()`` before anything else, so a
packaged build without an OpenSSL trust store still has one.

No socket is opened. ``WebSocketApp`` is replaced by a recorder, so what the
module passes to ``run_forever`` is the whole of what is under test.
"""
import os
import ssl
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from voice.xunfei import xunfei_tts as xtts

APPID = "sentinel-appid-0001"
API_KEY = "sentinel-apikey-0002"
API_SECRET = "sentinel-apisecret-0003"


class _FakeApp(object):
    """Stands in for ``websocket.WebSocketApp`` and records how it is driven."""

    def __init__(self, url, on_message=None, on_error=None, on_close=None):
        self.url = url
        self.on_open = None
        self.runs = []

    def run_forever(self, **kwargs):
        # The real loop needs a live socket; returning at once keeps the test
        # offline. The callbacks are never reached, so nothing is written.
        self.runs.append(kwargs)


class _FakeWebsocket(object):
    """Just enough of the ``websocket`` module to reach ``run_forever``."""

    def __init__(self):
        self.app = None
        self.traced = []

    def enableTrace(self, flag):
        self.traced.append(flag)

    def WebSocketApp(self, url, **handlers):
        self.app = _FakeApp(url, **handlers)
        return self.app


class XunfeiTtsTlsVerificationTest(unittest.TestCase):
    def setUp(self):
        self.out_file = os.path.join(os.path.dirname(__file__), "sentinel-reply.mp3")
        self.fake = _FakeWebsocket()
        self.real = xtts.websocket
        xtts.websocket = self.fake
        # A real call, so the signed URL and the arguments are the ones the
        # bot actually sends.
        self.returned = xtts.xunfei_tts(APPID, API_KEY, API_SECRET, {}, "hello", self.out_file)
        self.run_kwargs = self.fake.app.runs[-1]

    def tearDown(self):
        xtts.websocket = self.real

    def test_the_stream_is_dialled_with_certificate_verification_on(self):
        """The context has to check the chain, not merely encrypt."""
        sslopt = self.run_kwargs.get("sslopt")
        self.assertTrue(
            isinstance(sslopt, dict),
            "run_forever got sslopt=%r; nothing here can be verified without it" % (sslopt,),
        )
        context = sslopt.get("context")
        self.assertIsInstance(
            context,
            ssl.SSLContext,
            "sslopt has no SSLContext, so websocket-client built its own -- one "
            "that trusts whatever answers",
        )
        self.assertEqual(
            context.verify_mode,
            ssl.CERT_REQUIRED,
            "the stream accepted an unverified certificate",
        )
        self.assertTrue(
            context.check_hostname,
            "a valid certificate for some other name was accepted, which is the "
            "same defect as accepting a forged one",
        )

    def test_the_escape_hatch_is_gone(self):
        """``cert_reqs: CERT_NONE`` must not reappear, here or on the socket."""
        sslopt = self.run_kwargs.get("sslopt") or {}
        self.assertNotEqual(
            sslopt.get("cert_reqs"),
            ssl.CERT_NONE,
            "the call still asks for an unverified socket",
        )
        with open(xtts.__file__, encoding="utf-8") as handle:
            source = handle.read()
        self.assertNotIn(
            "CERT_NONE",
            source,
            "the module still disables certificate verification somewhere",
        )

    def test_the_dialed_url_is_unchanged(self):
        """Secure default only: wss, and the signed auth params still sent."""
        url = self.fake.app.url
        self.assertTrue(
            url.startswith("wss://tts-api.xfyun.cn/v2/tts?"),
            "the TTS endpoint moved: %r" % (url,),
        )
        for param in ("authorization=", "date=", "host="):
            self.assertIn(param, url, "%s is missing from the signed URL" % param)

    def test_the_stream_bounds_and_the_return_value_survive(self):
        """The ping that ends a stalled stream, and the file it returns."""
        self.assertEqual(self.run_kwargs.get("ping_interval"), xtts.WS_PING_INTERVAL)
        self.assertEqual(self.run_kwargs.get("ping_timeout"), xtts.WS_PING_TIMEOUT)
        self.assertIsNone(
            xtts.stream_error,
            "a stream that never opened is not a failed stream, so nothing should be raised",
        )
        self.assertEqual(self.returned, self.out_file)


if __name__ == "__main__":
    unittest.main()
