# encoding:utf-8
"""Whether the Xunfei ASR stream is willing to talk to a server it cannot identify.

``xunfei_asr`` dials ``wss://ws-api.xfyun.cn/v2/iat`` and then hands
``run_forever`` an ``sslopt`` of ``{"cert_reqs": ssl.CERT_NONE}``. websocket-client
builds its TLS context out of that dict, so the flag switches certificate checking
off for the entire stream. The session is still encrypted, but nothing proves
the far end is Xunfei: a certificate for any name at all completes the handshake
and the client accepts it.

That matters more here than on an ordinary call, because the credential rides in
the query string. ``authorization`` is the base64 of the api_key together with an
HMAC computed from APISecret, so an interceptor on the path -- a hostile hotspot,
a hijacked router -- reads the API key out of the request line and reuses the
application for as long as the vendor keeps it alive. The base64 PCM frames
coming back the other way are just as editable, and the ``result`` the bot
extracts from them is the text it acts on: a voice command that moves a file,
spends money or contacts a person can be replaced wholesale with something the
user never said.

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

from voice.xunfei import xunfei_asr as xasr

APPID = "sentinel-appid-0001"
API_KEY = "sentinel-apikey-0002"
API_SECRET = "sentinel-apisecret-0003"

# Never opened: the audio file is read from the ``on_open`` callback, which only a
# real WebSocketApp fires.
AUDIO_FILE = os.path.join(os.path.dirname(__file__), "no-such-file.wav")


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


class XunfeiAsrTlsVerificationTest(unittest.TestCase):
    def setUp(self):
        self.fake = _FakeWebsocket()
        self.real = xasr.websocket
        xasr.websocket = self.fake
        # A real call, so the signed URL and the arguments are the ones the
        # bot actually sends.
        xasr.xunfei_asr(APPID, API_SECRET, API_KEY, {}, AUDIO_FILE)
        self.run_kwargs = self.fake.app.runs[-1]

    def tearDown(self):
        xasr.websocket = self.real

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
        with open(xasr.__file__, encoding="utf-8") as handle:
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
            url.startswith("wss://ws-api.xfyun.cn/v2/iat?"),
            "the ASR endpoint moved: %r" % (url,),
        )
        for param in ("authorization=", "date=", "host="):
            self.assertIn(param, url, "%s is missing from the signed URL" % param)
        # Nothing arrived, so the transcription is empty rather than garbage.
        self.assertEqual(xasr.whole_dict, {})


if __name__ == "__main__":
    unittest.main()
