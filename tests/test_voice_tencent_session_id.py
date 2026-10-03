# encoding:utf-8
"""
Unit tests for the SessionId that ``TencentVoice.textToVoice`` sends.

The id was derived from a whole-second timestamp::

    req.SessionId = str(int(time.time()))

``int()`` truncates to the second, so every TTS request issued inside the same
wall-clock second carried an identical SessionId while carrying different text.
Tencent treats SessionId as the identity of the synthesis session, so the second
request of such a pair is answered as a repeat of the first: it is rejected or
conflated with it, and that caller gets no audio back.

Nothing in the method rate-limits, so the collision needs nothing exotic -- a
gateway fanning out one reply into two voice messages, a scheduler reading two
rows in a batch, or two users served by the same event loop are all enough.

The clock here advances inside a single second on purpose. A clock frozen on one
exact value would pass just as well with a per-second id, because there would be
nothing to collide; what has to be shown is that two ids minted 800ms apart, in
the same second, are still distinguishable.

``tencentcloud`` is an optional extra, so it is stubbed the way the other
optional-dependency tests in this suite do.
"""
import os
import sys
import types
import unittest
import unittest.mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _module(name, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[name] = module
    if "." in name:
        parent, _, leaf = name.rpartition(".")
        setattr(sys.modules[parent], leaf, module)
    return module


class _Request:
    """Stands in for the SDK's request objects: attributes are assigned freely."""


class _Credential:
    def __init__(self, secret_id, secret_key):
        self.secret_id = secret_id
        self.secret_key = secret_key


_SESSION_IDS = []


class _TtsClient:
    """Records the SessionId of every request it is handed."""

    def __init__(self, cred, region):
        self.cred = cred
        self.region = region

    def TextToVoice(self, req):
        _SESSION_IDS.append(req.SessionId)
        return types.SimpleNamespace(Audio=None)


class _AsrClient:
    def __init__(self, cred, region):
        self.cred = cred
        self.region = region


_module("tencentcloud")
_module("tencentcloud.common", credential=types.SimpleNamespace(Credential=_Credential))
_module("tencentcloud.asr")
_module("tencentcloud.asr.v20190614",
        asr_client=types.SimpleNamespace(AsrClient=_AsrClient),
        models=types.SimpleNamespace(SentenceRecognitionRequest=_Request))
_module("tencentcloud.tts")
_module("tencentcloud.tts.v20190823",
        tts_client=types.SimpleNamespace(TtsClient=_TtsClient),
        models=types.SimpleNamespace(TextToVoiceRequest=_Request))

from bridge.reply import ReplyType  # noqa: E402
from voice.tencent import tencent_voice as tv  # noqa: E402


class _ClockWithinOneSecond:
    """A clock that hands out sub-second offsets of one fixed second.

    Every value truncates to the same integer second, so ``int(time.time())`` is
    constant across the whole run -- which is exactly the condition the defect
    needs, and what a fully frozen clock could not express.
    """

    def __init__(self, base=1712345678.0, offsets=(0.1, 0.9, 0.3, 0.7, 0.5, 0.2)):
        self.base = base
        self.offsets = list(offsets)
        self.index = 0
        self.readings = []

    def time(self):
        value = self.base + self.offsets[self.index % len(self.offsets)]
        self.index += 1
        self.readings.append(value)
        return value


def _voice():
    """A TencentVoice without running __init__: the shipped config.json is a
    template, so the real constructor only logs that the credentials are absent."""
    voice = tv.TencentVoice.__new__(tv.TencentVoice)
    voice.secret_id = "secret-id"
    voice.secret_key = "secret-key"
    voice.voice_type = 1003
    return voice


class TestTencentSessionIdResolution(unittest.TestCase):
    def setUp(self):
        del _SESSION_IDS[:]

    def test_two_requests_in_the_same_second_get_different_session_ids(self):
        """The defect: both requests carried the same per-second id."""
        clock = _ClockWithinOneSecond()

        with unittest.mock.patch.object(tv, "time", clock):
            _voice().textToVoice("第一条")
            _voice().textToVoice("第二条")

        self.assertEqual(len(_SESSION_IDS), 2)
        # The premise, asserted: both readings really were in the same second.
        self.assertEqual(int(clock.readings[0]), int(clock.readings[1]))
        self.assertNotEqual(
            _SESSION_IDS[0], _SESSION_IDS[1],
            "two requests {}ms apart shared SessionId {}".format(
                int((clock.readings[1] - clock.readings[0]) * 1000), _SESSION_IDS[0]),
        )

    def test_a_synthesis_still_returns_a_voice_reply(self):
        """The control: the id change must not disturb the working path."""

        class _Client(_TtsClient):
            def TextToVoice(self, req):
                _SESSION_IDS.append(req.SessionId)
                return types.SimpleNamespace(Audio="QUJD")

        clock = _ClockWithinOneSecond()
        with unittest.mock.patch.object(tv, "time", clock):
            with unittest.mock.patch.object(tv.tts_client, "TtsClient", _Client):
                reply = _voice().textToVoice("你好")

        self.assertEqual(reply.type, ReplyType.VOICE)
        self.assertTrue(reply.content.endswith(".mp3"))
        self.assertTrue(os.path.exists(reply.content))
        os.remove(reply.content)

    def test_a_failed_synthesis_still_returns_an_error_reply(self):
        """The control for the error path, which must stay an error reply."""
        clock = _ClockWithinOneSecond()
        with unittest.mock.patch.object(tv, "time", clock):
            reply = _voice().textToVoice("你好")

        self.assertEqual(reply.type, ReplyType.ERROR)


if __name__ == "__main__":
    unittest.main()
