# encoding:utf-8
"""Two web-console sessions must not overwrite each other's first user_datas write.

``Config.get_user_data`` lazily creates the per-user dict and hands the caller a
reference to keep writing into -- the chat channel stores a user's
``openai_api_key`` and ``gpt_model`` through it on the first inbound message,
and ``plugins/godcmd`` reads and writes the same dict. It did that with a
get-then-set on one dict shared by every session thread::

    if self.user_datas.get(user) is None:
        self.user_datas[user] = {}

Nothing makes those two steps one step, so two sessions opening the same
brand-new username at the same time both read ``None``, each built its own
``{}``, and the second ``__setitem__`` replaced the first. The first session's
dict was then unreachable from ``user_datas`` while it kept writing into it, so
everything it stored -- including the api key it had just saved -- was dropped
on the next ``save_user_datas``, with no error anywhere.

The interleaving is a rare accident of timing, so the first test below forces it
rather than hoping for it: a ``user_datas`` whose ``get`` releases both threads
at the point where the old code had already decided to create a dict. The
remaining tests pin the behaviour that had to survive the fix.
"""

import os
import sys
import threading
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import config as config_module


class _LosingUserDatas(dict):
    """``user_datas`` that forces the interleaving which loses a write.

    Two things have to line up for the loss, and neither happens on its own:
    both callers must read ``None`` before either stores, *and* the second store
    must land after the first caller has already been handed its dict. Ordering
    the callers that way is what turns a rare race into a repeatable one.

    Only ``get`` and ``__setitem__`` synchronise, and a fixed ``get_user_data``
    calls neither -- ``dict.setdefault`` does its lookup and insert in C without
    dispatching back to either -- so with the fix in place this class is inert
    and the test runs at full speed.
    """

    def __init__(self):
        super().__init__()
        self._lock = threading.Lock()
        self._arrivals = 0
        self._both_arrived = threading.Barrier(2, timeout=10)
        self._other_stored = threading.Event()

    def get(self, key, default=None):
        value = super().get(key, default)
        with self._lock:
            self._arrivals += 1
            first_arrival = self._arrivals == 1
        self._both_arrived.wait()
        if first_arrival:
            # Let the other caller store its dict first, so that ours is the one
            # replaced underneath us -- after we have been handed it.
            self._other_stored.wait(timeout=10)
        return value

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        # Only the second caller can reach this while the first one waits above.
        self._other_stored.set()


class GetUserDataAtomicTest(unittest.TestCase):
    """``get_user_data`` hands every caller for a user the same dict."""

    def _run_concurrently(self, conf, users, body):
        """Run ``body(user)`` on one thread per entry of ``users``, and join."""
        errors = []

        def worker(user):
            try:
                body(user)
            except BaseException as e:  # surfaced in the main thread
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(u,)) for u in users]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=15)
        self.assertEqual([], errors, "worker thread raised")

    def test_two_sessions_opening_the_same_new_user_keep_each_others_writes(self):
        """The interleaving the old code loses, forced rather than raced for.

        The two writes are the ones the chat channel makes on a user's first
        inbound message: ``openai_api_key`` and ``gpt_model`` from the same
        ``get_user_data`` reference. When the second session's dict replaces the
        first, one of the two is written into a dict ``user_datas`` no longer
        holds, so ``save_user_datas`` persists a user who is missing a key the
        console just saved for them.
        """
        conf = config_module.Config()
        conf.user_datas = _LosingUserDatas()

        def first_message(session, field):
            data = conf.get_user_data("alice")
            data[field] = "written-by-" + session

        threads = [
            threading.Thread(target=first_message, args=("first", "openai_api_key")),
            threading.Thread(target=first_message, args=("second", "gpt_model")),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=15)
        self.assertEqual([], [t for t in threads if t.is_alive()], "thread hung")

        stored = conf.user_datas["alice"]
        self.assertEqual("written-by-first", stored.get("openai_api_key"))
        self.assertEqual("written-by-second", stored.get("gpt_model"))

    def test_many_threads_on_one_new_user_share_a_single_dict(self):
        """No thread may receive a dict that ``user_datas`` does not hold."""
        conf = config_module.Config()
        users = ["bob"] * 16
        handed_out = []
        collected = threading.Lock()

        def first_message(user):
            data = conf.get_user_data(user)
            data["seen"] = data.get("seen", 0) + 1
            with collected:
                handed_out.append(data)

        self._run_concurrently(conf, users, first_message)

        self.assertEqual(16, len(handed_out))
        for data in handed_out:
            self.assertIs(data, conf.user_datas["bob"])
        self.assertEqual(16, conf.user_datas["bob"]["seen"])

    def test_distinct_users_still_get_distinct_dicts(self):
        """The fix must not collapse every user onto one shared dict."""
        conf = config_module.Config()

        self.assertIsNot(conf.get_user_data("carol"), conf.get_user_data("dave"))

    def test_an_existing_entry_is_returned_as_is(self):
        """A user who already has data keeps the very same dict and its contents."""
        conf = config_module.Config()
        stored = {"openai_api_key": "sk-existing", "gpt_model": "gpt-4o"}
        conf.user_datas["erin"] = stored

        self.assertIs(stored, conf.get_user_data("erin"))
        self.assertEqual("sk-existing", conf.get_user_data("erin")["openai_api_key"])

    def test_a_stored_none_is_replaced_with_a_dict(self):
        """The guarantee the old ``is None`` check gave is kept.

        ``user_datas`` is restored from a pickle, so a ``None`` can be sitting in
        it; callers are documented to receive a dict and would raise on
        ``user_data.get(...)`` if they got ``None`` back.
        """
        conf = config_module.Config()
        conf.user_datas["frank"] = None

        self.assertEqual({}, conf.get_user_data("frank"))


if __name__ == "__main__":
    unittest.main()
