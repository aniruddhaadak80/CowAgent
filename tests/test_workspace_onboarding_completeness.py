# encoding:utf-8
"""Onboarding is finished when *both* identity files are filled in, not one.

``BOOTSTRAP.md`` states the rule it is written to enforce: the agent deletes the
file "when the core fields of AGENT.md **and** USER.md are filled in". It is the
onboarding script -- the thing that tells the agent to ask the remaining
questions -- so getting the "is it done?" answer wrong is not cosmetic.

``_is_onboarding_done`` (``agent/prompt/workspace.py``) is the backstop for an
agent that filled the files and forgot to delete the script, and
``load_context_files`` calls it to decide whether to delete BOOTSTRAP.md and
skip loading it. It walked ``_ONBOARDING_PLACEHOLDERS`` and returned True from
inside the loop on the *first* file whose placeholder was gone -- so one filled
file declared the whole onboarding complete, and the order the dict happened to
be written in decided which file counted.

The consequence is a workspace that silently stops onboarding. A user who
answers only "call yourself X" gets that one field written to AGENT.md; on the
next turn the backstop sees AGENT.md is filled, deletes BOOTSTRAP.md, and the
questions that would have filled USER.md -- what to call the user, their name,
the conversational style -- are never asked. USER.md stays a template for the
life of the workspace, and there is nothing left in it to ask.

These pin the rule as written: every identity file that is present has to have
had its placeholder filled before the onboarding script is retired.
"""

import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


AGENT_TEMPLATE_AGENT = """# AGENT.md

- **名字**: *(在首次对话时填写，可以是用户给你起的名字)*
- **角色**: *(AI助理、智能管家、技术顾问等)*
"""

AGENT_FILLED = """# AGENT.md

- **名字**: 小牛
- **角色**: 技术顾问
- **性格**: 友好、专业
"""

USER_TEMPLATE = """# USER.md

- **姓名**: *(在首次对话时询问)*
- **称呼**: *(在首次对话时询问)*
"""

USER_FILLED = """# USER.md

- **姓名**: 王小明
- **称呼**: 小明
"""

BOOTSTRAP = """# BOOTSTRAP.md - 首次初始化引导

Ask the user what to call you and what to call them, then write both files.
"""


class OnboardingNeedsEveryIdentityFile(unittest.TestCase):
    """A workspace mid-onboarding, with BOOTSTRAP.md still in place."""

    def setUp(self):
        self.workspace = tempfile.mkdtemp(prefix="cow-onboarding-")
        self.addCleanup(shutil.rmtree, self.workspace, True)

    def _write(self, filename, content):
        path = os.path.join(self.workspace, filename)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(content)
        return path

    def _filled_flags(self):
        """What ``_is_onboarding_done`` makes of the workspace as it stands."""
        from agent.prompt.workspace import _is_onboarding_done

        return _is_onboarding_done(self.workspace)

    def test_answering_only_the_agents_name_does_not_finish_onboarding(self):
        """The case that loses the questions: AGENT.md is filled, USER.md is
        still the template the user was never asked about."""
        self._write("AGENT.md", AGENT_FILLED)
        self._write("USER.md", USER_TEMPLATE)

        self.assertFalse(
            self._filled_flags(),
            "one filled file declared onboarding complete, so the questions "
            "for the other one are never asked and it stays a template",
        )

    def test_answering_only_the_users_name_does_not_finish_onboarding(self):
        """The same mistake from the other side. Which file counts must not
        depend on the order the placeholders happen to be written in."""
        self._write("AGENT.md", AGENT_TEMPLATE_AGENT)
        self._write("USER.md", USER_FILLED)

        self.assertFalse(self._filled_flags())

    def test_onboarding_is_done_once_both_are_filled(self):
        """The guard against a check that answers "no" forever and leaves
        BOOTSTRAP.md sitting in every workspace's context."""
        self._write("AGENT.md", AGENT_FILLED)
        self._write("USER.md", USER_FILLED)

        self.assertTrue(self._filled_flags())

    def test_a_workspace_with_neither_identity_file_is_not_done(self):
        """Nothing was filled, so nothing was finished. Without a file having
        been read at all there is no evidence to retire the script on."""
        self.assertFalse(self._filled_flags())

    def test_the_bootstrap_script_survives_a_partly_answered_onboarding(self):
        """What the user actually loses.

        ``load_context_files`` deletes BOOTSTRAP.md and skips loading it as soon
        as onboarding looks complete, so a half-answered onboarding loses the
        script that carries the remaining questions -- and with it the only
        prompt that was going to fill USER.md in.
        """
        from agent.prompt.workspace import DEFAULT_BOOTSTRAP_FILENAME, load_context_files

        self._write("AGENT.md", AGENT_FILLED)
        self._write("USER.md", USER_TEMPLATE)
        bootstrap = self._write("BOOTSTRAP.md", BOOTSTRAP)

        loaded = {f.path for f in load_context_files(self.workspace)}

        self.assertTrue(
            os.path.exists(bootstrap),
            "BOOTSTRAP.md was deleted while USER.md was still a template",
        )
        self.assertIn(
            DEFAULT_BOOTSTRAP_FILENAME, loaded,
            "BOOTSTRAP.md was dropped from the context, so the remaining "
            "onboarding questions are never asked",
        )


if __name__ == "__main__":
    unittest.main()
