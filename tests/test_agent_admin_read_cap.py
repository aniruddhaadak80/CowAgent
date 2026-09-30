# encoding:utf-8
"""``read_core_file`` must apply the same size cap as ``write_core_file``.

``write_core_file`` refuses content over ``MAX_CORE_FILE_BYTES`` (1 MiB), but
the read side had no limit at all, so a core file grown past 1 MiB by the
agent's own ``write``/``edit`` tools was read whole and returned inside a JSON
response. The API therefore had a write limit and no read limit.
"""

import json

import pytest

from agent.admin import AgentAdminError
from agent.registry import AgentRegistry, set_agent_registry
from agent import team
from agent.admin import AgentAdminService

MAX = 1024 * 1024


@pytest.fixture
def admin(tmp_path):
    primary = tmp_path / "primary"
    primary.mkdir()
    settings = {
        "agent_workspace": str(tmp_path),
        "default_agent_id": "primary",
        "agents": [
            {"id": "primary", "name": "Primary", "workspace": str(primary), "enabled": True}
        ],
        "channel_instances": [],
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(settings), encoding="utf-8")
    set_agent_registry(AgentRegistry.from_config(team.resolve(settings)))
    try:
        yield AgentAdminService(str(config_path)), primary
    finally:
        set_agent_registry(None)


def test_reading_an_oversized_core_file_is_refused(admin):
    service, workspace = admin
    (workspace / "AGENT.md").write_text("x" * (MAX + 1), encoding="utf-8")

    with pytest.raises(AgentAdminError) as excinfo:
        service.read_core_file("primary", "AGENT.md")

    assert "1 MiB" in str(excinfo.value)


def test_a_core_file_exactly_at_the_cap_is_still_readable(admin):
    service, workspace = admin
    (workspace / "AGENT.md").write_text("x" * MAX, encoding="utf-8")

    result = service.read_core_file("primary", "AGENT.md")

    assert result["exists"] is True
    assert len(result["content"]) == MAX


def test_a_normal_core_file_reads_back_unchanged(admin):
    service, workspace = admin
    (workspace / "AGENT.md").write_text("# persona\n", encoding="utf-8")

    result = service.read_core_file("primary", "AGENT.md")

    assert result["content"] == "# persona\n"
    assert result["filename"] == "AGENT.md"
    assert result["exists"] is True


def test_a_missing_core_file_still_reports_absent(admin):
    service, _workspace = admin

    result = service.read_core_file("primary", "AGENT.md")

    assert result["exists"] is False
    assert result["content"] == ""
