
from agent.llm.agent_runtime import AgentCognitiveRuntime


class FakeWorkingMemory:
    def __init__(self):
        self.goal = "Buka kotak"
        self.context = {"location": "ruangan"}
        self.current_action = None
        self.plan = ["periksa_kotak", "buka_kotak"]
        self.reasoning = "Kotak masih tertutup"
        self.recent_observation = {"box": "closed"}


class FakeAgent:
    name = "test-agent"

    def __init__(self):
        self.working_memory = FakeWorkingMemory()


def test_resolve_relevant_memory_normalizes_working_memory():
    runtime = AgentCognitiveRuntime(
        agent=FakeAgent(),
        orchestrator=object(),
    )

    result = runtime._resolve_relevant_memory(None)

    assert isinstance(result, list)
    assert len(result) == 1

    memory = result[0]
    assert memory["goal"] == "Buka kotak"
    assert memory["context"] == {"location": "ruangan"}
    assert memory["plan"] == ["periksa_kotak", "buka_kotak"]
    assert memory["reasoning"] == "Kotak masih tertutup"
    assert memory["recent_observation"] == {"box": "closed"}
    assert "current_action" not in memory


def test_explicit_relevant_memory_is_preserved():
    runtime = AgentCognitiveRuntime(
        agent=FakeAgent(),
        orchestrator=object(),
    )
    explicit = ["catatan A", "catatan B"]

    result = runtime._resolve_relevant_memory(explicit)

    assert result is explicit


def test_missing_working_memory_returns_empty_list():
    class AgentWithoutWorkingMemory:
        name = "empty-agent"

    runtime = AgentCognitiveRuntime(
        agent=AgentWithoutWorkingMemory(),
        orchestrator=object(),
    )

    assert runtime._resolve_relevant_memory(None) == []
