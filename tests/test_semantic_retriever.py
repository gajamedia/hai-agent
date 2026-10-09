from agent.memory.semantic import SemanticMemory
from agent.memory.semantic_retriever import (
    SemanticMemoryRetriever,
)
from agent.memory.semantic_store import (
    SemanticMemoryStore,
)


def test_retriever_prefers_exact_context():

    store = SemanticMemoryStore()

    exact = SemanticMemory(
        content="normal box opens",
        subject="box",
        predicate="changes_from",
        object={
            "before": "closed",
            "after": "open",
        },
        context={
            "condition": "normal",
        },
        confidence=0.8,
        evidence_count=8,
    )

    partial = SemanticMemory(
        content="generic box rule",
        subject="box",
        predicate="changes_from",
        object={
            "before": "closed",
            "after": "open",
        },
        context={},
        confidence=0.9,
        evidence_count=9,
    )

    store.add(exact)
    store.add(partial)

    retriever = SemanticMemoryRetriever(
        store
    )

    results = retriever.retrieve(
        subject="box",
        predicate="changes_from",
        context={
            "condition": "normal",
        },
    )

    assert len(results) == 2
    assert results[0] is exact


def test_retriever_uses_confidence_and_evidence():

    store = SemanticMemoryStore()

    first = SemanticMemory(
        content="knowledge one",
        subject="box",
        predicate="changes_from",
        object={
            "before": "closed",
            "after": "open",
        },
        context={
            "condition": "normal",
        },
        confidence=0.9,
        evidence_count=8,
    )

    second = SemanticMemory(
        content="knowledge two",
        subject="box",
        predicate="changes_from",
        object={
            "before": "closed",
            "after": "open",
        },
        context={
            "condition": "normal",
        },
        confidence=0.5,
        evidence_count=2,
    )

    store.add(first)
    store.add(second)

    retriever = SemanticMemoryRetriever(
        store
    )

    results = retriever.retrieve(
        subject="box",
        predicate="changes_from",
        context={
            "condition": "normal",
        },
    )

    assert results[0] is first


def test_retriever_penalizes_contradictions():

    store = SemanticMemoryStore()

    clean = SemanticMemory(
        content="clean knowledge",
        subject="box",
        predicate="changes_from",
        object={
            "before": "closed",
            "after": "open",
        },
        context={
            "condition": "normal",
        },
        confidence=0.8,
        evidence_count=8,
        contradiction_count=0,
    )

    contradicted = SemanticMemory(
        content="contradicted knowledge",
        subject="box",
        predicate="changes_from",
        object={
            "before": "closed",
            "after": "open",
        },
        context={
            "condition": "normal",
        },
        confidence=0.8,
        evidence_count=8,
        contradiction_count=5,
    )

    store.add(clean)
    store.add(contradicted)

    retriever = SemanticMemoryRetriever(
        store
    )

    results = retriever.retrieve(
        subject="box",
        predicate="changes_from",
        context={
            "condition": "normal",
        },
    )

    assert results[0] is clean

def test_ranked_retrieval_returns_scores():

    store = SemanticMemoryStore()

    memory = SemanticMemory(
        content="box opens",
        subject="box",
        predicate="changes_from",
        object={
            "before": "closed",
            "after": "open",
        },
        context={
            "condition": "normal",
        },
        confidence=0.8,
        evidence_count=5,
    )

    store.add(memory)

    retriever = SemanticMemoryRetriever(
        store
    )

    results = retriever.retrieve_ranked(
        subject="box",
        predicate="changes_from",
        context={
            "condition": "normal",
        },
    )

    assert len(results) == 1

    result = results[0]

    assert result.memory is memory
    assert result.score > 0.0
