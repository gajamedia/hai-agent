from agent.memory.semantic import SemanticMemory
from agent.memory.semantic_retriever import (
    SemanticMemoryRetriever,
)
from agent.memory.semantic_store import (
    SemanticMemoryStore,
)


def print_result(
    index,
    result,
):

    memory = result.memory

    print(
        f"Result #{index}"
    )

    print(
        "  Score:",
        result.score,
    )

    print(
        "  Subject:",
        memory.subject,
    )

    print(
        "  Predicate:",
        memory.predicate,
    )

    print(
        "  Object:",
        memory.object,
    )

    print(
        "  Context:",
        memory.context,
    )

    print(
        "  Confidence:",
        memory.confidence,
    )

    print(
        "  Evidence:",
        memory.evidence_count,
    )

    print(
        "  Contradictions:",
        memory.contradiction_count,
    )

    print()


def main():

    print("================================")
    print("CONTEXT-AWARE MEMORY RETRIEVAL")
    print("================================")
    print()

    store = SemanticMemoryStore()

    # ----------------------------------------------
    # General knowledge
    # ----------------------------------------------

    general = SemanticMemory(
        content=(
            "box usually changes from "
            "closed to open"
        ),
        subject="box",
        predicate="changes_from",
        object={
            "before": "closed",
            "after": "open",
        },
        context={},
        confidence=0.9,
        evidence_count=10,
    )

    # ----------------------------------------------
    # Normal-context knowledge
    # ----------------------------------------------

    normal = SemanticMemory(
        content=(
            "box opens under normal condition"
        ),
        subject="box",
        predicate="changes_from",
        object={
            "before": "closed",
            "after": "open",
        },
        context={
            "condition": "normal",
        },
        confidence=0.85,
        evidence_count=8,
    )

    # ----------------------------------------------
    # Jammed-context knowledge
    # ----------------------------------------------

    jammed = SemanticMemory(
        content=(
            "box does not open when jammed"
        ),
        subject="box",
        predicate="changes_from",
        object={
            "before": "closed",
            "after": "closed",
        },
        context={
            "condition": "jammed",
        },
        confidence=0.95,
        evidence_count=6,
    )

    # ----------------------------------------------

    store.add(general)
    store.add(normal)
    store.add(jammed)

    retriever = SemanticMemoryRetriever(
        store
    )

    # ==================================================
    # NORMAL CONTEXT
    # ==================================================

    print("================================")
    print("QUERY: NORMAL")
    print("================================")
    print()

    normal_results = (
        retriever.retrieve_ranked(
            subject="box",
            predicate="changes_from",
            context={
                "condition": "normal",
            },
        )
    )

    for index, result in enumerate(
        normal_results,
        start=1,
    ):

        print_result(
            index=index,
            result=result,
        )

    # ==================================================
    # JAMMED CONTEXT
    # ==================================================

    print("================================")
    print("QUERY: JAMMED")
    print("================================")
    print()

    jammed_results = (
        retriever.retrieve_ranked(
            subject="box",
            predicate="changes_from",
            context={
                "condition": "jammed",
            },
        )
    )

    for index, result in enumerate(
        jammed_results,
        start=1,
    ):

        print_result(
            index=index,
            result=result,
        )

    # ==================================================
    # SUMMARY
    # ==================================================

    print("================================")
    print("SUMMARY")
    print("================================")
    print()

    if normal_results:

        print(
            "Best normal knowledge:",
            normal_results[0].memory.content,
        )

    if jammed_results:

        print(
            "Best jammed knowledge:",
            jammed_results[0].memory.content,
        )


if __name__ == "__main__":
    main()
