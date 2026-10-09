from typing import Any, Dict, List, Optional, Tuple

from agent.memory.retrieval import RetrievalResult
from agent.memory.semantic import SemanticMemory
from agent.memory.semantic_store import SemanticMemoryStore


class SemanticMemoryRetriever:

    def __init__(
        self,
        store: SemanticMemoryStore,
    ):
        self.store = store

    def retrieve(
        self,
        subject: str,
        predicate: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        limit: int = 5,
    ) -> List[SemanticMemory]:

        ranked_results = self.retrieve_ranked(
            subject=subject,
            predicate=predicate,
            context=context,
            limit=limit,
        )

        return [
            result.memory
            for result in ranked_results
        ]

    def retrieve_ranked(
        self,
        subject: str,
        predicate: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        limit: int = 5,
    ) -> List[RetrievalResult]:

        query_context = (
            context
            if context is not None
            else {}
        )

        memories = self.store.find(
            subject=subject,
            predicate=predicate,
        )

        ranked: List[RetrievalResult] = []

        for memory in memories:

            score = self._score(
                memory=memory,
                context=query_context,
            )

            if score <= 0.0:
                continue

            ranked.append(
                RetrievalResult(
                    memory=memory,
                    score=score,
                )
            )

        ranked.sort(
            key=lambda result: result.score,
            reverse=True,
        )

        return ranked[:limit]

    def _score(
        self,
        memory: SemanticMemory,
        context: Dict[str, Any],
    ) -> float:

        context_score = (
            self._context_score(
                memory=memory,
                context=context,
            )
        )

        confidence_score = (
            memory.confidence
        )

        evidence_score = (
            self._evidence_score(
                memory
            )
        )

        contradiction_penalty = (
            self._contradiction_penalty(
                memory
            )
        )

        score = (
            0.50 * context_score
            + 0.30 * confidence_score
            + 0.20 * evidence_score
            - contradiction_penalty
        )

        return round(
            max(
                0.0,
                score,
            ),
            6,
        )

    def _context_score(
        self,
        memory: SemanticMemory,
        context: Dict[str, Any],
    ) -> float:

        # General knowledge has no specific context.
        # It remains usable as fallback knowledge.
        if not memory.context:
            return 0.5

        if not context:
            return 0.0

        matched = 0

        for key, expected in memory.context.items():

            if context.get(key) == expected:
                matched += 1

        return round(
            matched / len(memory.context),
            6,
        )

    def _evidence_score(
        self,
        memory: SemanticMemory,
    ) -> float:

        return min(
            1.0,
            memory.evidence_count / 10.0,
        )

    def _contradiction_penalty(
        self,
        memory: SemanticMemory,
    ) -> float:

        if memory.contradiction_count == 0:
            return 0.0

        return min(
            0.5,
            memory.contradiction_count * 0.05,
        )
