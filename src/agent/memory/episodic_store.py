# from typing import List, Optional

# from agent.memory.episodic import EpisodicMemory


# class EpisodicMemoryStore:

#     def __init__(
#         self,
#         repository=None,
#     ):
#         self.memories: List[
#             EpisodicMemory
#         ] = []

#         self.repository = repository

#         if self.repository is not None:
#             self.load()

#     def add(
#         self,
#         memory: EpisodicMemory,
#     ) -> None:

#         self.memories.append(memory)

#         self._persist(memory)

#     def all(
#         self,
#     ) -> List[EpisodicMemory]:

#         return list(self.memories)

#     def latest(
#         self,
#     ) -> Optional[EpisodicMemory]:

#         if not self.memories:
#             return None

#         return self.memories[-1]

#     def count(self) -> int:
#         return len(self.memories)

#     def clear(self) -> None:

#         self.memories.clear()

#         if self.repository is not None:
#             self.repository.clear()

#     def load(self) -> None:

#         if self.repository is None:
#             return

#         self.memories = (
#             self.repository.all()
#         )

#     def save_all(self) -> None:

#         if self.repository is None:
#             return

#         self.repository.save_many(
#             self.memories
#         )

#     def _persist(
#         self,
#         memory: EpisodicMemory,
#     ) -> None:

#         if self.repository is None:
#             return

#         self.repository.save(memory)


import re
from typing import Any, Dict, List, Optional, Set, Tuple

from agent.memory.episodic import EpisodicMemory
from agent.memory.retrieval import RetrievalResult


class EpisodicMemoryStore:
    """Store episodic memories and retrieve relevant past experiences."""

    def __init__(self, repository=None):
        self.memories: List[EpisodicMemory] = []
        self.repository = repository

        if self.repository is not None:
            self.load()

    def add(self, memory: EpisodicMemory) -> None:
        self.memories.append(memory)
        self._persist(memory)

    def all(self) -> List[EpisodicMemory]:
        return list(self.memories)

    def latest(self) -> Optional[EpisodicMemory]:
        if not self.memories:
            return None
        return self.memories[-1]

    def count(self) -> int:
        return len(self.memories)

    def clear(self) -> None:
        self.memories.clear()

        if self.repository is not None:
            self.repository.clear()

    def load(self) -> None:
        if self.repository is None:
            return

        self.memories = self.repository.all()

    def save_all(self) -> None:
        if self.repository is None:
            return

        self.repository.save_many(self.memories)

    def retrieve(
        self,
        goal: Optional[str] = None,
        action: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        limit: int = 5,
        min_score: float = 0.0,
    ) -> List[EpisodicMemory]:
        """Return relevant episodes, ordered from highest to lowest score."""
        return [
            result.memory
            for result in self.retrieve_ranked(
                goal=goal,
                action=action,
                context=context,
                limit=limit,
                min_score=min_score,
            )
        ]

    def retrieve_ranked(
        self,
        goal: Optional[str] = None,
        action: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        limit: int = 5,
        min_score: float = 0.0,
    ) -> List[RetrievalResult]:
        """
        Rank episodes by goal, action, and observed-state relevance.

        The score measures relevance and evidence quality, not the
        probability that repeating an action will succeed.
        """
        if limit <= 0:
            return []

        if not 0.0 <= min_score <= 1.0:
            raise ValueError("min_score must be between 0.0 and 1.0")

        query_context = context or {}
        ranked: List[Tuple[int, RetrievalResult]] = []

        for index, memory in enumerate(self.memories):
            relevance = self._relevance_score(
                memory=memory,
                goal=goal,
                action=action,
                context=query_context,
            )

            if relevance <= 0.0:
                continue

            evidence_quality = self._evidence_quality(memory)
            score = round(
                0.80 * relevance + 0.20 * evidence_quality,
                6,
            )

            if score < min_score:
                continue

            ranked.append(
                (
                    index,
                    RetrievalResult(memory=memory, score=score),
                )
            )

        # Stable ordering: higher scores first; insertion order breaks ties.
        ranked.sort(key=lambda item: (-item[1].score, item[0]))

        return [result for _, result in ranked[:limit]]

    def _relevance_score(
        self,
        memory: EpisodicMemory,
        goal: Optional[str],
        action: Optional[str],
        context: Dict[str, Any],
    ) -> float:
        weighted_score = 0.0
        total_weight = 0.0

        if goal and memory.goal:
            weighted_score += 0.45 * self._text_similarity(
                goal,
                memory.goal,
            )
            total_weight += 0.45

        if action and memory.action:
            weighted_score += 0.25 * self._text_similarity(
                action,
                memory.action,
            )
            total_weight += 0.25

        if context:
            context_similarity = self._context_similarity(
                memory=memory,
                context=context,
            )

            # Only include context when the episode has comparable state data.
            if context_similarity is not None:
                weighted_score += 0.30 * context_similarity
                total_weight += 0.30

        if total_weight == 0.0:
            return 0.0

        return max(0.0, min(1.0, weighted_score / total_weight))

    def _context_similarity(
        self,
        memory: EpisodicMemory,
        context: Dict[str, Any],
    ) -> Optional[float]:
        candidates = (
            memory.observation_before,
            memory.observation_after,
        )

        comparable: List[Dict[str, Any]] = [
            state
            for state in candidates
            if isinstance(state, dict) and state
        ]

        if not comparable:
            return None

        best_match = 0.0

        for state in comparable:
            matched = sum(
                1
                for key, value in context.items()
                if key in state and state[key] == value
            )

            # Unmatched query keys are treated as mismatches.
            score = matched / max(1, len(context))
            best_match = max(best_match, score)

        return best_match

    def _evidence_quality(self, memory: EpisodicMemory) -> float:
        """Estimate evidence quality without treating success as certainty."""
        confidence = self._bounded_number(memory.confidence)
        importance = self._bounded_number(memory.importance)

        result = memory.result
        if isinstance(result, dict):
            success = result.get("success")
        else:
            success = None

        # Successful and failed episodes are both useful evidence.
        # Their outcome does not independently increase the score.
        outcome_evidence = 1.0 if isinstance(success, bool) else 0.5

        return (
            0.60 * confidence
            + 0.20 * importance
            + 0.20 * outcome_evidence
        )

    @staticmethod
    def _bounded_number(value: Any) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return 0.0

        if number != number:  # NaN
            return 0.0

        return max(0.0, min(1.0, number))

    @staticmethod
    def _tokens(value: Any) -> Set[str]:
        if value is None:
            return set()

        return {
            token
            for token in re.findall(r"\w+", str(value).casefold())
            if token
        }

    @classmethod
    def _text_similarity(cls, first: Any, second: Any) -> float:
        first_tokens = cls._tokens(first)
        second_tokens = cls._tokens(second)

        if not first_tokens or not second_tokens:
            return 0.0

        union = first_tokens | second_tokens
        return len(first_tokens & second_tokens) / len(union)

    def _persist(self, memory: EpisodicMemory) -> None:
        if self.repository is None:
            return

        self.repository.save(memory)
