from typing import Any, Dict, List


class StateChange:

    def __init__(
        self,
        key: str,
        before: Any,
        after: Any,
    ):
        self.key = key
        self.before = before
        self.after = after

    def __repr__(self) -> str:
        return (
            f"StateChange("
            f"key={self.key!r}, "
            f"before={self.before!r}, "
            f"after={self.after!r})"
        )


class ChangeDetector:

    def detect(
        self,
        before: Dict[str, Any],
        after: Dict[str, Any],
    ) -> List[StateChange]:

        changes: List[StateChange] = []

        keys = set(before.keys()) | set(after.keys())

        for key in keys:

            before_value = before.get(key)
            after_value = after.get(key)

            if before_value != after_value:

                changes.append(
                    StateChange(
                        key=key,
                        before=before_value,
                        after=after_value,
                    )
                )

        return changes
