"""Explicit, read-only DB-API connection lifecycle management.

No credentials are stored here. Factories obtain credentials through the
caller's approved configuration mechanism. A driver-specific guard is
mandatory and must fail closed unless read-only permissions/session policy
are verified for every new connection.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from threading import RLock
from typing import Any, Callable, Dict, Iterator, Optional, Protocol


class DBAPIConnection(Protocol):
    def cursor(self) -> Any: ...
    def close(self) -> None: ...


ConnectionFactory = Callable[[], DBAPIConnection]
ReadOnlyGuard = Callable[[DBAPIConnection], None]


class ConnectionManagerError(RuntimeError):
    """Base error for connection manager failures."""


class ConnectionPolicyError(ConnectionManagerError):
    """Raised when a connection is not explicitly authorized/read-only."""


class ConnectionNotConfiguredError(ConnectionManagerError):
    """Raised when a requested connection profile does not exist."""


@dataclass(frozen=True)
class ConnectionProfile:
    name: str
    database: str
    authorization_ref: str
    read_only: bool
    enabled: bool


@dataclass(frozen=True)
class ConnectionHealth:
    profile_name: str
    healthy: bool
    database: str
    error_type: Optional[str] = None


@dataclass
class _RegisteredProfile:
    profile: ConnectionProfile
    factory: ConnectionFactory
    read_only_guard: ReadOnlyGuard


class _ManagedConnectionHandle:
    """DB-API proxy; close() releases the underlying manager lease."""

    def __init__(self, manager_context: Any, connection: DBAPIConnection) -> None:
        self._manager_context = manager_context
        self._connection = connection
        self._closed = False
        self._lock = RLock()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._connection, name)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._manager_context.__exit__(None, None, None)


class ConnectionManager:
    """Manage fresh, short-lived, explicitly authorized read-only connections.

    This is not a connection pool. Each request obtains a fresh connection
    and closes it when the caller closes the returned handle or exits the
    context manager. The connection_factory() method is suitable for
    injection into DB-API adapters that call connection.close().
    """

    def __init__(self) -> None:
        self._profiles: Dict[str, _RegisteredProfile] = {}
        self._active: Dict[int, DBAPIConnection] = {}
        self._lock = RLock()
        self._closed = False

    def register(
        self,
        *,
        name: str,
        database: str,
        authorization_ref: str,
        connection_factory: ConnectionFactory,
        read_only_guard: ReadOnlyGuard,
        read_only: bool = True,
        enabled: bool = True,
    ) -> ConnectionProfile:
        normalized_name = name.strip() if isinstance(name, str) else ""
        normalized_database = database.strip().lower() if isinstance(database, str) else ""
        normalized_auth = authorization_ref.strip() if isinstance(authorization_ref, str) else ""

        if not normalized_name:
            raise ValueError("connection profile name is required")
        if not normalized_database:
            raise ValueError("database type is required")
        if not normalized_auth:
            raise ConnectionPolicyError("an explicit authorization_ref is required")
        if not callable(connection_factory):
            raise TypeError("connection_factory must be callable")
        if not callable(read_only_guard):
            raise ConnectionPolicyError("a driver-specific read_only_guard is required")
        if read_only is not True:
            raise ConnectionPolicyError("only read-only database profiles are supported")
        if enabled is not True:
            raise ValueError("disabled profiles must not be registered")

        profile = ConnectionProfile(
            name=normalized_name,
            database=normalized_database,
            authorization_ref=normalized_auth,
            read_only=True,
            enabled=True,
        )
        with self._lock:
            if self._closed:
                raise ConnectionManagerError("connection manager is shut down")
            if normalized_name in self._profiles:
                raise ValueError(f"connection profile already registered: {normalized_name}")
            self._profiles[normalized_name] = _RegisteredProfile(
                profile=profile,
                factory=connection_factory,
                read_only_guard=read_only_guard,
            )
        return profile

    def get_profile(self, name: str) -> ConnectionProfile:
        with self._lock:
            registered = self._profiles.get(name)
            if registered is None or not registered.profile.enabled:
                raise ConnectionNotConfiguredError(
                    f"connection profile is not configured: {name}"
                )
            return registered.profile

    @contextmanager
    def connection(self, name: str) -> Iterator[DBAPIConnection]:
        with self._lock:
            if self._closed:
                raise ConnectionManagerError("connection manager is shut down")
            registered = self._profiles.get(name)
            if registered is None or not registered.profile.enabled:
                raise ConnectionNotConfiguredError(
                    f"connection profile is not configured: {name}"
                )

        connection: Optional[DBAPIConnection] = None
        connection_id: Optional[int] = None
        try:
            connection = registered.factory()
            if connection is None:
                raise ConnectionManagerError("connection factory returned None")

            # Must verify/apply backend-specific read-only restrictions.
            registered.read_only_guard(connection)

            connection_id = id(connection)
            with self._lock:
                if self._closed:
                    raise ConnectionManagerError("connection manager shut down during connect")
                self._active[connection_id] = connection

            yield connection
        finally:
            if connection is not None:
                if connection_id is not None:
                    with self._lock:
                        self._active.pop(connection_id, None)
                try:
                    connection.close()
                except Exception:
                    pass

    def connect(self, name: str) -> DBAPIConnection:
        """Return a DB-API-compatible handle that owns its connection lease."""
        manager_context = self.connection(name)
        connection = manager_context.__enter__()
        return _ManagedConnectionHandle(manager_context, connection)

    def connection_factory(self, name: str) -> ConnectionFactory:
        """Return a fresh-connection factory for DB-API adapters.

        The consuming adapter must call close() on the returned handle.
        """
        self.get_profile(name)
        return lambda: self.connect(name)

    def health_check(self, name: str) -> ConnectionHealth:
        profile = self.get_profile(name)
        try:
            with self.connection(name) as connection:
                cursor = connection.cursor()
                try:
                    cursor.execute("SELECT 1")
                    row = cursor.fetchone()
                    if not row or row[0] != 1:
                        raise ConnectionManagerError(
                            "health probe returned an unexpected result"
                        )
                finally:
                    close_cursor = getattr(cursor, "close", None)
                    if callable(close_cursor):
                        close_cursor()
            return ConnectionHealth(name, True, profile.database)
        except Exception as exc:
            # Do not expose raw errors that may contain credentials or host data.
            return ConnectionHealth(
                profile_name=name,
                healthy=False,
                database=profile.database,
                error_type=type(exc).__name__,
            )

    def shutdown(self) -> None:
        with self._lock:
            self._closed = True
            active = list(self._active.values())
            self._active.clear()
            self._profiles.clear()
        for connection in active:
            try:
                connection.close()
            except Exception:
                pass

    @property
    def active_connection_count(self) -> int:
        with self._lock:
            return len(self._active)

    @property
    def registered_profile_names(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(sorted(self._profiles))
