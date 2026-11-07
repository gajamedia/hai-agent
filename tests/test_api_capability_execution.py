from __future__ import annotations

import sqlite3
from pathlib import Path
from threading import RLock

from fastapi.testclient import TestClient

from agent.core.action import Action
from agent.llm.authenticated_capability_execution_gateway import AuthenticatedCapabilityExecutionGateway
from agent.llm.business_database_adapter import BusinessDatabaseAdapter, ReadOnlyTableSpec
from agent.llm.capability_registry import TrustedCapabilityRegistry
from agent.llm.execution_security import ExecutionSecurityPolicy
from agent.llm.execution_trace_store import ExecutionTraceStore
from agent.llm.room_environment_adapter import RoomEnvironmentAdapter
from agent.llm.traced_authenticated_capability_execution_gateway import TracedAuthenticatedCapabilityExecutionGateway
from api.main import create_app
from api.security.authentication import AuthenticatedPrincipal
from api.security.authorization import AuthorizationPolicy

PERMISSION = 'capability:read:execute'
ROOM_CAPABILITY = RoomEnvironmentAdapter.OBSERVE_CAPABILITY
DB_CAPABILITY = BusinessDatabaseAdapter.SELECT_CAPABILITY


class StaticProvider:
    def __init__(self, principal):
        self.principal = principal

    async def authenticate(self, request):
        return self.principal


class FakeEnvironment:
    def __init__(self, fail=False):
        self.calls = 0
        self.fail = fail

    def observe(self):
        self.calls += 1
        if self.fail:
            raise OSError('private adapter failure detail')
        return {'door': 'locked', 'box': 'closed'}


class FakeExecutionRuntime:
    def __init__(self, environment):
        self.environment = environment
        self.lock = RLock()


class FakeRepository:
    def close(self):
        pass


def _database_adapter(tmp_path: Path):
    path = tmp_path / 'business-test.sqlite3'
    with sqlite3.connect(path) as connection:
        connection.execute('CREATE TABLE orders (id INTEGER PRIMARY KEY, status TEXT, total INTEGER)')
        connection.executemany(
            'INSERT INTO orders (id, status, total) VALUES (?, ?, ?)',
            [(1, 'paid', 150000), (2, 'pending', 90000), (3, 'paid', 225000)],
        )
    return BusinessDatabaseAdapter(
        connection_factory=lambda: sqlite3.connect(path),
        table_specs={'orders': ReadOnlyTableSpec(
            table='orders', columns=('id', 'status', 'total'),
            filterable_columns=('status',), sortable_columns=('id', 'total'),
            default_limit=20, max_limit=50,
        )},
        paramstyle='qmark',
    )


def _make_client(tmp_path, monkeypatch, *, principal=None, environment=None, fail_trace_start=False):
    tmp_path.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv('HAI_AUTH_ENABLED', 'true')
    monkeypatch.setenv('HAI_AUTH_REQUIRED', 'false')
    monkeypatch.setenv('HAI_ENV', 'development')
    env = environment or FakeEnvironment()
    holders = {}

    def gateway_factory(app):
        registry = TrustedCapabilityRegistry()
        for capability_id, action_name, description in (
            (ROOM_CAPABILITY, 'read_room_state', 'Read isolated room state.'),
            (DB_CAPABILITY, 'read_business_data', 'Read allowlisted business data.'),
        ):
            registry.register_trusted(
                capability_id=capability_id,
                action=Action(name=action_name, description=description, handler=lambda **_: {'success': True}),
                risk_level='low', permissions=(PERMISSION,),
            )
        trace_store = ExecutionTraceStore(tmp_path / 'capability-trace.jsonl', fsync=False)
        if fail_trace_start:
            trace_store.append = lambda event: (_ for _ in ()).throw(OSError('trace disk unavailable'))
        adapters = [RoomEnvironmentAdapter(env, RLock()), _database_adapter(tmp_path)]
        gateway = AuthenticatedCapabilityExecutionGateway(
            adapters,
            capability_registry=registry,
            authorization_policy=app.state.authorization_policy,
            execution_security_policy=ExecutionSecurityPolicy(
                action_risks={ROOM_CAPABILITY: 0.1, DB_CAPABILITY: 0.1},
            ),
        )
        holders['registry'] = registry
        holders['store'] = trace_store
        app.state.capability_execution_registry = registry
        app.state.execution_trace_store = trace_store
        return TracedAuthenticatedCapabilityExecutionGateway(gateway, trace_store)

    app = create_app(
        runtime_factory=lambda: object(),
        execution_runtime_factory=lambda: FakeExecutionRuntime(env),
        capability_store_factory=lambda: object(),
        capability_execution_gateway_factory=gateway_factory,
        authentication_provider_factory=lambda app: StaticProvider(principal),
        authorization_policy_factory=lambda app: AuthorizationPolicy(),
        autonomous_testing_repository_factory=lambda path: FakeRepository(),
        autonomous_testing_manager_factory=lambda repository: object(),
    )
    return TestClient(app), app, env, holders


def _principal():
    return AuthenticatedPrincipal(actor_id='integration-operator', permissions=(PERMISSION,))


def test_business_database_read_completes_http_auth_gateway_adapter_and_trace(tmp_path, monkeypatch):
    client, _, _, holders = _make_client(tmp_path, monkeypatch, principal=_principal())
    with client:
        response = client.post('/api/v1/capability-reads', json={
            'target': 'business_database', 'capability_id': DB_CAPABILITY,
            'arguments': {'table': 'orders', 'columns': ['id', 'status', 'total'],
                          'filters': {'status': 'paid'}, 'order_by': ['-total'], 'limit': 10},
        })
    assert response.status_code == 200, response.text
    body = response.json()
    assert body['data'] == [
        {'id': 3, 'status': 'paid', 'total': 225000},
        {'id': 1, 'status': 'paid', 'total': 150000},
    ]
    assert body['read_only'] is True
    assert 'actor_id' not in body and 'arguments' not in body
    records = holders['store'].read_all()
    assert [r['event_type'] for r in records] == ['started', 'authorized', 'completed']
    assert all('arguments' not in r['payload'] and 'data' not in r['payload'] for r in records)


def test_room_environment_uses_same_generic_http_gateway(tmp_path, monkeypatch):
    client, _, env, _ = _make_client(tmp_path, monkeypatch, principal=_principal())
    with client:
        response = client.post('/api/v1/capability-reads', json={
            'target': 'room_environment', 'capability_id': ROOM_CAPABILITY, 'arguments': {},
        })
    assert response.status_code == 200, response.text
    assert response.json()['data'] == {'door': 'locked', 'box': 'closed'}
    assert env.calls == 1


def test_missing_principal_is_rejected_before_adapter(tmp_path, monkeypatch):
    client, _, env, _ = _make_client(tmp_path, monkeypatch, principal=None)
    with client:
        response = client.post('/api/v1/capability-reads', json={
            'target': 'room_environment', 'capability_id': ROOM_CAPABILITY,
        })
    assert response.status_code == 401
    assert env.calls == 0


def test_missing_route_permission_is_rejected_before_gateway(tmp_path, monkeypatch):
    client, _, env, _ = _make_client(
        tmp_path, monkeypatch, principal=AuthenticatedPrincipal(actor_id='no-permission'),
    )
    with client:
        response = client.post('/api/v1/capability-reads', json={
            'target': 'room_environment', 'capability_id': ROOM_CAPABILITY,
        })
    assert response.status_code == 403
    assert response.json()['detail']['error'] == 'permission_denied'
    assert env.calls == 0


def test_caller_cannot_supply_actor_or_execution_id(tmp_path, monkeypatch):
    client, _, env, _ = _make_client(tmp_path, monkeypatch, principal=_principal())
    with client:
        response = client.post('/api/v1/capability-reads', json={
            'target': 'room_environment', 'capability_id': ROOM_CAPABILITY,
            'actor_id': 'forged', 'execution_id': 'forged',
        })
    assert response.status_code == 422
    assert env.calls == 0


def test_disabled_and_revoked_capabilities_are_denied_and_traced(tmp_path, monkeypatch):
    for transition in ('disable', 'revoke'):
        client, _, env, holders = _make_client(tmp_path / transition, monkeypatch, principal=_principal())
        with client:
            if transition == 'disable':
                holders['registry'].disable(ROOM_CAPABILITY)
            else:
                holders['registry'].revoke(ROOM_CAPABILITY)
            response = client.post('/api/v1/capability-reads', json={
                'target': 'room_environment', 'capability_id': ROOM_CAPABILITY,
            })
        assert response.status_code == 403
        assert env.calls == 0
        assert [r['event_type'] for r in holders['store'].read_all()] == ['started', 'denied']


def test_adapter_failure_is_sanitized_and_traced(tmp_path, monkeypatch):
    env = FakeEnvironment(fail=True)
    client, _, _, holders = _make_client(tmp_path, monkeypatch, principal=_principal(), environment=env)
    with client:
        response = client.post('/api/v1/capability-reads', json={
            'target': 'room_environment', 'capability_id': ROOM_CAPABILITY,
        })
    assert response.status_code == 502
    assert 'private adapter failure detail' not in response.text
    assert [r['event_type'] for r in holders['store'].read_all()] == ['started', 'failed']


def test_trace_start_failure_prevents_adapter_access(tmp_path, monkeypatch):
    env = FakeEnvironment()
    client, _, _, _ = _make_client(
        tmp_path, monkeypatch, principal=_principal(), environment=env, fail_trace_start=True,
    )
    with client:
        response = client.post('/api/v1/capability-reads', json={
            'target': 'room_environment', 'capability_id': ROOM_CAPABILITY,
        })
    assert response.status_code == 503
    assert env.calls == 0


def test_terminal_trace_failure_withholds_read_result(tmp_path, monkeypatch):
    client, _, env, holders = _make_client(tmp_path, monkeypatch, principal=_principal())
    with client:
        store = holders['store']
        original_append = store.append
        def fail_completed(event):
            if event['event_type'] == 'completed':
                raise OSError('simulated trace disk full')
            return original_append(event)
        store.append = fail_completed
        response = client.post('/api/v1/capability-reads', json={
            'target': 'room_environment', 'capability_id': ROOM_CAPABILITY,
        })
    assert response.status_code == 503
    assert env.calls == 1
    assert [r['event_type'] for r in store.read_all()] == ['started', 'authorized']


def test_unconfigured_gateway_fails_closed(tmp_path, monkeypatch):
    monkeypatch.setenv('HAI_AUTH_ENABLED', 'true')
    monkeypatch.setenv('HAI_AUTH_REQUIRED', 'false')
    app = create_app(
        runtime_factory=lambda: object(), execution_runtime_factory=lambda: FakeExecutionRuntime(FakeEnvironment()),
        capability_store_factory=lambda: object(),
        authentication_provider_factory=lambda app: StaticProvider(_principal()),
        authorization_policy_factory=lambda app: AuthorizationPolicy(),
        autonomous_testing_repository_factory=lambda path: FakeRepository(),
        autonomous_testing_manager_factory=lambda repository: object(),
    )
    with TestClient(app) as client:
        response = client.post('/api/v1/capability-reads', json={
            'target': 'room_environment', 'capability_id': ROOM_CAPABILITY,
        })
    assert response.status_code == 503
    assert response.json()['detail']['error'] == 'capability_execution_unavailable'
