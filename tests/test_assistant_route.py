"""The one-page assistant's router: it only classifies; names only go to the model; permissions are respected."""
from __future__ import annotations

import json
import time

from agent.workflow.intent_router import classify_intent
from api.security.authentication import AuthenticatedPrincipal
from tests.test_data_changes_api import USERS
from tests.test_sqlite_write import _call, _env, world  # noqa: F401  (fixtures)

HISTORY = "task:history:read"
USERS["asst-maker"] = AuthenticatedPrincipal(
    actor_id="asst-maker", permissions=("database:create:manage", HISTORY), auth_time=int(time.time()))
USERS["asst-nobody"] = AuthenticatedPrincipal(actor_id="asst-nobody", permissions=(), auth_time=int(time.time()))

CSV = b"Produk,Qty\nkopi,10\nteh,5\n"


def _say(intent, confidence=0.9, **extra):
    return json.dumps({"intent": intent, "confidence": confidence, "database": None, "table": None,
                       "question": None, **extra})


def test_a_message_about_a_new_database_is_routed_to_creation(world):
    client, provider, _ = world
    provider.answers.append(_say("create_database", database="hai_toko", question="buat database toko"))
    out = _call(client, "POST", "/assistant/route", "asst-maker", json={"message": "buatkan database toko"})
    assert out.status_code == 200, out.text
    body = out.json()
    assert body["intent"] == "create_database" and body["database"] == "hai_toko"
    assert body["allowed"]["create_database"] is True and body["allowed"]["change_data"] is False


def test_a_file_goes_to_the_model_as_a_name_and_headers_never_as_values(world, monkeypatch):
    client, provider, _ = world
    seen = []
    original = provider.complete

    def spy(request):
        seen.append(" ".join(m.content for m in request.messages))
        return original(request)

    monkeypatch.setattr(provider, "complete", spy)
    provider.answers.append(_say("create_database"))
    out = client.request("POST", "/api/v1/assistant/route-file?filename=jual.csv&message=jadikan%20tabel",
                         content=CSV, headers={"X-Test-User": "asst-maker",
                                               "Content-Type": "application/octet-stream"})
    assert out.status_code == 200, out.text
    assert out.json()["file"] == {"name": "jual.csv", "rows": 2, "headers": ["Produk", "Qty"]}
    assert seen and "Produk" in seen[0] and "kopi" not in seen[0] and "teh" not in seen[0]


def test_an_intent_the_person_may_not_run_is_not_offered(world):
    client, provider, _ = world
    provider.answers.append(_say("change_data"))
    body = _call(client, "POST", "/assistant/route", "asst-maker", json={"message": "ubah harga kopi"}).json()
    assert body["intent"] == "unclear" and body["denied"] == "change_data"


def test_a_user_without_any_role_cannot_use_the_assistant_at_all(world):
    client, provider, _ = world
    out = _call(client, "POST", "/assistant/route", "asst-nobody", json={"message": "halo"})
    assert out.status_code == 403 and provider.answers == []


def test_a_bad_body_or_an_unreadable_file_is_refused_without_asking_the_model(world):
    client, provider, _ = world
    assert _call(client, "POST", "/assistant/route", "asst-maker", json={"message": "  "}).status_code == 422
    assert _call(client, "POST", "/assistant/route", "asst-maker", json={"message": "x", "y": 1}).status_code == 422
    bad = client.request("POST", "/api/v1/assistant/route-file?filename=a.pdf", content=b"x",
                         headers={"X-Test-User": "asst-maker"})
    assert bad.json()["intent"] == "unclear" and "refused" in bad.json()


class _Llm:
    def __init__(self, text=None, error=None):
        self.text, self.error = text, error

    def complete(self, messages, temperature=0):
        if self.error:
            raise self.error
        return type("R", (), {"content": self.text})()


def test_anything_doubtful_becomes_unclear_instead_of_a_guess():
    assert classify_intent(_Llm(error=RuntimeError("down")), "x")["intent"] == "unclear"
    assert classify_intent(_Llm("not json"), "x")["intent"] == "unclear"
    assert classify_intent(_Llm(_say("delete_everything")), "x")["intent"] == "unclear"
    assert classify_intent(_Llm(_say("record_data", 0.3)), "x")["intent"] == "unclear"
    assert classify_intent(_Llm(_say("record_data", True)), "x")["intent"] == "unclear"


def test_names_from_the_model_must_look_like_database_names():
    out = classify_intent(_Llm(_say("record_data", database="hai toko; drop", table="tb_jual")), "x")
    assert out["intent"] == "record_data" and out["database"] is None and out["table"] == "tb_jual"


def test_a_report_format_is_read_from_the_message_only_for_viewing_data():
    ask = lambda intent, fmt: classify_intent(_Llm(_say(intent, format=fmt)), "x")["format"]  # noqa: E731
    assert ask("view_data", "xlsx") == "xlsx" and ask("view_data", "pdf") == "pdf"
    assert ask("view_data", None) is None and ask("view_data", "docx") is None
    assert ask("record_data", "xlsx") is None                                         # only reports have a format
