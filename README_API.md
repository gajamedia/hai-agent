# hAI Agent API scaffold

This scaffold adds a FastAPI layer around the existing provider-neutral
`AgentCognitiveRuntime` and `CognitiveOrchestrator`. It does not replace the
Agent core.

## Files

- `api/main.py`: FastAPI application, lifespan and health endpoint.
- `api/dependencies.py`: constructs `Agent`, `LLMService`, `OpenCodeProvider`,
  `CognitiveOrchestrator`, and `AgentCognitiveRuntime`.
- `api/schemas.py`: request/response validation.
- `api/routes/cognitive.py`: cognitive-cycle endpoint.
- `tests/test_api.py`: API tests using a fake runtime.

Copy the `api` directory into the project root, and copy `tests/test_api.py`
into the project's existing `tests` directory. Merge the `dependencies` list
from `pyproject.toml` in this scaffold into the project's existing
`[project]` table rather than replacing unrelated project metadata.

## Install and run (PowerShell, from project root)

```powershell
python -m pip install -e .
python -m pip install httpx
uvicorn api.main:app --reload
```

Open `http://127.0.0.1:8000/docs` for interactive API documentation.

## Environment variables (optional)

```powershell
$env:OPENCODE_MODEL = "provider/model"
$env:OPENCODE_AGENT = "build"
$env:OPENCODE_TIMEOUT = "120"
$env:HAI_AGENT_NAME = "hAI"
```

The `OPENCODE_MODEL` and `OPENCODE_AGENT` values are examples, not required
defaults. Use values configured in the user's OpenCode installation.

## Endpoint

`POST /api/v1/cognitive/run`

Example body:

```json
{
  "observation": {
    "location": "lab",
    "door": "closed"
  },
  "goal": "open the door",
  "metadata": {
    "request_id": "demo-001"
  },
  "include_trace": true
}
```

The endpoint runs interpretation → reasoning → planning and returns structured
results. It deliberately does not execute plan steps or mutate Agent memory.

## Tests

```powershell
python -m pip install httpx
pytest -q tests/test_api.py
```

The tests use a fake runtime; they do not require a live OpenCode request.
