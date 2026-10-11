
export const API_BASE_URL = (
  import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000"
).replace(/\/+$/, "");

async function parseResponse(response) {
  const text = await response.text();
  let body;

  try {
    body = text ? JSON.parse(text) : {};
  } catch {
    body = { message: text || "Respons kosong dari server." };
  }

  if (!response.ok) {
    const detail = body?.detail;
    const message =
      (typeof detail === "object" ? detail?.message : detail) ||
      body?.message ||
      body?.error ||
      `HTTP ${response.status}`;

    const error = new Error(message);
    error.status = response.status;
    error.body = body;
    throw error;
  }

  return body;
}

async function request(path, options = {}) {
  const response = await fetch(`${API_BASE_URL}${path}`, options);
  return parseResponse(response);
}

export async function checkHealth(signal) {
  return request("/health", { signal });
}

export async function runCognitive(payload, signal) {
  return request("/api/v1/cognitive/run", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
    signal,
  });
}

export async function getCapabilityTelemetry({
  limit = 20,
  signal,
} = {}) {
  const query = new URLSearchParams({ limit: String(limit) });

  return request(
    `/api/v1/telemetry/capabilities?${query.toString()}`,
    { signal },
  );
}

/**
 * Read the current simulation state and available actions.
 */
export async function getExecutionState(signal) {
  return request("/api/v1/execution/state", { signal });
}

/**
 * Execute one action or an ordered list of actions.
 *
 * Example:
 * {
 *   actions: ["open_box", "take_key", "unlock_door"],
 *   goal_description: "Unlock the door",
 *   target_state: { door: "unlocked" }
 * }
 */
export async function runExecution(payload, signal) {
  return request("/api/v1/execution/run", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
    signal,
  });
}

/**
 * Reset the simulation environment.
 * Backend telemetry is expected to be retained.
 */
export async function resetExecution(signal) {
  return request("/api/v1/execution/reset", {
    method: "POST",
    signal,
  });
}
