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

export async function checkHealth(signal) {
  const response = await fetch(`${API_BASE_URL}/health`, { signal });
  return parseResponse(response);
}

export async function runCognitive(payload, signal) {
  const response = await fetch(`${API_BASE_URL}/api/v1/cognitive/run`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
    signal,
  });

  return parseResponse(response);
}

export async function getCapabilityTelemetry({ limit = 20, signal } = {}) {
  const query = new URLSearchParams({ limit: String(limit) });
  const response = await fetch(
    `${API_BASE_URL}/api/v1/telemetry/capabilities?${query.toString()}`,
    { signal },
  );

  return parseResponse(response);
}
