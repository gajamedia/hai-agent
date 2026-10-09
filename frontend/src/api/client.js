const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000').replace(/\/+$/, '');
async function parseResponse(response) {
  const text = await response.text(); let body;
  try { body = text ? JSON.parse(text) : {}; } catch { body = { message: text || 'Respons kosong dari server.' }; }
  if (!response.ok) { const error = new Error(body?.detail?.message || body?.message || body?.detail || body?.error || `HTTP ${response.status}`); error.status = response.status; error.body = body; throw error; }
  return body;
}
export async function checkHealth(signal) { return parseResponse(await fetch(`${API_BASE_URL}/health`, { signal })); }
export async function runCognitive(payload, signal) { return parseResponse(await fetch(`${API_BASE_URL}/api/v1/cognitive/run`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload), signal })); }
export { API_BASE_URL };
