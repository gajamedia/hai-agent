export const API_BASE_URL = (
  import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000"
).replace(/\/+$/, "");

let accessTokenProvider = null;

/**
 * Register the authenticated-session token provider.
 * Pass null during auth teardown or in isolated tests.
 */
export function setAccessTokenProvider(provider) {
  if (provider !== null && typeof provider !== "function") {
    throw new TypeError("Access token provider must be a function or null.");
  }
  accessTokenProvider = provider;
}

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
    let message =
      (typeof detail === "object" ? detail?.message : detail) ||
      body?.message ||
      body?.error ||
      `HTTP ${response.status}`;
    // A 403 names the permission it needed; showing it tells the user which role to ask for.
    if (response.status === 403 && typeof detail === "object" && typeof detail?.permission === "string") {
      message = `${message} (izin yang dibutuhkan: ${detail.permission})`;
    }

    const error = new Error(message);
    error.status = response.status;
    error.body = body;
    throw error;
  }

  return body;
}

function isPublicEndpoint(path) {
  return path === "/health" || path.startsWith("/health?");
}

async function request(path, options = {}) {
  const headers = { ...(options.headers || {}) };

  if (!isPublicEndpoint(path) && accessTokenProvider) {
    const token = await accessTokenProvider();
    if (!token || typeof token !== "string") {
      throw new Error("Sesi Keycloak tidak memiliki access token yang valid.");
    }
    headers.Authorization = `Bearer ${token}`;
  }

  const response = await fetch(`${API_BASE_URL}${path}`, {
    ...options,
    headers,
  });
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
  return request(`/api/v1/telemetry/capabilities?${query.toString()}`, {
    signal,
  });
}

export async function getCapabilityTelemetrySources({ limit = 20, signal } = {}) {
  const query = new URLSearchParams({ limit: String(limit) });
  return request(
    `/api/v1/telemetry/capabilities/sources?${query.toString()}`,
    { signal },
  );
}

export async function getStrategies(signal) {
  return request("/api/v1/strategies", { signal });
}

const ENVIRONMENT_WORKSPACE_BASE = "/api/v1/capabilities/environment-workspace";

export async function getEnvironmentProviders(signal) {
  return request(`${ENVIRONMENT_WORKSPACE_BASE}/providers`, { signal });
}

export async function getEnvironmentInstances(signal) {
  return request(`${ENVIRONMENT_WORKSPACE_BASE}/instances`, { signal });
}

export async function getEnvironmentInstanceCapabilities(instanceId, signal) {
  if (!instanceId || typeof instanceId !== "string") {
    throw new TypeError("Environment instance ID is required.");
  }
  return request(
    `${ENVIRONMENT_WORKSPACE_BASE}/instances/${encodeURIComponent(instanceId)}/capabilities`,
    { signal },
  );
}

export async function getExecutionState(signal) {
  return request("/api/v1/execution/state", { signal });
}

export async function runExecution(payload, signal) {
  return request("/api/v1/execution/run", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
    signal,
  });
}

export async function runCognitivePlan(
  payload,
  signal,
  idempotencyKey,
) {
  const headers = {
    "Content-Type": "application/json",
  };

  if (idempotencyKey) {
    headers["Idempotency-Key"] = idempotencyKey;
  }

  return request("/api/v1/execution/plan-run", {
    method: "POST",
    headers,
    body: JSON.stringify(payload),
    signal,
  });
}

export async function resetExecution(signal) {
  return request("/api/v1/execution/reset", {
    method: "POST",
    signal,
  });
}

const AUTONOMOUS_TESTING_BASE = "/api/v1/autonomous-testing";

export async function createAutonomousTestBatch(payload, signal) {
  return request(`${AUTONOMOUS_TESTING_BASE}/batches`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
    signal,
  });
}

export async function startAutonomousTestBatch(batchId, signal) {
  return request(
    `${AUTONOMOUS_TESTING_BASE}/batches/${encodeURIComponent(batchId)}/run`,
    { method: "POST", signal },
  );
}

export async function getAutonomousTestBatches({ limit = 50, signal } = {}) {
  const query = new URLSearchParams({ limit: String(limit) });
  return request(`${AUTONOMOUS_TESTING_BASE}/batches?${query.toString()}`, {
    signal,
  });
}

export async function getAutonomousTestBatch(batchId, signal) {
  return request(
    `${AUTONOMOUS_TESTING_BASE}/batches/${encodeURIComponent(batchId)}`,
    { signal },
  );
}

export async function getAutonomousTestBatchResults(batchId, signal) {
  return request(
    `${AUTONOMOUS_TESTING_BASE}/batches/${encodeURIComponent(batchId)}/results`,
    { signal },
  );
}

const API_V1 = "/api/v1";
const jsonPost = (path, payload, signal, extraHeaders = {}) =>
  request(path, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...extraHeaders },
    body: JSON.stringify(payload),
    signal,
  });

export async function listDatabaseConnections(signal) {
  return request(`${API_V1}/database-connections`, { signal });
}

export async function connectDatabase(payload, signal) {
  return jsonPost(`${API_V1}/database-connections`, payload, signal);
}

export async function refreshDatabaseConnection(name, signal) {
  return jsonPost(`${API_V1}/database-connections/${encodeURIComponent(name)}/refresh`, {}, signal);
}

export async function approveDatabaseTables(name, tables, signal) {
  return jsonPost(
    `${API_V1}/database-connections/${encodeURIComponent(name)}/approve`,
    { tables },
    signal,
  );
}

export async function disconnectDatabase(name, signal) {
  return request(`${API_V1}/database-connections/${encodeURIComponent(name)}`, {
    method: "DELETE",
    signal,
  });
}

export async function askTableReport(payload, signal) {
  return jsonPost(`${API_V1}/table-reports`, payload, signal);
}

export async function getDatabaseSchema(name, signal) {
  return request(`${API_V1}/database-connections/${encodeURIComponent(name)}/schema`, { signal });
}

export async function listDatabaseProfiles(signal) {
  return request(`${API_V1}/database-connections/profiles`, { signal });
}

export async function listEngineRequests(signal) {
  return request(`${API_V1}/database-engine-requests`, { signal });
}

export async function createEngineRequest(payload, signal) {
  return jsonPost(`${API_V1}/database-engine-requests`, payload, signal);
}

// Two-phase reports: generate a plan for review (reads no data), then run exactly that plan.
export async function createReportPlan(payload, signal) {
  return jsonPost(`${API_V1}/table-reports/plans`, payload, signal);
}

export async function runReportPlan(planId, idempotencyKey, signal) {
  return jsonPost(
    `${API_V1}/table-reports/plans/${encodeURIComponent(planId)}/run`,
    {},
    signal,
    idempotencyKey ? { "Idempotency-Key": idempotencyKey } : {},
  );
}

export async function cancelReportPlan(planId, signal) {
  return request(`${API_V1}/table-reports/plans/${encodeURIComponent(planId)}`, {
    method: "DELETE",
    signal,
  });
}

// Report files need the bearer token, so a plain link cannot be used: fetch, then save the blob.
export async function downloadReport(url, signal) {
  const headers = {};
  if (accessTokenProvider) {
    const token = await accessTokenProvider();
    if (!token || typeof token !== "string") {
      throw new Error("Sesi Keycloak tidak memiliki access token yang valid.");
    }
    headers.Authorization = `Bearer ${token}`;
  }
  const response = await fetch(`${API_BASE_URL}${url}`, { headers, signal });
  if (!response.ok) {
    await parseResponse(response); // throws a readable error from the JSON body
    throw new Error(`HTTP ${response.status}`);
  }
  const disposition = response.headers.get("Content-Disposition") || "";
  const match = /filename="([^"]+)"/.exec(disposition);
  return { blob: await response.blob(), filename: match ? match[1] : "report" };
}

// Data changes (insert / update / delete). Running needs a fresh login and the exact phrase.
export async function getDataChangeConfig(signal) {
  return request(`${API_V1}/data-changes/config`, { signal });
}

export async function createDataChangePlan(payload, signal) {
  return jsonPost(`${API_V1}/data-changes/plans`, payload, signal);
}

export async function getDataChangePlan(planId, signal) {
  return request(`${API_V1}/data-changes/plans/${encodeURIComponent(planId)}`, { signal });
}

export async function runDataChangePlan(planId, confirm, idempotencyKey, signal) {
  return jsonPost(
    `${API_V1}/data-changes/plans/${encodeURIComponent(planId)}/run`,
    { confirm },
    signal,
    idempotencyKey ? { "Idempotency-Key": idempotencyKey } : {},
  );
}

export async function cancelDataChangePlan(planId, signal) {
  return request(`${API_V1}/data-changes/plans/${encodeURIComponent(planId)}`, {
    method: "DELETE",
    signal,
  });
}

export async function listDataChangeApprovals(signal) {
  return request(`${API_V1}/data-changes/approvals`, { signal });
}

export async function decideDataChange(planId, decision, signal) {
  return jsonPost(
    `${API_V1}/data-changes/approvals/${encodeURIComponent(planId)}/${decision === "reject" ? "reject" : "approve"}`,
    {},
    signal,
  );
}

// Creating a new database (and tables): preview, fresh login, exact phrase.
export async function getDatabaseCreationConfig(signal) {
  return request(`${API_V1}/database-creation/config`, { signal });
}

export async function createDatabasePlan(payload, signal) {
  return jsonPost(`${API_V1}/database-creation/plans`, payload, signal);
}

export async function getDatabasePlan(planId, signal) {
  return request(`${API_V1}/database-creation/plans/${encodeURIComponent(planId)}`, { signal });
}

export async function runDatabasePlan(planId, confirm, idempotencyKey, signal) {
  return jsonPost(
    `${API_V1}/database-creation/plans/${encodeURIComponent(planId)}/run`,
    { confirm },
    signal,
    idempotencyKey ? { "Idempotency-Key": idempotencyKey } : {},
  );
}

export async function cancelDatabasePlan(planId, signal) {
  return request(`${API_V1}/database-creation/plans/${encodeURIComponent(planId)}`, {
    method: "DELETE",
    signal,
  });
}
