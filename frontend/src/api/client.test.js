
import { afterEach, describe, expect, it, vi } from "vitest";

import {
  API_BASE_URL,
  runCognitivePlan,
  runExecution,
} from "./client.js";

afterEach(() => {
  vi.unstubAllGlobals();
});

function mockJsonResponse(body, status = 200) {
  return {
    ok: status >= 200 && status < 300,
    status,
    text: async () => JSON.stringify(body),
  };
}

describe("API client", () => {
  it("uses the configured default API base URL", () => {
    expect(API_BASE_URL).toMatch(/^https?:\/\//);
    expect(API_BASE_URL).not.toMatch(/\/$/);
  });

  it("sends a plan-run request with its idempotency key", async () => {
    const payload = {
      observation: { description: "A closed box." },
      goal_description: "Open the box",
      target_state: { box: "open" },
      actions: ["open_box"],
    };

    const responseBody = {
      success: true,
      state_after: { box: "open" },
    };

    const fetchMock = vi.fn().mockResolvedValue(
      mockJsonResponse(responseBody),
    );

    vi.stubGlobal("fetch", fetchMock);

    const result = await runCognitivePlan(
      payload,
      undefined,
      "test-idempotency-key",
    );

    expect(fetchMock).toHaveBeenCalledTimes(1);

    const [url, options] = fetchMock.mock.calls[0];

    expect(url).toBe(`${API_BASE_URL}/api/v1/execution/plan-run`);
    expect(options.method).toBe("POST");
    expect(options.headers).toEqual({
      "Content-Type": "application/json",
      "Idempotency-Key": "test-idempotency-key",
    });
    expect(JSON.parse(options.body)).toEqual(payload);
    expect(result).toEqual(responseBody);
  });

  it("does not send an idempotency header when no key is provided", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      mockJsonResponse({ success: true }),
    );

    vi.stubGlobal("fetch", fetchMock);

    await runCognitivePlan({
      goal_description: "Open the box",
      target_state: { box: "open" },
      actions: ["open_box"],
    });

    const [, options] = fetchMock.mock.calls[0];

    expect(options.headers).toEqual({
      "Content-Type": "application/json",
    });
    expect(options.headers["Idempotency-Key"]).toBeUndefined();
  });

  it("preserves structured unknown-outcome errors for reconciliation", async () => {
    const errorBody = {
      detail: {
        error: "idempotency_outcome_unknown",
        message: "Execution outcome is unknown.",
        reconciliation_required: true,
        retry_safe: false,
      },
    };

    const fetchMock = vi.fn().mockResolvedValue(
      mockJsonResponse(errorBody, 409),
    );

    vi.stubGlobal("fetch", fetchMock);

    await expect(
      runCognitivePlan(
        {
          goal_description: "Open the box",
          target_state: { box: "open" },
          actions: ["open_box"],
        },
        undefined,
        "uncertain-key",
      ),
    ).rejects.toMatchObject({
      message: "Execution outcome is unknown.",
      status: 409,
      body: errorBody,
    });
  });

  it("sends execution requests to the correct endpoint", async () => {
    const payload = {
      actions: ["open_box"],
      goal_description: "Open the box",
      target_state: { box: "open" },
    };

    const fetchMock = vi.fn().mockResolvedValue(
      mockJsonResponse({ success: true }),
    );

    vi.stubGlobal("fetch", fetchMock);

    await runExecution(payload);

    const [url, options] = fetchMock.mock.calls[0];

    expect(url).toBe(`${API_BASE_URL}/api/v1/execution/run`);
    expect(options.method).toBe("POST");
    expect(JSON.parse(options.body)).toEqual(payload);
  });
});
