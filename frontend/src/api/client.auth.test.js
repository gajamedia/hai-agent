import { afterEach, describe, expect, it, vi } from "vitest";
import {
  checkHealth,
  getExecutionState,
  setAccessTokenProvider,
} from "./client.js";

function response(body = {}, status = 200) {
  return {
    ok: status >= 200 && status < 300,
    status,
    text: async () => JSON.stringify(body),
  };
}

describe("API client Keycloak bearer token integration", () => {
  afterEach(() => {
    setAccessTokenProvider(null);
    vi.restoreAllMocks();
  });

  it("attaches a refreshed bearer token to protected API requests", async () => {
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValue(response({ state: { box: "closed" } }));
    const tokenProvider = vi.fn().mockResolvedValue("test-access-token");

    setAccessTokenProvider(tokenProvider);

    await getExecutionState();

    expect(tokenProvider).toHaveBeenCalledOnce();
    expect(fetchMock).toHaveBeenCalledOnce();
    const [, options] = fetchMock.mock.calls[0];
    expect(options.headers.Authorization).toBe("Bearer test-access-token");
  });

  it("does not attach a bearer token to the public health endpoint", async () => {
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValue(response({ status: "ok" }));
    const tokenProvider = vi.fn().mockResolvedValue("test-access-token");

    setAccessTokenProvider(tokenProvider);

    await checkHealth();

    expect(tokenProvider).not.toHaveBeenCalled();
    const [, options] = fetchMock.mock.calls[0];
    expect(options.headers.Authorization).toBeUndefined();
  });

  it("does not send a protected request when token refresh fails", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch");
    setAccessTokenProvider(vi.fn().mockRejectedValue(new Error("refresh failed")));

    await expect(getExecutionState()).rejects.toThrow("refresh failed");
    expect(fetchMock).not.toHaveBeenCalled();
  });
});
