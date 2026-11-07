import { afterEach, describe, expect, it, vi } from "vitest";

// keycloak-js is not needed to test the helper and must not start a real instance.
vi.mock("keycloak-js", () => ({ default: class { init() {} } }));

import { currentPageRedirectUri } from "./keycloak.js";

afterEach(() => window.history.pushState({}, "", "/"));

describe("currentPageRedirectUri", () => {
  it("returns the page the user asked for, so a reload does not land on the home dashboard", () => {
    window.history.pushState({}, "", "/data-reports");
    expect(currentPageRedirectUri()).toBe(`${window.location.origin}/data-reports`);
  });

  it("keeps the query string but drops any fragment", () => {
    window.history.pushState({}, "", "/data-reports?database=sales#state=abc&code=xyz");
    expect(currentPageRedirectUri()).toBe(`${window.location.origin}/data-reports?database=sales`);
  });

  it("is the root for the root page", () => {
    window.history.pushState({}, "", "/");
    expect(currentPageRedirectUri()).toBe(`${window.location.origin}/`);
  });
});
