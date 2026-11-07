import { afterEach, describe, expect, it, vi } from "vitest";
import { API_BASE_URL, downloadReport, setAccessTokenProvider } from "./client.js";

afterEach(() => {
  setAccessTokenProvider(null);
  vi.unstubAllGlobals();
});

const fileResponse = (disposition) =>
  new Response("PDFDATA", { status: 200, headers: { "Content-Disposition": disposition } });

describe("downloadReport", () => {
  it("sends the bearer token and returns the blob with the server's file name", async () => {
    const fetchMock = vi.fn().mockResolvedValue(fileResponse('attachment; filename="report-abcd1234.pdf"'));
    vi.stubGlobal("fetch", fetchMock);
    setAccessTokenProvider(async () => "tok-123");

    const { blob, filename } = await downloadReport("/api/v1/table-reports/reports/w1/download");

    expect(fetchMock).toHaveBeenCalledWith(
      `${API_BASE_URL}/api/v1/table-reports/reports/w1/download`,
      expect.objectContaining({ headers: { Authorization: "Bearer tok-123" } }),
    );
    expect(filename).toBe("report-abcd1234.pdf");
    expect(await blob.text()).toBe("PDFDATA");
  });

  it("falls back to a generic name when the header is missing", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response("x", { status: 200 })));
    expect((await downloadReport("/x")).filename).toBe("report");
  });

  it("surfaces the server's message for a failed download", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response(JSON.stringify({ detail: { message: "No such report file." } }), { status: 404 }),
      ),
    );
    await expect(downloadReport("/x")).rejects.toThrow("No such report file.");
  });

  it("refuses to send a request when the session has no usable token", async () => {
    const fetchMock = vi.fn();
    vi.stubGlobal("fetch", fetchMock);
    setAccessTokenProvider(async () => "");
    await expect(downloadReport("/x")).rejects.toThrow(/access token/);
    expect(fetchMock).not.toHaveBeenCalled();
  });
});
