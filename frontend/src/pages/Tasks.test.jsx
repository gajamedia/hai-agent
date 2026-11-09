import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { cleanup, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import Tasks from "./Tasks";
import * as api from "../api/client";

vi.mock("../api/client", () => ({
  listTasks: vi.fn(),
  getTask: vi.fn(),
  downloadReport: vi.fn(),
}));

const base = { created_at: 1_760_000_000, updated_at: 1_760_000_100, expires_in_seconds: null, approved_by: null,
  verification: null, error: null };
const DONE = { ...base, task_id: "t-done", operation: "database.change", label: "Ubah data",
  title: "Hapus data: 2 baris di tb_pegawai", status: "completed", verification: "verified" };
const WAITING = { ...base, task_id: "t-wait", operation: "database.change", label: "Ubah data",
  title: "Hapus data pada tabel tb_pegawai", status: "awaiting_approval", expires_in_seconds: 600 };
const UNVERIFIED = { ...base, task_id: "t-un", operation: "database.change", label: "Ubah data",
  title: "Ubah data: 1 baris di tb_pegawai", status: "unverified", verification: "unverified" };
const REPORT = { ...base, task_id: "t-rep", operation: "database.report", label: "Laporan tabel",
  title: "omzet per wilayah (4 baris)", status: "completed", verification: "verified" };

const renderPage = () => render(<MemoryRouter><Tasks /></MemoryRouter>);

beforeEach(() => {
  api.listTasks.mockResolvedValue({ tasks: [DONE, WAITING], has_more: false });
});
afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

describe("Riwayat tugas", () => {
  it("lists the tasks with an honest status and verification", async () => {
    api.listTasks.mockResolvedValue({ tasks: [DONE, WAITING, UNVERIFIED], has_more: false });
    renderPage();
    const table = await screen.findByRole("table", { name: "Tugas" });
    expect(within(table).getByText("Hapus data: 2 baris di tb_pegawai")).toBeTruthy();
    expect(within(table).getByText("Selesai")).toBeTruthy();
    expect(within(table).getByText("Terverifikasi")).toBeTruthy();
    expect(within(table).getByText("Menunggu persetujuan orang lain")).toBeTruthy();
    // a result that could not be confirmed is never shown as plainly "Selesai"
    expect(within(table).getByText("Selesai, hasil belum terverifikasi")).toBeTruthy();
  });

  it("says so when there is nothing yet", async () => {
    api.listTasks.mockResolvedValue({ tasks: [], has_more: false });
    renderPage();
    expect(await screen.findByText("Belum ada tugas")).toBeTruthy();
    expect(screen.queryByRole("table", { name: "Tugas" })).toBeNull();
  });

  it("shows a readable error and keeps the page usable", async () => {
    api.listTasks.mockRejectedValue(Object.assign(new Error("HTTP 503"), { body: { detail: { message: "Server sibuk." } } }));
    renderPage();
    expect((await screen.findByRole("alert")).textContent).toContain("Server sibuk.");
    expect(screen.getByRole("button", { name: /Muat ulang/ })).toBeTruthy();
  });

  it("asks for the group the person picks", async () => {
    const user = userEvent.setup();
    renderPage();
    await screen.findByRole("table", { name: "Tugas" });
    await user.selectOptions(screen.getByLabelText("Tampilkan"), "waiting");
    await waitFor(() => expect(api.listTasks).toHaveBeenLastCalledWith(expect.objectContaining({ status: "waiting" })));
    await user.selectOptions(screen.getByLabelText("Tampilkan"), "");
    await waitFor(() => expect(api.listTasks).toHaveBeenLastCalledWith(expect.objectContaining({ status: undefined })));
  });

  it("tells when only the newest are shown", async () => {
    api.listTasks.mockResolvedValue({ tasks: [DONE], has_more: true });
    renderPage();
    expect(await screen.findByText(/Menampilkan tugas terbaru saja/)).toBeTruthy();
  });
});

describe("Rincian tugas", () => {
  const TIMELINE = [
    { at: 1_760_000_000, event: "task_planned", actor: "alice" },
    { at: 1_760_000_050, event: "task_approved", actor: "bob", self_approved: false },
    { at: 1_760_000_060, event: "task_executing", actor: "alice" },
    { at: 1_760_000_061, event: "task_completed", actor: "alice", verdict: "verified" },
  ];

  it("shows the timeline in words, with who did what", async () => {
    const user = userEvent.setup();
    api.getTask.mockResolvedValue({ ...DONE, approved_by: "bob", result: { rows_affected: 2, table: "tb_pegawai" },
      download: null, timeline: TIMELINE });
    renderPage();
    await user.click(await screen.findByRole("button", { name: /Lihat rincian: Hapus data: 2 baris/ }));
    const panel = await screen.findByLabelText("Rincian tugas");
    const steps = within(await within(panel).findByRole("list", { name: "Jejak tugas" })).getAllByRole("listitem");
    expect(steps.map((li) => li.querySelector("b").textContent)).toEqual([
      "Rencana dibuat", "Disetujui", "Mulai dijalankan", "Terverifikasi dan selesai"]);
    expect(steps[1].textContent).toContain("oleh bob");
    expect(within(panel).getByLabelText("Hasil").textContent).toContain("rows_affected: 2");
    expect(api.getTask).toHaveBeenCalledWith("t-done", expect.anything());
  });

  it("warns that an unverified result needs checking", async () => {
    const user = userEvent.setup();
    api.listTasks.mockResolvedValue({ tasks: [UNVERIFIED], has_more: false });
    api.getTask.mockResolvedValue({ ...UNVERIFIED, result: null, download: null, timeline: TIMELINE });
    renderPage();
    await user.click(await screen.findByRole("button", { name: /Lihat rincian/ }));
    expect((await screen.findByRole("status")).textContent).toContain("belum bisa dipastikan");
  });

  it("points an open task to the page where it is continued", async () => {
    const user = userEvent.setup();
    api.getTask.mockResolvedValue({ ...WAITING, result: null, download: null, timeline: TIMELINE.slice(0, 1) });
    renderPage();
    await user.click(await screen.findByRole("button", { name: /Lihat rincian: Hapus data pada tabel/ }));
    const link = await screen.findByRole("link", { name: /Buka halamannya untuk melanjutkan/ });
    expect(link.getAttribute("href")).toBe("/database-agent/changes");
  });

  it("downloads a report file through the authenticated client", async () => {
    const user = userEvent.setup();
    api.listTasks.mockResolvedValue({ tasks: [REPORT], has_more: false });
    api.getTask.mockResolvedValue({ ...REPORT, result: { row_count: 4 }, timeline: TIMELINE.slice(0, 1),
      download: { url: "/api/v1/table-reports/reports/w1/download", filename: "laporan.xlsx" } });
    api.downloadReport.mockResolvedValue({ blob: new Blob(["x"]), filename: "report-w1.xlsx" });
    globalThis.URL.createObjectURL = vi.fn(() => "blob:x");
    globalThis.URL.revokeObjectURL = vi.fn();
    renderPage();
    await user.click(await screen.findByRole("button", { name: /Lihat rincian/ }));
    await user.click(await screen.findByRole("button", { name: /Unduh laporan.xlsx/ }));
    await waitFor(() => expect(api.downloadReport).toHaveBeenCalledWith("/api/v1/table-reports/reports/w1/download"));
    expect(globalThis.URL.revokeObjectURL).toHaveBeenCalled();
  });

  it("shows why a detail could not be loaded", async () => {
    const user = userEvent.setup();
    api.getTask.mockRejectedValue(Object.assign(new Error("HTTP 404"), { body: { detail: { message: "Task not found." } } }));
    renderPage();
    await user.click(await screen.findByRole("button", { name: /Lihat rincian: Hapus data: 2 baris/ }));
    expect((await within(await screen.findByLabelText("Rincian tugas")).findByRole("alert")).textContent)
      .toContain("Task not found.");
  });
});
