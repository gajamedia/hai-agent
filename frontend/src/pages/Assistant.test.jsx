import { afterEach, describe, it, expect, vi, beforeEach } from "vitest";
import { cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import Assistant from "./Assistant";
import * as api from "../api/client";

vi.mock("../api/client", () => ({
  cancelDataChangePlan: vi.fn(),
  decideDataChange: vi.fn(),
  listDataChangeApprovals: vi.fn(),
  cancelDataEntry: vi.fn(),
  cancelReportPlan: vi.fn(),
  createDataChangePlan: vi.fn(),
  createDataChangePlanFromFile: vi.fn(),
  createDataEntry: vi.fn(),
  createDataEntryFromFile: vi.fn(),
  createReportPlan: vi.fn(),
  downloadReport: vi.fn(),
  getDataChangeConfig: vi.fn(),
  getDataChangePlan: vi.fn(),
  getDataEntryDatabases: vi.fn(),
  listDatabaseConnections: vi.fn(),
  runDataChangePlan: vi.fn(),
  runDataEntry: vi.fn(),
  runReportPlan: vi.fn(),
  cancelDatabasePlan: vi.fn(),
  createDatabasePlan: vi.fn(),
  createDatabasePlanFromFile: vi.fn(),
  createDatabasePlanFromImage: vi.fn(),
  getDatabaseCreationConfig: vi.fn(),
  getDatabasePlan: vi.fn(),
  routeAssistantFile: vi.fn(),
  routeAssistantMessage: vi.fn(),
  runDatabasePlan: vi.fn(),
}));
vi.mock("../auth/keycloak", () => ({ signInAgain: vi.fn() }));

const PLAN = {
  status: "planned", plan_id: "d1", profile: "sqlite", confirm_phrase: "BUAT DATABASE hai_toko",
  plan: { database: "hai_toko", tables: [{ name: "tb_bahan", columns: [{ name: "nama", type: "text", nullable: false }] }] },
};

const APPROVAL = {
  plan_id: "a1", owner: "budi", own: false, database: "toko", question: "hapus baris 3", expected_rows: 1,
  plan: { operation: "delete", table: "t", max_rows: 10 }, preview: {}, column_order: [],
};
const renderPage = () => render(<MemoryRouter><Assistant /></MemoryRouter>);
const say = async (text) => {
  await userEvent.type(screen.getByLabelText("Pesan"), text);
  await userEvent.click(screen.getByRole("button", { name: /Kirim/ }));
};

beforeEach(() => {
  vi.clearAllMocks();
  sessionStorage.clear();
  api.getDatabaseCreationConfig.mockResolvedValue({ profiles: [{ name: "sqlite", prefix: "hai_", type: "sqlite" }] });
});
afterEach(cleanup);

describe("Assistant", () => {
  it("routes a message about a new database into the creation pipeline and runs it after the phrase", async () => {
    api.routeAssistantMessage.mockResolvedValue({ intent: "create_database", question: "buat toko", database: null });
    api.createDatabasePlan.mockResolvedValue(PLAN);
    api.runDatabasePlan.mockResolvedValue({ database: "hai_toko", tables: ["tb_bahan"], profile: "sqlite", mode: "new" });
    renderPage();
    await say("buatkan database toko");
    expect(await screen.findByText(/Belum ada yang dibuat/)).toBeTruthy();
    expect(api.createDatabasePlan).toHaveBeenCalledWith({ question: "buat toko", profile: "sqlite" });
    const run = screen.getByRole("button", { name: "Jalankan" });
    expect(run.disabled).toBe(true);
    await userEvent.type(screen.getByLabelText("Frasa konfirmasi"), "BUAT DATABASE hai_toko");
    await userEvent.click(run);
    await waitFor(() => expect(api.runDatabasePlan).toHaveBeenCalledWith("d1", "BUAT DATABASE hai_toko", "d1"));
    expect(await screen.findByText(/Database hai_toko dibuat di server sqlite dengan tabel tb_bahan/)).toBeTruthy();
  });

  it("asks for the database name when a file is attached and then plans from the file", async () => {
    api.routeAssistantFile.mockResolvedValue({ intent: "create_database", question: "x", database: null, table: null });
    api.createDatabasePlanFromFile.mockResolvedValue(PLAN);
    renderPage();
    const file = new File(["a,b\n1,2\n"], "bahan.csv", { type: "text/csv" });
    await userEvent.upload(screen.getByLabelText("Lampirkan file"), file);
    await say("jadikan tabel");
    expect(await screen.findByText(/Nama database-nya apa/)).toBeTruthy();
    await say("hai_toko");
    await waitFor(() => expect(api.createDatabasePlanFromFile).toHaveBeenCalled());
    expect(api.createDatabasePlanFromFile.mock.calls[0][1]).toMatchObject({ database: "hai_toko", profile: "sqlite" });
  });

  it("records rows from a sentence: preview, then Simpan, with no phrase", async () => {
    api.routeAssistantMessage.mockResolvedValue({ intent: "record_data", question: "catat kopi", database: "toko" });
    api.getDataEntryDatabases.mockResolvedValue({ databases: [{ name: "toko", tables: ["tb_jual"] }, { name: "lain", tables: ["x"] }] });
    api.createDataEntry.mockResolvedValue({
      status: "planned", plan_id: "e1", database: "toko", expected_rows: 1, plan: { table: "tb_jual" },
      column_order: ["produk"], preview: { after: [{ produk: "kopi" }] },
    });
    api.runDataEntry.mockResolvedValue({ rows_affected: 1, table: "tb_jual", verification: "verified" });
    renderPage();
    await say("catat kopi");
    expect(await screen.findByText(/Belum ada yang tersimpan/)).toBeTruthy();
    expect(api.createDataEntry).toHaveBeenCalledWith({ question: "catat kopi", database: "toko" });
    await userEvent.click(screen.getByRole("button", { name: "Simpan" }));
    await waitFor(() => expect(api.runDataEntry).toHaveBeenCalledWith("e1", "e1", { confirmDuplicate: false }));
    expect(await screen.findByText(/Tersimpan. 1 baris dicatat pada tabel tb_jual/)).toBeTruthy();
  });

  it("asks which database when it cannot tell, then carries on with the pick", async () => {
    api.routeAssistantMessage.mockResolvedValue({ intent: "record_data", question: "catat", database: null });
    api.getDataEntryDatabases.mockResolvedValue({ databases: [{ name: "toko", tables: ["a"] }, { name: "gudang", tables: ["b"] }] });
    api.createDataEntry.mockResolvedValue({ status: "refused", reason: "tidak bisa" });
    renderPage();
    await say("catat sesuatu");
    await userEvent.click(await screen.findByRole("button", { name: "gudang" }));
    await waitFor(() => expect(api.createDataEntry).toHaveBeenCalledWith({ question: "catat", database: "gudang" }));
    expect(await screen.findByText("tidak bisa")).toBeTruthy();
  });

  it("shows a column mapping for a file and plans again with the person's mapping", async () => {
    api.routeAssistantFile.mockResolvedValue({ intent: "record_data", question: "x", database: "toko", table: "tb_jual" });
    api.getDataEntryDatabases.mockResolvedValue({ databases: [{ name: "toko", tables: ["tb_jual"] }] });
    api.createDataEntryFromFile.mockResolvedValueOnce({
      status: "needs_mapping", headers: ["Barang"], mapping: { Barang: "produk" }, columns: [{ name: "produk", type: "text" }],
    }).mockResolvedValueOnce({ status: "refused", reason: "selesai" });
    renderPage();
    await userEvent.upload(screen.getByLabelText("Lampirkan file"), new File(["Barang,x"], "j.csv"));
    await say("catat file ini");
    await userEvent.click(await screen.findByRole("button", { name: "Pakai pemetaan ini" }));
    await waitFor(() => expect(api.createDataEntryFromFile).toHaveBeenCalledTimes(2));
    expect(api.createDataEntryFromFile.mock.calls[1][1].mapping).toEqual({ Barang: "produk" });
  });

  it("changes data after the exact phrase, asking for a new login when the server says so", async () => {
    const keycloak = await import("../auth/keycloak");
    api.routeAssistantMessage.mockResolvedValue({ intent: "change_data", question: "naikkan harga", database: null });
    api.getDataChangeConfig.mockResolvedValue({ default_max_rows: 10, max_rows_ceiling: 100 });
    api.listDatabaseConnections.mockResolvedValue({ connections: [{ name: "toko", writable_tables: ["tb_jual"] }] });
    api.createDataChangePlan.mockResolvedValue({
      status: "planned", plan_id: "c1", confirm_phrase: "UBAH 1 BARIS", expected_rows: 1, state: "planned",
      plan: { operation: "update", table: "tb_jual", max_rows: 10, set: { harga: 5 } }, preview: {}, column_order: [],
    });
    api.runDataChangePlan.mockRejectedValue({ body: { detail: { error: "step_up_required" } } });
    renderPage();
    await say("naikkan harga kopi");
    expect(await screen.findByText(/Belum ada yang diubah/)).toBeTruthy();
    expect(api.createDataChangePlan).toHaveBeenCalledWith({ question: "naikkan harga", max_rows: 10, database: "toko" });
    const run = screen.getByRole("button", { name: "Jalankan" });
    expect(run.disabled).toBe(true);
    await userEvent.type(screen.getByLabelText("Frasa konfirmasi"), "UBAH 1 BARIS");
    await userEvent.click(run);
    await waitFor(() => expect(keycloak.signInAgain).toHaveBeenCalled());
    expect(api.runDataChangePlan).toHaveBeenCalledWith("c1", "UBAH 1 BARIS", "c1");
  });

  it("waits for a second person's approval before a change can run", async () => {
    api.routeAssistantMessage.mockResolvedValue({ intent: "change_data", question: "hapus", database: "toko" });
    api.getDataChangeConfig.mockResolvedValue(null);
    api.listDatabaseConnections.mockResolvedValue({ connections: [{ name: "toko", writable_tables: ["t"] }] });
    api.createDataChangePlan.mockResolvedValue({
      status: "planned", plan_id: "c2", confirm_phrase: "HAPUS", expected_rows: 1, state: "planned", requires_approval: true,
      plan: { operation: "delete", table: "t", max_rows: 10 }, preview: {}, column_order: [],
    });
    renderPage();
    await say("hapus baris");
    await screen.findByText(/perlu disetujui dulu/);
    await userEvent.type(screen.getByLabelText("Frasa konfirmasi"), "HAPUS");
    expect(screen.getByRole("button", { name: "Jalankan" }).disabled).toBe(true);
  });

  it("answers a question about data: plan, run, then the rows", async () => {
    api.routeAssistantMessage.mockResolvedValue({ intent: "view_data", question: "total penjualan", database: null });
    api.listDatabaseConnections.mockResolvedValue({ connections: [{ name: "toko", status: "active" }, { name: "off", status: "unavailable" }] });
    api.createReportPlan.mockResolvedValue({
      status: "planned", plan_id: "r1",
      plan: { table: "tb_jual", kind: "list", columns: ["produk"], limit: 10, filters: {}, format: "json" },
    });
    api.runReportPlan.mockResolvedValue({
      status: "succeeded", kind: "list", plan: { table: "tb_jual" }, rows: [{ produk: "kopi" }], row_count: 1,
    });
    renderPage();
    await say("total penjualan");
    await userEvent.click(await screen.findByRole("button", { name: "Layar saja (JSON)" }));
    expect(await screen.findByText(/belum ada data yang dibaca/)).toBeTruthy();
    expect(api.createReportPlan).toHaveBeenCalledWith({ question: "total penjualan", database: "toko" });
    await userEvent.click(screen.getByRole("button", { name: "Jalankan" }));
    expect(await screen.findByText("kopi")).toBeTruthy();
  });

  it("asks again when it is unsure and says so when the account may not use the pipeline", async () => {
    api.routeAssistantMessage.mockResolvedValueOnce({ intent: "unclear" })
      .mockResolvedValueOnce({ intent: "unclear", denied: "change_data" });
    renderPage();
    await say("halo");
    expect(await screen.findByText(/belum yakin/)).toBeTruthy();
    await say("ubah harga");
    expect(await screen.findByText(/tidak punya izin/)).toBeTruthy();
  });

  it("asks which server when several can create databases", async () => {
    api.getDatabaseCreationConfig.mockResolvedValue({ profiles: [{ name: "mysql_a" }, { name: "sqlite" }] });
    api.routeAssistantMessage.mockResolvedValue({ intent: "create_database", question: "buat toko" });
    api.createDatabasePlan.mockResolvedValue(PLAN);
    renderPage();
    await say("buat database toko");
    await userEvent.click(await screen.findByRole("button", { name: "sqlite" }));
    await waitFor(() => expect(api.createDatabasePlan).toHaveBeenCalledWith({ question: "buat toko", profile: "sqlite" }));
    // the pick stays in the conversation as the person's own answer, and a reload keeps it next to the question
    expect(screen.getAllByText("sqlite").length).toBe(1);
    expect(JSON.parse(sessionStorage.getItem("hai.assistant.history")).map((i) => i.text)).toContain("sqlite");
  });

  it("lets an approver approve a waiting change in the chat, asking for a new login first if needed", async () => {
    const keycloak = await import("../auth/keycloak");
    api.listDataChangeApprovals.mockResolvedValue({ approvals: [APPROVAL] });
    api.decideDataChange.mockRejectedValueOnce({ body: { detail: { error: "step_up_required" } } })
      .mockResolvedValueOnce({});
    renderPage();
    expect(await screen.findByText(/menunggu persetujuan Anda/)).toBeTruthy();
    await userEvent.click(screen.getByRole("button", { name: "Setujui" }));
    await waitFor(() => expect(keycloak.signInAgain).toHaveBeenCalled());
    await userEvent.click(screen.getByRole("button", { name: "Setujui" }));
    expect(await screen.findByText(/Perubahan disetujui/)).toBeTruthy();
    expect(api.decideDataChange).toHaveBeenLastCalledWith("a1", "approve");
  });

  it("stays quiet for a person who is not an approver", async () => {
    api.listDataChangeApprovals.mockRejectedValue({ body: { detail: { error: "permission_denied" } } });
    renderPage();
    await waitFor(() => expect(api.listDataChangeApprovals).toHaveBeenCalled());
    expect(screen.queryByText(/menunggu persetujuan Anda/)).toBeNull();
  });

  it("sends a picture straight to database design, warning that it goes to the AI, without asking the router", async () => {
    api.createDatabasePlanFromImage.mockResolvedValue({ ...PLAN, file: { name: "erd.png", kind: "image", sent_to_model: true } });
    renderPage();
    await userEvent.upload(screen.getByLabelText("Lampirkan file"), new File([new Uint8Array(8)], "erd.png", { type: "image/png" }));
    expect(screen.getByText(/dikirim utuh ke model AI/)).toBeTruthy();
    await userEvent.click(screen.getByRole("button", { name: /Kirim/ }));
    expect(await screen.findByText(/Gambar erd.png dikirim ke model AI/)).toBeTruthy();
    expect(api.createDatabasePlanFromImage.mock.calls[0][1]).toMatchObject({ profile: "sqlite" });
    expect(api.routeAssistantFile).not.toHaveBeenCalled();
    expect(api.createDatabasePlanFromFile).not.toHaveBeenCalled();
  });

  it("keeps the conversation when the page reloads (as after signing in again) and can start a new one", async () => {
    api.routeAssistantMessage.mockResolvedValue({ intent: "unclear" });
    const first = renderPage();
    await say("halo apa kabar");
    expect(await screen.findByText(/belum yakin/)).toBeTruthy();
    first.unmount();
    renderPage();                                                   // the page is loaded again
    expect(screen.getByText("halo apa kabar")).toBeTruthy();
    expect(screen.getByText(/belum yakin/)).toBeTruthy();
    await userEvent.click(screen.getByRole("button", { name: "Percakapan baru" }));
    expect(screen.queryByText("halo apa kabar")).toBeNull();
    expect(screen.getByText(/Tulis apa yang ingin dilakukan/)).toBeTruthy();
  });

  it("does not store rows of data in the saved conversation", async () => {
    api.routeAssistantMessage.mockResolvedValue({ intent: "record_data", question: "catat", database: "toko" });
    api.getDataEntryDatabases.mockResolvedValue({ databases: [{ name: "toko", tables: ["t"] }] });
    api.createDataEntry.mockResolvedValue({
      status: "planned", plan_id: "e9", database: "toko", expected_rows: 1, plan: { table: "t" },
      column_order: ["nama"], preview: { after: [{ nama: "RAHASIA-BUDI" }] },
    });
    renderPage();
    await say("catat budi");
    await screen.findByText(/Belum ada yang tersimpan/);
    expect(sessionStorage.getItem("hai.assistant.history")).not.toMatch(/RAHASIA-BUDI/);
  });

  it("does not keep a question whose buttons can no longer be pressed after a reload", async () => {
    api.getDatabaseCreationConfig.mockResolvedValue({ profiles: [{ name: "mysql_a" }, { name: "sqlite" }] });
    api.routeAssistantMessage.mockResolvedValue({ intent: "create_database", question: "buat toko" });
    const first = renderPage();
    await say("buat database toko");
    await screen.findByText("Dibuat di server mana?");
    first.unmount();
    renderPage();
    expect(screen.getByText("buat database toko")).toBeTruthy();
    expect(screen.queryByText("Dibuat di server mana?")).toBeNull();
  });

  it("asks for the report format only when looking at data, and offers the file for download", async () => {
    api.routeAssistantMessage.mockResolvedValue({ intent: "view_data", question: "daftar pegawai", database: null });
    api.listDatabaseConnections.mockResolvedValue({ connections: [{ name: "hr", status: "active" }] });
    api.createReportPlan.mockResolvedValue({
      status: "planned", plan_id: "r2", plan: { table: "pegawai", kind: "list", columns: ["nama"], limit: 10, filters: {}, format: "xlsx" },
    });
    api.runReportPlan.mockResolvedValue({
      status: "succeeded", kind: "list", plan: { table: "pegawai" }, rows: [{ nama: "Ani" }], row_count: 1,
      download: { url: "/x", filename: "laporan.xlsx" },
    });
    renderPage();
    expect(screen.queryByLabelText("Format file laporan")).toBeNull();            // no permanent selector any more
    await say("daftar pegawai");
    await userEvent.click(await screen.findByRole("button", { name: "Excel" }));
    await screen.findByText(/belum ada data yang dibaca/);
    expect(api.createReportPlan).toHaveBeenCalledWith({ question: "daftar pegawai", database: "hr", format: "xlsx" });
    await userEvent.click(screen.getByRole("button", { name: "Jalankan" }));
    expect(await screen.findByRole("button", { name: "Unduh XLSX" })).toBeTruthy();
  });

  it("does not ask for the format when the message already names it, and never for other requests", async () => {
    api.routeAssistantMessage.mockResolvedValueOnce({ intent: "view_data", question: "daftar pegawai", database: null, format: "pdf" })
      .mockResolvedValueOnce({ intent: "record_data", question: "catat" });
    api.listDatabaseConnections.mockResolvedValue({ connections: [{ name: "hr", status: "active" }] });
    api.createReportPlan.mockResolvedValue({ status: "refused", reason: "stop" });
    api.getDataEntryDatabases.mockResolvedValue({ databases: [{ name: "hr", tables: ["t"] }] });
    api.createDataEntry.mockResolvedValue({ status: "refused", reason: "stop2" });
    renderPage();
    await say("daftar pegawai dalam bentuk pdf");
    await screen.findByText("stop");
    expect(api.createReportPlan).toHaveBeenCalledWith({ question: "daftar pegawai", database: "hr", format: "pdf" });
    expect(screen.queryByText("Hasilnya dalam bentuk apa?")).toBeNull();
    await say("catat sesuatu");
    await screen.findByText("stop2");
    expect(screen.queryByText("Hasilnya dalam bentuk apa?")).toBeNull();
  });
});
