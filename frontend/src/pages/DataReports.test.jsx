import { afterEach, describe, it, expect, vi, beforeEach } from "vitest";
import { act, cleanup, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import DataReports, { PLAN_TIMEOUT_MS } from "./DataReports";
import * as api from "../api/client";

vi.mock("../api/client", () => ({
  listDatabaseConnections: vi.fn(),
  connectDatabase: vi.fn(),
  approveDatabaseTables: vi.fn(),
  disconnectDatabase: vi.fn(),
  createReportPlan: vi.fn(),
  runReportPlan: vi.fn(),
  cancelReportPlan: vi.fn(),
  downloadReport: vi.fn(),
  getDatabaseSchema: vi.fn(),
  listDatabaseProfiles: vi.fn(),
  listEngineRequests: vi.fn(),
  createEngineRequest: vi.fn(),
}));

const SCHEMA = {
  tables: [
    {
      name: "sales",
      columns: [
        { name: "id", declared_type: "INTEGER", primary_key: true },
        { name: "region", declared_type: "TEXT", primary_key: false },
        { name: "amount", declared_type: "REAL", primary_key: false },
      ],
    },
  ],
};

const PLAN = {
  status: "planned",
  plan_id: "p1",
  database: "sales",
  plan: {
    table: "sales",
    group_by: ["region"],
    metrics: [{ output: "revenue", operation: "sum", column: "amount" }],
    filters: { region: "west" },
  },
};

const renderPage = (state) =>
  render(
    <MemoryRouter initialEntries={[{ pathname: "/", state }]}>
      <DataReports />
    </MemoryRouter>,
  );

afterEach(cleanup);

describe("DataReports", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    api.listDatabaseConnections.mockResolvedValue({ connections: [] });
    api.listDatabaseProfiles.mockResolvedValue({ profiles: [] });
    api.listEngineRequests.mockResolvedValue({ requests: [] });
  });

  it("connects, approves chosen columns only, then shows a report", async () => {
    const user = userEvent.setup();
    api.connectDatabase.mockResolvedValue({ name: "sales", schema: SCHEMA });
    api.approveDatabaseTables.mockResolvedValue({});
    renderPage();

    await user.type(screen.getByLabelText(/Nama koneksi/), "sales");
    await user.type(screen.getByLabelText(/Nama file/), "sales.sqlite3");
    await user.click(screen.getByRole("button", { name: /Hubungkan/ }));
    expect(api.connectDatabase).toHaveBeenCalledWith({ name: "sales", filename: "sales.sqlite3" });

    await user.click(await screen.findByRole("checkbox", { name: "sales" }));
    await user.click(screen.getByRole("checkbox", { name: /amount/ })); // un-approve a column
    const filterBoxes = screen.getAllByRole("checkbox", { name: /boleh difilter/ });
    await user.click(filterBoxes[1]); // region
    await user.click(screen.getByRole("button", { name: /Setujui untuk sales/ }));

    expect(api.approveDatabaseTables).toHaveBeenCalledWith("sales", {
      sales: { columns: ["id", "region"], filterable_columns: ["region"], key_column: "id" },
    });
  });

  it("shows unsupported and failed outcomes without a table", async () => {
    const user = userEvent.setup();
    api.listDatabaseConnections.mockResolvedValue({
      connections: [{ name: "sales", status: "active", tables: ["sales"] }],
    });
    api.createReportPlan.mockResolvedValueOnce({
      status: "unsupported", reason: "no date filters", plan_id: null,
    });
    renderPage();
    await user.type(await screen.findByLabelText(/Pertanyaan/), "bulan lalu");
    await user.click(screen.getByRole("button", { name: /Buat rencana/ }));
    expect(await screen.findByText("no date filters")).toBeTruthy();
    expect(screen.queryByRole("table")).toBeNull();
    expect(screen.queryByRole("button", { name: "Jalankan" })).toBeNull();

    api.createReportPlan.mockResolvedValueOnce(PLAN);
    api.runReportPlan.mockResolvedValueOnce({ status: "failed", error_code: "no_matching_rows" });
    await user.click(screen.getByRole("button", { name: /Buat rencana/ }));
    await user.click(await screen.findByRole("button", { name: "Jalankan" }));
    expect(await screen.findByText(/no_matching_rows/)).toBeTruthy();
  });

  it("renders server errors from a rejected connect", async () => {
    const user = userEvent.setup();
    api.connectDatabase.mockRejectedValue(new Error("filename resolves outside the data folder"));
    renderPage();
    await user.type(screen.getByLabelText(/Nama koneksi/), "x1");
    await user.type(screen.getByLabelText(/Nama file/), "../a.sqlite3");
    await user.click(screen.getByRole("button", { name: /Hubungkan/ }));
    await waitFor(() => expect(screen.getByRole("alert").textContent).toMatch(/outside the data folder/));
  });

  it("renders result rows in a table", async () => {
    const user = userEvent.setup();
    api.listDatabaseConnections.mockResolvedValue({
      connections: [{ name: "sales", status: "active", tables: ["sales"] }],
    });
    api.createReportPlan.mockResolvedValue(PLAN);
    api.runReportPlan.mockResolvedValue({
      status: "succeeded",
      plan: PLAN.plan,
      rows: [{ region: "west", revenue: 217 }, { region: "east", revenue: 90 }],
    });
    renderPage();
    await user.type(await screen.findByLabelText(/Pertanyaan/), "total per wilayah");
    await user.click(screen.getByRole("button", { name: /Buat rencana/ }));

    // The plan is shown for review before anything runs.
    const card = await screen.findByLabelText("Rencana laporan");
    expect(card.textContent).toMatch(/belum ada data yang dibaca/);
    expect(card.textContent).toMatch(/Tabel: sales/);
    expect(card.textContent).toMatch(/sum\(amount\)/);
    expect(card.textContent).toMatch(/region = west/);
    expect(api.runReportPlan).not.toHaveBeenCalled();

    await user.click(screen.getByRole("button", { name: "Jalankan" }));
    expect(api.runReportPlan).toHaveBeenCalledTimes(1);
    const [planId, idempotencyKey] = api.runReportPlan.mock.calls[0];
    expect(planId).toBe("p1");
    expect(typeof idempotencyKey).toBe("string");
    expect(idempotencyKey.length).toBeGreaterThan(8);
    const table = await screen.findByRole("table", { name: "Hasil laporan" });
    expect(table.textContent).toContain("217");
    expect(table.querySelectorAll("tbody tr")).toHaveLength(2);
    expect(screen.queryByLabelText("Rencana laporan")).toBeNull();
  });

  it("lets the user cancel a plan without running it", async () => {
    const user = userEvent.setup();
    api.listDatabaseConnections.mockResolvedValue({
      connections: [{ name: "sales", status: "active", tables: ["sales"] }],
    });
    api.createReportPlan.mockResolvedValue(PLAN);
    api.cancelReportPlan.mockResolvedValue({ status: "cancelled" });
    renderPage();
    await user.type(await screen.findByLabelText(/Pertanyaan/), "x");
    await user.click(screen.getByRole("button", { name: /Buat rencana/ }));
    await user.click(await screen.findByRole("button", { name: "Batal" }));
    expect(api.cancelReportPlan).toHaveBeenCalledWith("p1");
    expect(api.runReportPlan).not.toHaveBeenCalled();
    expect(screen.queryByLabelText("Rencana laporan")).toBeNull();
  });

  it("flags a restored connection that needs re-approval and lets the user review it", async () => {
    const user = userEvent.setup();
    api.listDatabaseConnections.mockResolvedValue({
      connections: [{ name: "sales", status: "needs_reapproval", tables: [], reason: "schema changed" }],
    });
    api.getDatabaseSchema.mockResolvedValue(SCHEMA);
    renderPage();
    expect(await screen.findByText(/perlu persetujuan ulang: schema changed/)).toBeTruthy();
    await user.click(screen.getByRole("button", { name: "Setujui" }));
    expect(api.getDatabaseSchema).toHaveBeenCalledWith("sales");
    expect(await screen.findByRole("checkbox", { name: "sales" })).toBeTruthy();
  });

  it("lets an active connection's approval be changed, starting from what is approved now", async () => {
    const user = userEvent.setup();
    api.listDatabaseConnections.mockResolvedValue({
      connections: [{ name: "sales", status: "active", tables: ["sales"] }],
    });
    api.getDatabaseSchema.mockResolvedValue({
      ...SCHEMA, approved: { sales: { columns: ["id", "amount"], filterable_columns: ["amount"] } },
    });
    api.approveDatabaseTables.mockResolvedValue({ status: "active" });
    renderPage();
    await user.click(await screen.findByRole("button", { name: "Ubah persetujuan" }));
    expect((await screen.findByRole("checkbox", { name: "sales" })).checked).toBe(true);
    await user.click(screen.getByRole("button", { name: /Setujui untuk sales/ }));
    const [, tables] = api.approveDatabaseTables.mock.calls[0];
    expect(tables.sales.columns).toEqual(["id", "amount"]);
    expect(tables.sales.filterable_columns).toEqual(["amount"]);
  });

  it("offers write permissions only when the server has a writer account, and sends them with the approval", async () => {
    const user = userEvent.setup();
    api.listDatabaseConnections.mockResolvedValue({
      connections: [{ name: "sales", status: "active", tables: ["sales"] }],
    });
    api.getDatabaseSchema.mockResolvedValue({
      ...SCHEMA, writer_available: true,
      approved: { sales: { columns: ["id", "region", "amount"], filterable_columns: ["id", "region"], writable: null } },
    });
    api.approveDatabaseTables.mockResolvedValue({ status: "active" });
    renderPage();
    await user.click(await screen.findByRole("button", { name: "Ubah persetujuan" }));
    const group = await screen.findByLabelText("Izin ubah data sales");
    expect(screen.getAllByLabelText(/boleh diubah/)).toHaveLength(2);   // the key column is never offered
    await user.click(screen.getAllByLabelText(/boleh diubah/)[1]);     // amount
    await user.click(within(group).getByRole("checkbox", { name: "Ubah" }));
    await user.click(screen.getByRole("button", { name: /Setujui untuk sales/ }));
    const [, tables] = api.approveDatabaseTables.mock.calls[0];
    expect(tables.sales.writable).toEqual({ operations: ["update"], columns: ["amount"] });
  });

  it("arriving from Buat database opens the new database's approval already filled in, and sends it when approved", async () => {
    const user = userEvent.setup();
    api.listDatabaseConnections.mockResolvedValue({
      connections: [{ name: "hai_toko", status: "connected_pending_approval", tables: [] }] });
    api.getDatabaseSchema.mockResolvedValue({ ...SCHEMA, writer_available: true, approved: {} });
    api.approveDatabaseTables.mockResolvedValue({ status: "active" });
    renderPage({ review: "hai_toko", message: "Database baru dibuat: hai_toko. Sudah terhubung." });
    expect(await screen.findByText(/Database baru dibuat: hai_toko/)).toBeTruthy();
    expect(api.getDatabaseSchema).toHaveBeenCalledWith("hai_toko");
    expect((await screen.findByRole("checkbox", { name: "sales" })).checked).toBe(true);
    const group = await screen.findByLabelText("Izin ubah data sales");
    expect(within(group).getByRole("checkbox", { name: "Tambah" }).checked).toBe(true);
    expect(within(group).getByRole("checkbox", { name: "Ubah" }).checked).toBe(true);
    expect(within(group).getByRole("checkbox", { name: "Hapus" }).checked).toBe(false);      // never pre-ticked
    await user.click(screen.getByRole("button", { name: /Setujui untuk hai_toko/ }));
    const [name, tables] = api.approveDatabaseTables.mock.calls[0];
    expect(name).toBe("hai_toko");
    expect(tables.sales.columns).toEqual(["id", "region", "amount"]);
    expect(tables.sales.filterable_columns).toEqual(["id", "region", "amount"]);
    expect(tables.sales.writable).toEqual({ operations: ["insert", "update"], columns: ["region", "amount"] });   // not the key
    expect(tables.sales.key_column).toBe("id");
  });

  it("the preset keeps what was approved before and fills in only the new tables, and offers no writing without a writer", async () => {
    api.listDatabaseConnections.mockResolvedValue({ connections: [{ name: "x", status: "active", tables: ["sales"] }] });
    api.getDatabaseSchema.mockResolvedValue({
      tables: [...SCHEMA.tables, { name: "tb_baru", columns: [{ name: "id", declared_type: "INT", primary_key: true },
                                                              { name: "nama", declared_type: "TEXT", primary_key: false }] }],
      approved: { sales: { columns: ["id"], filterable_columns: [], writable: null } } });
    renderPage({ review: "x", message: "m" });
    expect((await screen.findByRole("checkbox", { name: "tb_baru" })).checked).toBe(true);     // new: filled in
    expect(screen.queryByLabelText(/Izin ubah data/)).toBeNull();                              // no writer account
    expect(screen.queryAllByRole("checkbox", { name: "region" })).toHaveLength(0);             // old table keeps its narrow choice
  });

  it("does not offer write permissions on a connection without a writer account", async () => {
    const user = userEvent.setup();
    api.listDatabaseConnections.mockResolvedValue({
      connections: [{ name: "sales", status: "connected_pending_approval", tables: [] }],
    });
    api.getDatabaseSchema.mockResolvedValue(SCHEMA);
    renderPage();
    await user.click(await screen.findByRole("button", { name: "Setujui" }));
    await screen.findByRole("checkbox", { name: "sales" });
    expect(screen.queryByLabelText(/boleh diubah/)).toBeNull();
    expect(screen.queryByLabelText(/Izin ubah data/)).toBeNull();
  });

  it("connects by server profile name without asking for a file or credentials", async () => {
    const user = userEvent.setup();
    api.listDatabaseProfiles.mockResolvedValue({ profiles: [{ name: "crm_pg", type: "postgresql" }] });
    api.connectDatabase.mockResolvedValue({ name: "crm", schema: SCHEMA, warnings: [] });
    renderPage();
    await user.type(screen.getByLabelText(/Nama koneksi/), "crm");
    await user.selectOptions(await screen.findByLabelText(/Sumber/), "crm_pg");
    expect(screen.queryByLabelText(/Nama file/)).toBeNull();
    expect(screen.queryByLabelText(/password/i)).toBeNull();
    await user.click(screen.getByRole("button", { name: /Hubungkan/ }));
    expect(api.connectDatabase).toHaveBeenCalledWith({ name: "crm", profile: "crm_pg" });
  });

  it("lets the user name a database made by the agent on a server that can create them, and says the connection name is only a label", async () => {
    const user = userEvent.setup();
    api.listDatabaseProfiles.mockResolvedValue({ profiles: [
      { name: "local_mysql", type: "mysql", prefix: "hai_" }, { name: "crm_pg", type: "postgresql" }] });
    api.connectDatabase.mockResolvedValue({ name: "testing", schema: SCHEMA, warnings: [] });
    renderPage();
    await user.type(screen.getByLabelText(/Nama koneksi/), "testing");
    const select = await screen.findByLabelText(/Sumber/);
    await user.selectOptions(select, "crm_pg");
    expect(screen.queryByLabelText("Database buatan agent")).toBeNull();           // no prefix: the field is not offered
    await user.selectOptions(select, "local_mysql");
    expect(screen.getByText(/hanya label/)).toBeTruthy();
    await user.type(screen.getByLabelText("Database buatan agent"), "hai_testingai");
    await user.click(screen.getByRole("button", { name: /Hubungkan/ }));
    expect(api.connectDatabase).toHaveBeenCalledWith({ name: "testing", profile: "local_mysql", database: "hai_testingai" });
  });

  it("shows where each connection comes from", async () => {
    api.listDatabaseConnections.mockResolvedValue({ connections: [{ name: "hai_testingai", status: "connected_pending_approval",
      database: "mysql", file: "local_mysql/hai_testingai" }] });
    renderPage();
    expect(await screen.findByText(/sumber: local_mysql\/hai_testingai/)).toBeTruthy();
  });

  it("lets the user ask for an unsupported database and shows the request status", async () => {
    const user = userEvent.setup();
    api.createEngineRequest.mockResolvedValue({ id: "r1" });
    api.listEngineRequests
      .mockResolvedValueOnce({ requests: [] })
      .mockResolvedValueOnce({ requests: [{ id: "r1", engine: "Oracle 19c", status: "open" }] });
    renderPage();
    await user.click(screen.getByText(/belum didukung/));
    await user.type(screen.getByLabelText("Nama database"), "Oracle 19c");
    await user.type(screen.getByLabelText("Alasan"), "finance");
    await user.click(screen.getByRole("button", { name: /Kirim permintaan/ }));
    expect(api.createEngineRequest).toHaveBeenCalledWith({ engine: "Oracle 19c", reason: "finance" });
    expect(await screen.findByText(/Oracle 19c · open/)).toBeTruthy();
  });

  it("sends the chosen file format with the plan and downloads the file with the token-based client", async () => {
    const user = userEvent.setup();
    api.listDatabaseConnections.mockResolvedValue({
      connections: [{ name: "sales", status: "active", tables: ["sales"] }],
    });
    api.createReportPlan.mockResolvedValue({ ...PLAN, plan: { ...PLAN.plan, format: "xlsx" } });
    api.runReportPlan.mockResolvedValue({
      status: "succeeded",
      plan: PLAN.plan,
      rows: [{ region: "west", revenue: 217 }],
      download: { url: "/api/v1/table-reports/reports/w1/download", filename: "report.xlsx" },
    });
    const blob = new Blob(["x"]);
    api.downloadReport.mockResolvedValue({ blob, filename: "report-abcd1234.xlsx" });
    const created = [];
    globalThis.URL.createObjectURL = vi.fn((b) => { created.push(b); return "blob:fake"; });
    globalThis.URL.revokeObjectURL = vi.fn();
    const clicked = [];
    const realClick = HTMLAnchorElement.prototype.click;
    HTMLAnchorElement.prototype.click = function () { clicked.push(this.download); };

    try {
      renderPage();
      await user.type(await screen.findByLabelText(/Pertanyaan/), "total");
      await user.selectOptions(screen.getByLabelText("Hasil"), "xlsx");
      await user.click(screen.getByRole("button", { name: /Buat rencana/ }));
      expect(api.createReportPlan).toHaveBeenCalledWith(
        expect.objectContaining({ question: "total", format: "xlsx" }),
        expect.any(AbortSignal),
      );
      expect((await screen.findByLabelText("Rencana laporan")).textContent).toMatch(/File: XLSX/);
      await user.click(screen.getByRole("button", { name: "Jalankan" }));
      await user.click(await screen.findByRole("button", { name: "Unduh XLSX" }));
      expect(api.downloadReport).toHaveBeenCalledWith("/api/v1/table-reports/reports/w1/download");
      expect(created).toEqual([blob]);
      expect(clicked).toEqual(["report-abcd1234.xlsx"]);
      expect(globalThis.URL.revokeObjectURL).toHaveBeenCalledWith("blob:fake");
    } finally {
      HTMLAnchorElement.prototype.click = realClick;
    }
  });

  it("omits the format when the user keeps the default and shows no download button without a file", async () => {
    const user = userEvent.setup();
    api.listDatabaseConnections.mockResolvedValue({
      connections: [{ name: "sales", status: "active", tables: ["sales"] }],
    });
    api.createReportPlan.mockResolvedValue(PLAN);
    api.runReportPlan.mockResolvedValue({ status: "succeeded", plan: PLAN.plan, rows: [{ region: "w", revenue: 1 }], download: null });
    renderPage();
    await user.type(await screen.findByLabelText(/Pertanyaan/), "total");
    await user.click(screen.getByRole("button", { name: /Buat rencana/ }));
    expect(api.createReportPlan.mock.calls[0][0].format).toBeUndefined();
    await user.click(await screen.findByRole("button", { name: "Jalankan" }));
    await screen.findByRole("table", { name: "Hasil laporan" });
    expect(screen.queryByRole("button", { name: /Unduh/ })).toBeNull();
  });

  it("shows a spinning indicator, counts the seconds and lets the user stop waiting", async () => {
    const user = userEvent.setup();
    api.listDatabaseConnections.mockResolvedValue({
      connections: [{ name: "sales", status: "active", tables: ["sales"] }],
    });
    let seenSignal;
    api.createReportPlan.mockImplementation(
      (_payload, signal) =>
        new Promise((_resolve, reject) => {
          seenSignal = signal;
          signal.addEventListener("abort", () => reject(new DOMException("Aborted", "AbortError")));
        }),
    );
    renderPage();
    await user.type(await screen.findByLabelText(/Pertanyaan/), "total");
    await user.click(screen.getByRole("button", { name: /Buat rencana/ }));

    const status = await screen.findByRole("status");
    expect(status.textContent).toMatch(/Agent sedang menyusun rencana/);
    expect(status.textContent).toMatch(/0 detik/);
    expect(status.querySelector("svg.spin")).not.toBeNull();                 // it really animates
    expect(screen.getByRole("button", { name: /Buat rencana/ }).querySelector("svg.spin")).not.toBeNull();
    expect(screen.getByRole("button", { name: /Buat rencana/ }).disabled).toBe(true);

    await user.click(screen.getByRole("button", { name: "Batalkan" }));
    expect(seenSignal.aborted).toBe(true);
    expect(await screen.findByText("Pembuatan rencana dibatalkan.")).toBeTruthy();
    expect(screen.queryByRole("status")).toBeNull();
    expect(screen.getByRole("button", { name: /Buat rencana/ }).disabled).toBe(false);   // can try again
  });

  it("counts elapsed seconds and gives up after four minutes with a pointer to the check tool", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    try {
      const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
      api.listDatabaseConnections.mockResolvedValue({
        connections: [{ name: "sales", status: "active", tables: ["sales"] }],
      });
      api.createReportPlan.mockImplementation(
        (_payload, signal) =>
          new Promise((_resolve, reject) => {
            signal.addEventListener("abort", () => reject(new DOMException("Aborted", "AbortError")));
          }),
      );
      renderPage();
      await user.type(await screen.findByLabelText(/Pertanyaan/), "total");
      await user.click(screen.getByRole("button", { name: /Buat rencana/ }));
      await screen.findByRole("status");

      await act(async () => { vi.advanceTimersByTime(7000); });
      expect(screen.getByRole("status").textContent).toMatch(/[67] detik/);

      await act(async () => { vi.advanceTimersByTime(PLAN_TIMEOUT_MS); });
      expect(await screen.findByText(/lebih dari 4 menit/)).toBeTruthy();
      expect(screen.getByRole("alert").textContent).toMatch(/check_llm_provider\.py/);
      expect(screen.queryByRole("status")).toBeNull();
      expect(screen.getByRole("button", { name: /Buat rencana/ }).disabled).toBe(false);
    } finally {
      vi.useRealTimers();
    }
  });

  it("does not leave the waiting notice behind after a normal answer", async () => {
    const user = userEvent.setup();
    api.listDatabaseConnections.mockResolvedValue({
      connections: [{ name: "sales", status: "active", tables: ["sales"] }],
    });
    api.createReportPlan.mockResolvedValue(PLAN);
    renderPage();
    await user.type(await screen.findByLabelText(/Pertanyaan/), "total");
    await user.click(screen.getByRole("button", { name: /Buat rencana/ }));
    await screen.findByLabelText("Rencana laporan");
    expect(screen.queryByRole("status")).toBeNull();
  });

  it("shows a listing plan as columns, order and a row limit instead of group and metrics", async () => {
    const user = userEvent.setup();
    api.listDatabaseConnections.mockResolvedValue({
      connections: [{ name: "sales", status: "active", tables: ["sales"] }],
    });
    api.createReportPlan.mockResolvedValue({
      status: "planned", plan_id: "p9", database: "sales",
      plan: {
        kind: "list", table: "sales", group_by: [], metrics: [], filters: { region: "west" }, format: "pdf",
        columns: ["id", "region", "amount"], limit: 50, order_by: [{ column: "amount", direction: "desc" }],
      },
    });
    renderPage();
    await user.type(await screen.findByLabelText(/Pertanyaan/), "tampilkan data");
    await user.click(screen.getByRole("button", { name: /Buat rencana/ }));
    const card = (await screen.findByLabelText("Rencana laporan")).textContent;
    expect(card).toMatch(/daftar baris/);
    expect(card).toMatch(/Kolom: id, region, amount/);
    expect(card).toMatch(/Urutan: amount menurun/);
    expect(card).toMatch(/Maksimal: 50 baris/);
    expect(card).toMatch(/region = west/);
    expect(card).not.toMatch(/Kelompokkan per|Hitung:/);
  });

  it("shows comparison conditions in plain Indonesian before anything runs", async () => {
    const user = userEvent.setup();
    api.listDatabaseConnections.mockResolvedValue({
      connections: [{ name: "sales", status: "active", tables: ["sales"] }],
    });
    api.createReportPlan.mockResolvedValue({
      status: "planned", plan_id: "p10", database: "sales",
      plan: {
        kind: "list", table: "sales", group_by: [], metrics: [], filters: {}, format: "json",
        columns: ["id"], limit: 100, order_by: [],
        conditions: [
          { column: "harga_beli", operator: "lt", value: 30000 },
          { column: "tanggal", operator: "between", value: ["2026-03-01", "2026-03-31"] },
          { column: "nama", operator: "contains", value: "gula" },
          { column: "kode", operator: "is_null", value: null },
        ],
      },
    });
    renderPage();
    await user.type(await screen.findByLabelText(/Pertanyaan/), "murah");
    await user.click(screen.getByRole("button", { name: /Buat rencana/ }));
    const card = (await screen.findByLabelText("Rencana laporan")).textContent;
    expect(card).toMatch(/harga_beli < 30000/);
    expect(card).toMatch(/tanggal antara 2026-03-01 dan 2026-03-31/);
    expect(card).toMatch(/nama mengandung "gula"/);
    expect(card).toMatch(/kode kosong/);
  });

  it("says plainly when a listing was cut off, and how many rows the file holds", async () => {
    const user = userEvent.setup();
    api.listDatabaseConnections.mockResolvedValue({
      connections: [{ name: "sales", status: "active", tables: ["sales"] }],
    });
    const plan = { kind: "list", table: "sales", group_by: [], metrics: [], filters: {}, format: "xlsx",
                   columns: ["id"], limit: 1000, order_by: [] };
    api.createReportPlan.mockResolvedValue({ status: "planned", plan_id: "p9", database: "sales", plan });
    api.runReportPlan.mockResolvedValue({
      status: "succeeded", kind: "list", plan, row_count: 1000, limit_reached: true,
      rows: Array.from({ length: 200 }, (_, i) => ({ id: i + 1 })),
      download: { url: "/x", filename: "report.xlsx" },
    });
    renderPage();
    await user.type(await screen.findByLabelText(/Pertanyaan/), "tampilkan semua");
    await user.click(screen.getByRole("button", { name: /Buat rencana/ }));
    await user.click(await screen.findByRole("button", { name: "Jalankan" }));
    await screen.findByRole("table", { name: "Hasil laporan" });
    const page = document.body.textContent;
    expect(page).toMatch(/menampilkan 200 dari 1000 baris/);
    expect(page).toMatch(/Dibatasi 1000 baris pertama/);
    expect(page).toMatch(/Masih ada baris lain di database/);
    expect(page).toMatch(/Layar menampilkan 200 baris pertama\. File yang diunduh memuat semua 1000 baris/);
    expect(screen.getByRole("table", { name: "Hasil laporan" }).querySelectorAll("tbody tr")).toHaveLength(200);
  });

  it("does not warn about a cut-off when the whole listing fits", async () => {
    const user = userEvent.setup();
    api.listDatabaseConnections.mockResolvedValue({
      connections: [{ name: "sales", status: "active", tables: ["sales"] }],
    });
    const plan = { kind: "list", table: "sales", group_by: [], metrics: [], filters: {}, format: "json",
                   columns: ["id"], limit: 100, order_by: [] };
    api.createReportPlan.mockResolvedValue({ status: "planned", plan_id: "p9", database: "sales", plan });
    api.runReportPlan.mockResolvedValue({
      status: "succeeded", kind: "list", plan, row_count: 3, limit_reached: false,
      rows: [{ id: 1 }, { id: 2 }, { id: 3 }], download: null,
    });
    renderPage();
    await user.type(await screen.findByLabelText(/Pertanyaan/), "tampilkan");
    await user.click(screen.getByRole("button", { name: /Buat rencana/ }));
    await user.click(await screen.findByRole("button", { name: "Jalankan" }));
    await screen.findByRole("table", { name: "Hasil laporan" });
    expect(document.body.textContent).toMatch(/menampilkan 3 dari 3 baris/);
    expect(document.body.textContent).not.toMatch(/Dibatasi|Layar menampilkan/);
  });

  it("keeps the summary wording for an aggregate report", async () => {
    const user = userEvent.setup();
    api.listDatabaseConnections.mockResolvedValue({
      connections: [{ name: "sales", status: "active", tables: ["sales"] }],
    });
    api.createReportPlan.mockResolvedValue(PLAN);
    api.runReportPlan.mockResolvedValue({
      status: "succeeded", kind: "aggregate", plan: PLAN.plan, row_count: 1, limit_reached: false,
      rows: [{ region: "west", revenue: 217 }],
    });
    renderPage();
    await user.type(await screen.findByLabelText(/Pertanyaan/), "total");
    await user.click(screen.getByRole("button", { name: /Buat rencana/ }));
    expect((await screen.findByLabelText("Rencana laporan")).textContent).toMatch(/Kelompokkan per: region/);
    await user.click(screen.getByRole("button", { name: "Jalankan" }));
    await screen.findByRole("table", { name: "Hasil laporan" });
    expect(document.body.textContent).toMatch(/Laporan selesai: sales per region/);
    expect(document.body.textContent).not.toMatch(/Daftar baris dari/);
  });
});

describe("DataReports relations", () => {
  const TWO = {
    tables: [
      { name: "tb_bahan", columns: [
        { name: "id", declared_type: "INTEGER", primary_key: true },
        { name: "id_jenis", declared_type: "INTEGER", primary_key: false },
        { name: "id_gudang", declared_type: "INTEGER", primary_key: false }] },
      { name: "tb_jenisbahan", columns: [{ name: "id", declared_type: "INTEGER", primary_key: true }] },
      { name: "tb_gudang", columns: [{ name: "id", declared_type: "INTEGER", primary_key: true }] },
    ],
    foreign_keys: [{ table: "tb_bahan", column: "id_jenis", ref_table: "tb_jenisbahan", ref_column: "id" }],
  };

  beforeEach(() => {
    vi.clearAllMocks();
    api.listDatabaseConnections.mockResolvedValue({ connections: [] });
    api.listDatabaseProfiles.mockResolvedValue({ profiles: [] });
    api.listEngineRequests.mockResolvedValue({ requests: [] });
    api.connectDatabase.mockResolvedValue({ name: "sales", schema: TWO });
    api.approveDatabaseTables.mockResolvedValue({});
  });

  const connect = async (user) => {
    renderPage();
    await user.type(screen.getByLabelText(/Nama koneksi/), "sales");
    await user.type(screen.getByLabelText(/Nama file/), "sales.sqlite3");
    await user.click(screen.getByRole("button", { name: /Hubungkan/ }));
  };

  it("offers the database foreign key and sends it, plus one the user declares", async () => {
    const user = userEvent.setup();
    await connect(user);
    for (const table of ["tb_bahan", "tb_jenisbahan", "tb_gudang"]) {
      await user.click(await screen.findByRole("checkbox", { name: table }));
    }
    const fk = screen.getByRole("checkbox", { name: /tb_bahan.id_jenis = tb_jenisbahan.id/ });
    expect(fk.checked).toBe(true);

    const add = screen.getByLabelText("Tambah relasi");
    await user.selectOptions(within(add).getByLabelText("Tabel"), "tb_bahan");
    await user.selectOptions(within(add).getByLabelText("Kolom Tabel"), "id_gudang");
    await user.selectOptions(within(add).getByLabelText("dengan tabel"), "tb_gudang");
    await user.selectOptions(within(add).getByLabelText("Kolom dengan tabel"), "id");
    await user.click(within(add).getByRole("button", { name: "Tambah relasi" }));
    expect(screen.getByText(/dibuat manual/)).toBeTruthy();

    await user.click(screen.getByRole("button", { name: /Setujui untuk sales/ }));
    const [, , , relations] = api.approveDatabaseTables.mock.calls[0];
    expect(relations).toEqual([
      { table: "tb_bahan", column: "id_jenis", ref_table: "tb_jenisbahan", ref_column: "id" },
      { table: "tb_bahan", column: "id_gudang", ref_table: "tb_gudang", ref_column: "id" },
    ]);
  });

  it("sends no relation the user unticked", async () => {
    const user = userEvent.setup();
    await connect(user);
    for (const table of ["tb_bahan", "tb_jenisbahan"]) {
      await user.click(await screen.findByRole("checkbox", { name: table }));
    }
    await user.click(screen.getByRole("checkbox", { name: /tb_bahan.id_jenis = tb_jenisbahan.id/ }));
    await user.click(screen.getByRole("button", { name: /Setujui untuk sales/ }));
    expect(api.approveDatabaseTables.mock.calls[0]).toHaveLength(2); // no relations argument
  });
});

describe("DataReports relation suggestions and join hint", () => {
  const SCHEMA_GUESS = {
    tables: [
      { name: "tb_bahan", columns: [
        { name: "id", declared_type: "INTEGER", primary_key: true },
        { name: "id_jenis", declared_type: "INTEGER", primary_key: false }] },
      { name: "tb_jenisbahan", columns: [{ name: "id", declared_type: "INTEGER", primary_key: true }] },
    ],
    foreign_keys: [],
    suggested_relations: [{
      table: "tb_bahan", column: "id_jenis", ref_table: "tb_jenisbahan", ref_column: "id",
      reason: "nama kolom id_jenis mirip nama tabel tb_jenisbahan",
    }],
  };

  beforeEach(() => {
    vi.clearAllMocks();
    api.listDatabaseConnections.mockResolvedValue({ connections: [] });
    api.listDatabaseProfiles.mockResolvedValue({ profiles: [] });
    api.listEngineRequests.mockResolvedValue({ requests: [] });
    api.connectDatabase.mockResolvedValue({ name: "sales", schema: SCHEMA_GUESS });
    api.approveDatabaseTables.mockResolvedValue({});
  });

  it("shows a suggestion from the column name, unticked, and sends it only when ticked", async () => {
    const user = userEvent.setup();
    renderPage();
    await user.type(screen.getByLabelText(/Nama koneksi/), "sales");
    await user.type(screen.getByLabelText(/Nama file/), "sales.sqlite3");
    await user.click(screen.getByRole("button", { name: /Hubungkan/ }));
    for (const table of ["tb_bahan", "tb_jenisbahan"]) {
      await user.click(await screen.findByRole("checkbox", { name: table }));
    }
    const guess = screen.getByRole("checkbox", { name: /tb_bahan.id_jenis = tb_jenisbahan.id/ });
    expect(guess.checked).toBe(false);
    expect(screen.getByText(/saran dari nama: nama kolom id_jenis mirip/)).toBeTruthy();

    await user.click(guess);
    await user.click(screen.getByRole("button", { name: /Setujui untuk sales/ }));
    expect(api.approveDatabaseTables.mock.calls[0][3]).toEqual([
      { table: "tb_bahan", column: "id_jenis", ref_table: "tb_jenisbahan", ref_column: "id" },
    ]);
  });

  it("points to the relation form when the question needed a join", async () => {
    const user = userEvent.setup();
    api.listDatabaseConnections.mockResolvedValue({
      connections: [{ name: "sales", status: "active", tables: ["tb_bahan", "tb_jenisbahan"] }],
    });
    api.createReportPlan.mockResolvedValueOnce({
      status: "unsupported", reason: "join belum didukung", plan_id: null, needs_join: true,
    });
    renderPage();
    await user.type(await screen.findByLabelText(/Pertanyaan/), "bahan dan jenisnya");
    await user.click(screen.getByRole("button", { name: /Buat rencana/ }));
    expect(await screen.findByText("join belum didukung")).toBeTruthy();
    expect(screen.getByText(/Relasi antar tabel/, { selector: "b" })).toBeTruthy();

    api.createReportPlan.mockResolvedValueOnce({
      status: "unsupported", reason: "no date filters", plan_id: null, needs_join: false,
    });
    await user.click(screen.getByRole("button", { name: /Buat rencana/ }));
    expect(await screen.findByText("no date filters")).toBeTruthy();
    expect(screen.queryByText(/Relasi antar tabel/, { selector: "b" })).toBeNull();
  });
});

describe("DataReports relation found by the AI", () => {
  it("explains the guess in plain words before the plan is run", async () => {
    const user = userEvent.setup();
    api.listDatabaseConnections.mockResolvedValue({
      connections: [{ name: "sales", status: "active", tables: ["tb_bahan", "tb_jenisbahan"] }],
    });
    api.listDatabaseProfiles.mockResolvedValue({ profiles: [] });
    api.listEngineRequests.mockResolvedValue({ requests: [] });
    api.createReportPlan.mockResolvedValueOnce({
      status: "planned", plan_id: "p9", database: "sales",
      plan: {
        kind: "list", table: "tb_bahan", format: "json", columns: ["tb_bahan.kode", "tb_jenisbahan.nama"],
        limit: 100, order_by: [], filters: {}, conditions: [],
        joins: [{ table: "tb_jenisbahan", type: "left", on: ["tb_jenisbahan.id", "tb_bahan.id_jenis"] }],
        proposed_relations: [
          { table: "tb_bahan", column: "id_jenis", ref_table: "tb_jenisbahan", ref_column: "id" },
        ],
      },
    });
    renderPage();
    await user.type(await screen.findByLabelText(/Pertanyaan/), "bahan dan jenisnya");
    await user.click(screen.getByRole("button", { name: /Buat rencana/ }));
    const notice = await screen.findByLabelText("Relasi usulan AI");
    expect(within(notice).getByText(/AI menduga hubungan antar tabel/)).toBeTruthy();
    expect(notice.textContent).toMatch(/kolom id_jenis di tabel tb_bahan sama dengan kolom id di tabel tb_jenisbahan/);
    expect(screen.getByRole("button", { name: "Jalankan" })).toBeTruthy();
  });
});
