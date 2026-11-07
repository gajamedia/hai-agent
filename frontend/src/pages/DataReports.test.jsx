import { afterEach, describe, it, expect, vi, beforeEach } from "vitest";
import { cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import DataReports from "./DataReports";
import * as api from "../api/client";

vi.mock("../api/client", () => ({
  listDatabaseConnections: vi.fn(),
  connectDatabase: vi.fn(),
  approveDatabaseTables: vi.fn(),
  disconnectDatabase: vi.fn(),
  askTableReport: vi.fn(),
  getDatabaseSchema: vi.fn(),
  listDatabaseProfiles: vi.fn(),
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

const renderPage = () =>
  render(
    <MemoryRouter>
      <DataReports />
    </MemoryRouter>,
  );

afterEach(cleanup);

describe("DataReports", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    api.listDatabaseConnections.mockResolvedValue({ connections: [] });
    api.listDatabaseProfiles.mockResolvedValue({ profiles: [] });
  });

  it("connects, approves chosen columns only, then shows a report", async () => {
    const user = userEvent.setup();
    api.connectDatabase.mockResolvedValue({ name: "sales", schema: SCHEMA });
    api.approveDatabaseTables.mockResolvedValue({});
    api.askTableReport.mockResolvedValue({
      status: "succeeded",
      plan: { table: "sales", group_by: ["region"], filters: { region: "west" } },
      rows: [{ region: "west", revenue: 217 }],
    });
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
    api.askTableReport.mockResolvedValueOnce({ status: "unsupported", reason: "no date filters" });
    renderPage();
    await user.type(await screen.findByLabelText(/Pertanyaan/), "bulan lalu");
    await user.click(screen.getByRole("button", { name: /Buat laporan/ }));
    expect(await screen.findByText("no date filters")).toBeTruthy();
    expect(screen.queryByRole("table")).toBeNull();

    api.askTableReport.mockResolvedValueOnce({ status: "failed", error_code: "no_matching_rows" });
    await user.click(screen.getByRole("button", { name: /Buat laporan/ }));
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
    api.askTableReport.mockResolvedValue({
      status: "succeeded",
      plan: { table: "sales", group_by: ["region"], filters: {} },
      rows: [{ region: "west", revenue: 217 }, { region: "east", revenue: 90 }],
    });
    renderPage();
    await user.type(await screen.findByLabelText(/Pertanyaan/), "total per wilayah");
    await user.click(screen.getByRole("button", { name: /Buat laporan/ }));
    const table = await screen.findByRole("table", { name: "Hasil laporan" });
    expect(table.textContent).toContain("217");
    expect(table.querySelectorAll("tbody tr")).toHaveLength(2);
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
});
