import { afterEach, describe, it, expect, vi, beforeEach } from "vitest";
import { cleanup, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import DataEntry from "./DataEntry";
import * as api from "../api/client";

vi.mock("../api/client", () => ({
  cancelDataEntry: vi.fn(),
  createDataEntry: vi.fn(),
  getDataEntryDatabases: vi.fn(),
  runDataEntry: vi.fn(),
}));

const ENTRY = {
  status: "planned", plan_id: "e1", database: "dbsales", expected_rows: 2,
  plan: { operation: "insert", table: "sales" },
  preview: { before: [], after: [{ product: "kopi", amount: 25 }, { product: "teh", amount: 15 }], note: "" },
};

afterEach(cleanup);

async function prepare(user) {
  await user.type(await screen.findByLabelText(/Yang dicatat/), "catat kopi dan teh");
  await user.click(screen.getByRole("button", { name: "Siapkan" }));
  return screen.findByLabelText("Yang akan dicatat");
}

describe("DataEntry", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    api.getDataEntryDatabases.mockResolvedValue({ databases: [{ name: "dbsales", tables: ["sales"] }] });
    api.createDataEntry.mockResolvedValue(ENTRY);
    api.runDataEntry.mockResolvedValue({ status: "committed", rows_affected: 2, table: "sales", verification: "verified" });
  });

  it("shows the rows first and records nothing until Simpan is pressed", async () => {
    const user = userEvent.setup();
    render(<DataEntry />);
    await prepare(user);
    expect(screen.getByText("kopi")).toBeTruthy();
    expect(api.runDataEntry).not.toHaveBeenCalled();
    await user.click(screen.getByRole("button", { name: "Simpan" }));
    expect(await screen.findByText(/Tersimpan/)).toBeTruthy();
    expect(api.runDataEntry).toHaveBeenCalledWith("e1", "e1"); // the plan id is the idempotency key
    expect(screen.queryByLabelText("Yang akan dicatat")).toBeNull();
  });

  it("asks for no phrase and offers Batal", async () => {
    const user = userEvent.setup();
    api.cancelDataEntry.mockResolvedValue({ status: "cancelled" });
    render(<DataEntry />);
    await prepare(user);
    expect(screen.queryByLabelText("Frasa konfirmasi")).toBeNull();
    await user.click(screen.getByRole("button", { name: "Batal" }));
    expect(api.cancelDataEntry).toHaveBeenCalledWith("e1");
    expect(screen.queryByLabelText("Yang akan dicatat")).toBeNull();
  });

  it("says why when the request is not a new record", async () => {
    const user = userEvent.setup();
    api.createDataEntry.mockResolvedValue({ status: "unsupported", reason: "Gunakan menu Ubah data.", plan_id: null });
    render(<DataEntry />);
    await user.type(await screen.findByLabelText(/Yang dicatat/), "hapus semua");
    await user.click(screen.getByRole("button", { name: "Siapkan" }));
    expect(await screen.findByText("Gunakan menu Ubah data.")).toBeTruthy();
    expect(screen.queryByLabelText("Yang akan dicatat")).toBeNull();
  });

  it("explains what to ask the administrator when no table accepts new data", async () => {
    api.getDataEntryDatabases.mockResolvedValue({ databases: [] });
    render(<DataEntry />);
    expect(await screen.findByText(/Belum ada tabel yang disetujui untuk pencatatan/)).toBeTruthy();
    expect(screen.queryByRole("button", { name: "Siapkan" })).toBeNull();
  });

  it("makes the person pick a database when several accept new data", async () => {
    const user = userEvent.setup();
    api.getDataEntryDatabases.mockResolvedValue({ databases: [
      { name: "dbsales", tables: ["sales"] }, { name: "hai_hris", tables: ["tb_cuti"] }] });
    render(<DataEntry />);
    await user.type(await screen.findByLabelText(/Yang dicatat/), "catat cuti");
    expect(screen.getByRole("button", { name: "Siapkan" }).disabled).toBe(true); // no database chosen yet
    await user.click(screen.getByRole("radio", { name: "hai_hris" }));
    await user.click(screen.getByRole("button", { name: "Siapkan" }));
    expect(api.createDataEntry).toHaveBeenCalledWith({ question: "catat cuti", database: "hai_hris" });
  });

  it("shows only the first tables of a database with a count, however many it has", async () => {
    const tables = Array.from({ length: 30 }, (_, i) => `tb_${i}`);
    api.getDataEntryDatabases.mockResolvedValue({ databases: [
      { name: "big", tables }, { name: "small", tables: ["a"] }] });
    render(<DataEntry />);
    expect(await screen.findByText("30 tabel dapat diisi")).toBeTruthy();
    expect(screen.getByText("+27 lainnya")).toBeTruthy();
    expect(screen.queryByText("tb_10")).toBeNull();
  });
});
