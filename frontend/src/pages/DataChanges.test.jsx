import { afterEach, describe, it, expect, vi, beforeEach } from "vitest";
import { cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import DataChanges from "./DataChanges";
import * as api from "../api/client";
import * as keycloak from "../auth/keycloak";

vi.mock("../api/client", () => ({
  cancelDataChangePlan: vi.fn(),
  createDataChangePlan: vi.fn(),
  decideDataChange: vi.fn(),
  getDataChangeConfig: vi.fn(),
  getDataChangePlan: vi.fn(),
  listDataChangeApprovals: vi.fn(),
  listDatabaseConnections: vi.fn(),
  runDataChangePlan: vi.fn(),
}));
vi.mock("../auth/keycloak", () => ({ signInAgain: vi.fn() }));

const PLAN = {
  status: "planned", plan_id: "c1", database: "dbsales", question: "turunkan harga",
  confirm_phrase: "UBAH 2 BARIS tb_bahan", expected_rows: 2, requires_approval: false, state: "planned",
  plan: {
    operation: "update", table: "tb_bahan", max_rows: 10, set: { harga_beli: 5000 },
    conditions: [{ column: "harga_beli", operator: "lt", value: 13000 }],
  },
  preview: {
    before: [{ id: 1, harga_beli: 11000 }, { id: 2, harga_beli: 12000 }],
    after: [{ id: 1, harga_beli: 5000 }, { id: 2, harga_beli: 5000 }], note: "",
  },
};

const renderPage = () => render(<MemoryRouter><DataChanges /></MemoryRouter>);

async function planIt(user) {
  await user.type(await screen.findByLabelText(/Perintah/), "turunkan harga");
  await user.click(screen.getByRole("button", { name: /Buat rencana/ }));
  return screen.findByLabelText("Rencana perubahan");
}

afterEach(cleanup);

describe("DataChanges", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    sessionStorage.clear();
    api.getDataChangeConfig.mockResolvedValue({ max_rows_ceiling: 500, default_max_rows: 100 });
    api.listDatabaseConnections.mockResolvedValue({
      connections: [{ name: "dbsales", status: "active", tables: ["tb_bahan"], writable_tables: ["tb_bahan"] }],
    });
    api.listDataChangeApprovals.mockRejectedValue(Object.assign(new Error("forbidden"), { status: 403 }));
    api.createDataChangePlan.mockResolvedValue(PLAN);
  });

  it("explains how to enable changes when no table is approved for writing", async () => {
    api.listDatabaseConnections.mockResolvedValue({ connections: [{ name: "x", status: "active", tables: ["t"] }] });
    renderPage();
    expect(await screen.findByText(/Belum ada koneksi yang tabelnya disetujui untuk diubah/)).toBeTruthy();
    expect(screen.queryByRole("button", { name: /Buat rencana/ })).toBeNull();
  });

  it("sends the user's own row limit, capped by the server ceiling shown to them", async () => {
    const user = userEvent.setup();
    renderPage();
    const limit = await screen.findByLabelText(/Maksimal baris/);
    expect(limit.max).toBe("500");
    await user.clear(limit);
    await user.type(limit, "25");
    await planIt(user);
    expect(api.createDataChangePlan).toHaveBeenCalledWith({
      question: "turunkan harga", max_rows: 25, database: "dbsales" });
  });

  it("shows what would change, before and after, and nothing has run yet", async () => {
    const user = userEvent.setup();
    renderPage();
    const card = (await planIt(user)).textContent;
    expect(card).toMatch(/Ubah data.*tb_bahan/);
    expect(card).toMatch(/harga_beli = 5000/);
    expect(card).toMatch(/harga_beli < 13000/);
    expect(card).toMatch(/Jumlah baris terdampak: 2/);
    expect(screen.getByLabelText("Sebelum")).toBeTruthy();
    expect(screen.getByLabelText("Sesudah")).toBeTruthy();
    expect(api.runDataChangePlan).not.toHaveBeenCalled();
  });

  it("only enables Jalankan once the exact phrase is typed, then runs it", async () => {
    const user = userEvent.setup();
    api.runDataChangePlan.mockResolvedValue({
      status: "committed", rows_affected: 2, operation: "update", table: "tb_bahan", replayed: false });
    renderPage();
    await planIt(user);
    const run = screen.getByRole("button", { name: "Jalankan" });
    expect(run.disabled).toBe(true);
    const box = screen.getByLabelText("Frasa konfirmasi");
    await user.type(box, "ubah 2 baris");
    expect(run.disabled).toBe(true);
    await user.clear(box);
    await user.type(box, "UBAH 2 BARIS tb_bahan");
    expect(run.disabled).toBe(false);
    await user.click(run);
    expect(api.runDataChangePlan).toHaveBeenCalledWith("c1", "UBAH 2 BARIS tb_bahan", "c1");
    expect(await screen.findByText(/2 baris diubah pada tabel tb_bahan/)).toBeTruthy();
  });

  it("sends the person to sign in again when the server asks, and shows the plan again on return", async () => {
    const user = userEvent.setup();
    api.runDataChangePlan.mockRejectedValue(Object.assign(new Error("403"), {
      status: 403, body: { detail: { error: "step_up_required", message: "Sign in again" } } }));
    renderPage();
    await planIt(user);
    await user.type(screen.getByLabelText("Frasa konfirmasi"), "UBAH 2 BARIS tb_bahan");
    await user.click(screen.getByRole("button", { name: "Jalankan" }));
    await waitFor(() => expect(keycloak.signInAgain).toHaveBeenCalledTimes(1));
    expect(sessionStorage.getItem("hai.dataChange.pending")).toBe("c1");

    cleanup();                                   // the browser comes back from Keycloak
    api.getDataChangePlan.mockResolvedValue(PLAN);
    renderPage();
    expect(await screen.findByText(/Anda sudah login ulang/)).toBeTruthy();
    expect(api.getDataChangePlan).toHaveBeenCalledWith("c1");
    expect(api.runDataChangePlan).toHaveBeenCalledTimes(1);   // it did NOT run by itself
    expect(sessionStorage.getItem("hai.dataChange.pending")).toBeNull();
  });

  it("tells the user a change needs someone else's approval", async () => {
    const user = userEvent.setup();
    api.createDataChangePlan.mockResolvedValue({
      ...PLAN, requires_approval: true, plan: { ...PLAN.plan, operation: "delete", set: undefined } });
    renderPage();
    await planIt(user);
    expect(screen.getByText(/perlu disetujui/)).toBeTruthy();
  });

  it("shows a refusal instead of a plan", async () => {
    const user = userEvent.setup();
    api.createDataChangePlan.mockResolvedValue({
      status: "refused", reason: "This would change more than 10 rows, the limit you set.", plan_id: null });
    renderPage();
    await user.type(await screen.findByLabelText(/Perintah/), "x");
    await user.click(screen.getByRole("button", { name: /Buat rencana/ }));
    expect(await screen.findByText(/more than 10 rows/)).toBeTruthy();
    expect(screen.queryByLabelText("Rencana perubahan")).toBeNull();
  });

  it("lets an approver see other people's changes and decide, signing in again first if needed", async () => {
    const user = userEvent.setup();
    api.listDataChangeApprovals.mockResolvedValue({ approvals: [
      { ...PLAN, plan_id: "c9", owner: "alice", requires_approval: true,
        plan: { ...PLAN.plan, operation: "delete", set: undefined } }] });
    api.decideDataChange.mockRejectedValueOnce(Object.assign(new Error("403"), {
      status: 403, body: { detail: { error: "step_up_required", message: "again" } } }));
    renderPage();
    expect(await screen.findByText(/Diajukan oleh/)).toBeTruthy();
    await user.click(screen.getByRole("button", { name: "Setujui" }));
    await waitFor(() => expect(keycloak.signInAgain).toHaveBeenCalled());
    expect(api.decideDataChange).toHaveBeenCalledWith("c9", "approve");
    await user.click(screen.getByRole("button", { name: "Tolak" }));
    expect(api.decideDataChange).toHaveBeenLastCalledWith("c9", "reject");
  });

  it("does not show the approvals list to someone who cannot approve", async () => {
    renderPage();
    await screen.findByLabelText(/Perintah/);
    expect(screen.queryByLabelText("Menunggu persetujuan")).toBeNull();
  });
});
