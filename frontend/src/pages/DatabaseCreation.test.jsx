import { afterEach, describe, it, expect, vi, beforeEach } from "vitest";
import { cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter, Route, Routes, useLocation } from "react-router-dom";
import DatabaseCreation from "./DatabaseCreation";
import * as api from "../api/client";
import * as keycloak from "../auth/keycloak";

vi.mock("../api/client", () => ({
  cancelDatabasePlan: vi.fn(),
  connectDatabase: vi.fn(),
  createDatabasePlan: vi.fn(),
  getDatabaseCreationConfig: vi.fn(),
  getDatabasePlan: vi.fn(),
  listDatabaseConnections: vi.fn(),
  refreshDatabaseConnection: vi.fn(),
  runDatabasePlan: vi.fn(),
}));
vi.mock("../auth/keycloak", () => ({ signInAgain: vi.fn() }));

const PLAN = {
  status: "planned", plan_id: "d1", profile: "local_mysql", confirm_phrase: "BUAT DATABASE hai_toko",
  statements: ["CREATE DATABASE hai_toko"],
  plan: { database: "hai_toko", tables: [{ name: "tb_bahan", columns: [
    { name: "id", type: "integer (otomatis)", nullable: false },
    { name: "nama", type: "text", nullable: false }] }] },
};

function Landing() {
  const state = useLocation().state;
  return <div>DATA REPORTS review={state?.review} :: {state?.message}</div>;
}

const renderPage = () => render(
  <MemoryRouter>
    <Routes>
      <Route path="/" element={<DatabaseCreation />} />
      <Route path="/data-reports" element={<Landing />} />
    </Routes>
  </MemoryRouter>);

async function planIt(user) {
  await user.type(await screen.findByLabelText(/Perintah/), "buat toko");
  await user.click(screen.getByRole("button", { name: /Buat rencana/ }));
  return screen.findByLabelText("Rencana database");
}

afterEach(cleanup);

describe("DatabaseCreation", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    sessionStorage.clear();
    api.getDatabaseCreationConfig.mockResolvedValue({ profiles: [{ name: "local_mysql", prefix: "hai_" }] });
    api.createDatabasePlan.mockResolvedValue(PLAN);
    api.listDatabaseConnections.mockResolvedValue({ connections: [] });
    api.connectDatabase.mockResolvedValue({});
    api.refreshDatabaseConnection.mockResolvedValue({});
  });

  it("explains how to enable it when no server can create databases", async () => {
    api.getDatabaseCreationConfig.mockResolvedValue({ profiles: [] });
    renderPage();
    expect(await screen.findByText(/Belum ada server yang boleh membuat database/)).toBeTruthy();
    expect(screen.queryByRole("button", { name: /Buat rencana/ })).toBeNull();
  });

  it("shows the design and the required name prefix, and nothing is created yet", async () => {
    const user = userEvent.setup();
    renderPage();
    expect((await screen.findByText(/diawali/)).textContent).toMatch(/hai_/);
    const card = (await planIt(user)).textContent;
    expect(card).toMatch(/Database baru: hai_toko/);
    expect(card).toMatch(/Tabel tb_bahan: id \(integer \(otomatis\), wajib\), nama \(text, wajib\)/);
    expect(api.runDatabasePlan).not.toHaveBeenCalled();
  });

  it("runs only with the exact phrase, then connects the new database and opens its approval at once", async () => {
    const user = userEvent.setup();
    api.runDatabasePlan.mockResolvedValue({ status: "created", database: "hai_toko", tables: ["tb_bahan"], profile: "local_mysql" });
    renderPage();
    await planIt(user);
    const run = screen.getByRole("button", { name: "Jalankan" });
    expect(run.disabled).toBe(true);
    await user.type(screen.getByLabelText("Frasa konfirmasi"), "BUAT DATABASE hai_toko");
    await user.click(run);
    expect(api.runDatabasePlan).toHaveBeenCalledWith("d1", "BUAT DATABASE hai_toko", "d1");
    expect(await screen.findByText(/DATA REPORTS review=hai_toko/)).toBeTruthy();
    expect(api.connectDatabase).toHaveBeenCalledWith({ name: "hai_toko", profile: "local_mysql", database: "hai_toko" });
    expect(api.refreshDatabaseConnection).not.toHaveBeenCalled();
    expect(screen.getByText(/Database baru dibuat: hai_toko/)).toBeTruthy();
  });

  it("reads the schema again, instead of connecting twice, when tables were added to a database that is already connected", async () => {
    const user = userEvent.setup();
    api.createDatabasePlan.mockResolvedValue({ ...PLAN, mode: "add_tables", confirm_phrase: "TAMBAH TABEL hai_toko" });
    api.runDatabasePlan.mockResolvedValue({ status: "created", mode: "add_tables", database: "hai_toko", tables: ["tb_baru"], profile: "local_mysql" });
    api.listDatabaseConnections.mockResolvedValue({ connections: [{ name: "toko", file: "local_mysql/hai_toko", status: "active" }] });
    renderPage();
    await planIt(user);
    await user.type(screen.getByLabelText("Frasa konfirmasi"), "TAMBAH TABEL hai_toko");
    await user.click(screen.getByRole("button", { name: "Jalankan" }));
    expect(await screen.findByText(/DATA REPORTS review=toko/)).toBeTruthy();
    expect(api.refreshDatabaseConnection).toHaveBeenCalledWith("toko");
    expect(api.connectDatabase).not.toHaveBeenCalled();
    expect(screen.getByText(/Tabel baru ditambahkan ke hai_toko/)).toBeTruthy();
  });

  it("stays on the page with a clear message and a manual button when the automatic connection fails", async () => {
    const user = userEvent.setup();
    api.runDatabasePlan.mockResolvedValue({ status: "created", database: "hai_toko", tables: ["tb_bahan"], profile: "local_mysql" });
    api.connectDatabase.mockRejectedValueOnce(Object.assign(new Error("403"), { status: 403,
      body: { detail: { message: "The principal lacks database:connection:manage." } } }));
    renderPage();
    await planIt(user);
    await user.type(screen.getByLabelText("Frasa konfirmasi"), "BUAT DATABASE hai_toko");
    await user.click(screen.getByRole("button", { name: "Jalankan" }));
    expect(await screen.findByText(/Database sudah dibuat, tetapi belum bisa dihubungkan otomatis/)).toBeTruthy();
    expect(screen.queryByText(/DATA REPORTS review/)).toBeNull();
    await user.click(await screen.findByRole("button", { name: /Hubungkan database ini/ }));
    expect(api.connectDatabase).toHaveBeenCalledTimes(2);
  });

  it("signs in again when asked, and shows the plan again without running it", async () => {
    const user = userEvent.setup();
    api.runDatabasePlan.mockRejectedValue(Object.assign(new Error("403"), {
      status: 403, body: { detail: { error: "step_up_required", message: "again" } } }));
    renderPage();
    await planIt(user);
    await user.type(screen.getByLabelText("Frasa konfirmasi"), "BUAT DATABASE hai_toko");
    await user.click(screen.getByRole("button", { name: "Jalankan" }));
    await waitFor(() => expect(keycloak.signInAgain).toHaveBeenCalledTimes(1));
    expect(sessionStorage.getItem("hai.databaseCreation.pending")).toBe("d1");

    cleanup();
    api.getDatabasePlan.mockResolvedValue(PLAN);
    renderPage();
    expect(await screen.findByText(/Anda sudah login ulang/)).toBeTruthy();
    expect(api.runDatabasePlan).toHaveBeenCalledTimes(1);
  });

  it("says plainly when tables are added to a database that already exists", async () => {
    const user = userEvent.setup();
    api.createDatabasePlan.mockResolvedValue({
      ...PLAN, mode: "add_tables", confirm_phrase: "TAMBAH TABEL hai_toko" });
    api.runDatabasePlan.mockResolvedValue({
      status: "created", mode: "add_tables", database: "hai_toko", tables: ["tb_bahan"], profile: "local_mysql" });
    renderPage();
    const card = (await planIt(user)).textContent;
    expect(card).toMatch(/Menambah tabel ke database yang sudah ada: hai_toko/);
    expect(card).toMatch(/tidak disentuh/);
    await user.type(screen.getByLabelText("Frasa konfirmasi"), "TAMBAH TABEL hai_toko");
    await user.click(screen.getByRole("button", { name: "Jalankan" }));
    expect(await screen.findByText(/Tabel baru ditambahkan ke hai_toko/)).toBeTruthy();
  });

  it("says which server is used when only one is configured, and where to set up another", async () => {
    api.getDatabaseCreationConfig.mockResolvedValue({ profiles: [{ name: "local_mysql", prefix: "hai_", type: "mysql" }] });
    renderPage();
    expect(await screen.findByText(/local_mysql \(mysql\)/)).toBeTruthy();
    expect(screen.getByText(/siapkan dulu profilnya/)).toBeTruthy();
    expect(screen.queryByLabelText("Server")).toBeNull();
  });

  it("lets the user pick between a MySQL and a SQL Server server, and sends the chosen profile", async () => {
    const user = userEvent.setup();
    api.getDatabaseCreationConfig.mockResolvedValue({ profiles: [
      { name: "local_mysql", prefix: "hai_", type: "mysql" },
      { name: "local_sqlserver", prefix: "hai_", type: "sqlserver" }] });
    renderPage();
    await user.selectOptions(await screen.findByLabelText("Server"), "local_sqlserver");
    expect(screen.getByRole("option", { name: "local_sqlserver (sqlserver)" })).toBeTruthy();
    await user.type(screen.getByLabelText(/Perintah/), "buat toko");
    await user.click(screen.getByRole("button", { name: /Buat rencana/ }));
    await screen.findByLabelText("Rencana database");
    expect(api.createDatabasePlan).toHaveBeenCalledWith({ question: "buat toko", profile: "local_sqlserver" });
  });

  it("shows a refusal (name already taken) instead of a plan", async () => {
    const user = userEvent.setup();
    api.createDatabasePlan.mockResolvedValue({ status: "refused", reason: "A database named 'hai_toko' already exists", plan_id: null });
    renderPage();
    await user.type(await screen.findByLabelText(/Perintah/), "buat toko");
    await user.click(screen.getByRole("button", { name: /Buat rencana/ }));
    expect(await screen.findByText(/already exists/)).toBeTruthy();
    expect(screen.queryByLabelText("Rencana database")).toBeNull();
  });
});
