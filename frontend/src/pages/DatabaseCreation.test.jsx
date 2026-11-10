import { afterEach, describe, it, expect, vi, beforeEach } from "vitest";
import { cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter, Route, Routes, useLocation } from "react-router-dom";
import DatabaseCreation from "./DatabaseCreation";
import * as api from "../api/client";
import * as keycloak from "../auth/keycloak";

vi.mock("../api/client", () => ({
  addDatabaseServer: vi.fn(),
  autoSetupDatabaseServer: vi.fn(),
  inspectDatabaseServer: vi.fn(),
  cancelDatabasePlan: vi.fn(),
  connectDatabase: vi.fn(),
  createDatabasePlan: vi.fn(),
  getDatabaseCreationConfig: vi.fn(),
  getDatabasePlan: vi.fn(),
  listDatabaseConnections: vi.fn(),
  listDatabaseProfiles: vi.fn(),
  removeDatabaseServer: vi.fn(),
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
      <Route path="/database-agent/connections" element={<Landing />} />
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
    api.listDatabaseProfiles.mockResolvedValue({ profiles: [] });
    api.connectDatabase.mockResolvedValue({});
    api.refreshDatabaseConnection.mockResolvedValue({});
  });

  it("explains how to enable it when no server can create databases", async () => {
    api.getDatabaseCreationConfig.mockResolvedValue({ profiles: [] });
    renderPage();
    expect(await screen.findByText(/Belum ada server yang boleh membuat database/)).toBeTruthy();
    expect(screen.queryByRole("button", { name: /Buat rencana/ })).toBeNull();
  });

  it("adds a server from the form and offers it in the Server list without a script or restart", async () => {
    const user = userEvent.setup();
    api.getDatabaseCreationConfig
      .mockResolvedValueOnce({ profiles: [{ name: "local_mysql", type: "mysql", prefix: "hai_" }] })
      .mockResolvedValue({ profiles: [
        { name: "local_mysql", type: "mysql", prefix: "hai_" },
        { name: "srv_two", type: "sqlserver", prefix: "hai_" }] });
    api.addDatabaseServer.mockResolvedValue({ name: "srv_two", type: "sqlserver", prefix: "hai_" });
    renderPage();
    await screen.findByLabelText(/Perintah/);
    await user.click(screen.getByLabelText(/Akun sudah ada/));
    await user.selectOptions(screen.getByLabelText("Jenis database"), "sqlserver");
    await user.type(screen.getByLabelText("Nama profil"), "srv_two");
    await user.type(screen.getByLabelText("Alamat server"), "10.62.10.34");
    await user.type(screen.getByLabelText("Nama database yang dibaca"), "master_data");
    await user.type(screen.getByLabelText("Akun baca: user"), "hai_reader");
    await user.type(screen.getByLabelText("Akun baca: password"), "Reader-pass-1");
    await user.type(screen.getByLabelText("Akun pembuat database: user"), "hai_creator");
    await user.type(screen.getByLabelText("Akun pembuat database: password"), "Creator-pass-1");
    await user.click(screen.getByRole("button", { name: /Uji dan simpan server/ }));
    expect(api.addDatabaseServer).toHaveBeenCalledWith({
      name: "srv_two", type: "sqlserver", host: "10.62.10.34", database: "master_data",
      user: "hai_reader", password: "Reader-pass-1",
      creator: { user: "hai_creator", password: "Creator-pass-1" } });
    expect(await screen.findByText(/srv_two \(sqlserver\) tersimpan/)).toBeTruthy();
    expect(screen.getByRole("option", { name: "srv_two (sqlserver)" })).toBeTruthy();
    expect(screen.getByLabelText("Akun baca: password").value).toBe("");     // password is not kept
  });

  it("removes a server profile from the list after confirmation", async () => {
    const user = userEvent.setup();
    vi.spyOn(window, "confirm").mockReturnValue(true);
    api.listDatabaseProfiles
      .mockResolvedValueOnce({ profiles: [{ name: "local_mysql", type: "mysql" }, { name: "old_srv", type: "mysql" }] })
      .mockResolvedValue({ profiles: [{ name: "local_mysql", type: "mysql" }] });
    api.removeDatabaseServer.mockResolvedValue({ name: "old_srv", status: "removed" });
    renderPage();
    await user.click(await screen.findByRole("button", { name: "Hapus server old_srv" }));
    expect(api.removeDatabaseServer).toHaveBeenCalledWith("old_srv");
    await waitFor(() => expect(screen.queryByRole("button", { name: "Hapus server old_srv" })).toBeNull());
    expect(screen.getByRole("button", { name: "Hapus server local_mysql" })).toBeTruthy();
  });

  it("shows the reason when a server cannot be removed (still connected)", async () => {
    const user = userEvent.setup();
    vi.spyOn(window, "confirm").mockReturnValue(true);
    api.listDatabaseProfiles.mockResolvedValue({ profiles: [{ name: "old_srv", type: "mysql" }] });
    api.removeDatabaseServer.mockRejectedValue(Object.assign(new Error("422"), {
      body: { detail: { message: "a connection still uses this server; disconnect it first" } } }));
    renderPage();
    await user.click(await screen.findByRole("button", { name: "Hapus server old_srv" }));
    expect(await screen.findByText(/disconnect it first/)).toBeTruthy();
    expect(screen.getByRole("button", { name: "Hapus server old_srv" })).toBeTruthy();
  });

  it("sets up the three accounts automatically from one administrator login, then forgets that password", async () => {
    const user = userEvent.setup();
    api.inspectDatabaseServer.mockResolvedValue({
      name_taken: false, databases: ["dbsales"], accounts: ["hai_reader"],
      plan: { reader: { user: "hai_rd_srv", exists: true }, creator: { user: "hai_cr_srv", exists: false },
        writer: { user: "hai_wr_srv", exists: false } } });
    api.autoSetupDatabaseServer.mockResolvedValue({
      name: "srv", type: "mysql", prefix: "hai_", accounts: { reader: "hai_rd_srv", creator: "hai_cr_srv" } });
    api.getDatabaseCreationConfig
      .mockResolvedValueOnce({ profiles: [{ name: "local_mysql", type: "mysql", prefix: "hai_" }] })
      .mockResolvedValue({ profiles: [{ name: "local_mysql", type: "mysql", prefix: "hai_" },
        { name: "srv", type: "mysql", prefix: "hai_" }] });
    renderPage();
    await user.type(await screen.findByLabelText("Nama profil"), "srv");
    await user.type(screen.getByLabelText("Alamat server"), "127.0.0.1");
    await user.type(screen.getByLabelText("Administrator: password"), "Adm1nSecret");
    await user.click(screen.getByRole("button", { name: "Periksa server" }));
    expect(api.inspectDatabaseServer).toHaveBeenCalledWith({
      name: "srv", type: "mysql", host: "127.0.0.1", admin_user: "root", admin_password: "Adm1nSecret",
      roles: ["reader", "creator", "writer"] });
    expect(await screen.findByText(/sudah ada, passwordnya diatur ulang/)).toBeTruthy();
    expect(screen.getByLabelText("Administrator: password").value).toBe("");          // dropped after use
    await user.type(screen.getByLabelText("Administrator: password"), "Adm1nSecret");
    await user.click(screen.getByLabelText(/Buat akun ubah data/));                   // no writer account
    await user.click(screen.getByRole("button", { name: "Buat akun dan simpan server" }));
    expect(api.autoSetupDatabaseServer).toHaveBeenCalledWith(expect.objectContaining({
      roles: ["reader", "creator"], admin_password: "Adm1nSecret" }));
    expect(await screen.findByText(/Password acak tersimpan di .env/)).toBeTruthy();
    expect(screen.getByRole("option", { name: "srv (mysql)" })).toBeTruthy();
    expect(screen.getByLabelText("Administrator: password").value).toBe("");
  });

  it("asks to sign in again when the server wants a fresh login for account setup", async () => {
    const user = userEvent.setup();
    api.inspectDatabaseServer.mockRejectedValue(Object.assign(new Error("403"), {
      status: 403, body: { detail: { error: "step_up_required", message: "again" } } }));
    renderPage();
    await user.type(await screen.findByLabelText("Nama profil"), "srv");
    await user.type(screen.getByLabelText("Alamat server"), "127.0.0.1");
    await user.type(screen.getByLabelText("Administrator: password"), "x");
    await user.click(screen.getByRole("button", { name: "Periksa server" }));
    await waitFor(() => expect(keycloak.signInAgain).toHaveBeenCalled());
  });

  it("shows why a server could not be added and keeps what was typed", async () => {
    const user = userEvent.setup();
    api.addDatabaseServer.mockRejectedValue(Object.assign(new Error("422"), {
      status: 422, body: { detail: { error: "invalid_request", message: "the server is not reachable" } } }));
    renderPage();
    await user.click(await screen.findByLabelText(/Akun sudah ada/));
    await user.type(await screen.findByLabelText("Nama profil"), "srv_two");
    await user.type(screen.getByLabelText("Alamat server"), "10.1.1.1");
    await user.type(screen.getByLabelText("Nama database yang dibaca"), "d");
    await user.type(screen.getByLabelText("Akun baca: user"), "u");
    await user.type(screen.getByLabelText("Akun baca: password"), "p");
    await user.click(screen.getByRole("button", { name: /Uji dan simpan server/ }));
    expect(await screen.findByText(/the server is not reachable/)).toBeTruthy();
    expect(screen.getByLabelText("Alamat server").value).toBe("10.1.1.1");
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
    expect(screen.getByText(/lewat panel/)).toBeTruthy();
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
