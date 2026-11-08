import { useState } from "react";
import { LoaderCircle } from "lucide-react";
import { addDatabaseServer, autoSetupDatabaseServer, inspectDatabaseServer } from "../api/client.js";
import { signInAgain } from "../auth/keycloak.js";

const EMPTY = {
  name: "", type: "mysql", host: "", port: "", database: "", user: "", password: "",
  creatorUser: "", creatorPassword: "", writerUser: "", writerPassword: "",
};
const HOST_HINT = {
  mysql: "mis. 127.0.0.1 atau 10.62.10.34",
  sqlserver: "mis. (local)\\SQLEXPRESS atau 10.62.10.34",
  postgresql: "mis. 127.0.0.1 atau 10.62.10.34",
};
const errorText = (e) => e?.body?.detail?.message || e?.message || "Terjadi kesalahan.";

/** Registers a database server from the browser. Only limited accounts belong here, never sa/root. */
function ManualForm({ onAdded }) {
  const [f, setF] = useState(EMPTY);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [done, setDone] = useState("");
  const set = (key) => (e) => setF({ ...f, [key]: e.target.value });
  const readOnly = f.type === "postgresql";
  const creatorGiven = !readOnly && Boolean(f.creatorUser.trim());

  const submit = async (event) => {
    event.preventDefault();
    setBusy(true); setError(""); setDone("");
    const body = {
      name: f.name.trim(), type: f.type, host: f.host.trim(),
      ...(f.database.trim() ? { database: f.database.trim() } : {}),
      user: f.user.trim(), password: f.password,
      ...(f.port.trim() ? { port: Number(f.port) } : {}),
      ...(!readOnly && f.creatorUser.trim()
        ? { creator: { user: f.creatorUser.trim(), password: f.creatorPassword } } : {}),
      ...(!readOnly && f.writerUser.trim()
        ? { writer: { user: f.writerUser.trim(), password: f.writerPassword } } : {}),
    };
    try {
      const added = await addDatabaseServer(body);
      setDone(`Server ${added.name} (${added.type}) tersimpan dan sudah bisa dipilih.`);
      setF(EMPTY);
      await onAdded?.(added);
    } catch (e) {
      setError(errorText(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div>
      <p>
        Isi akun terbatas yang sudah disiapkan administrator database (akun baca, dan bila perlu akun pembuat
        database dan akun ubah data). <b>Jangan</b> isi akun administrator seperti sa atau root. Koneksi diuji dulu;
        server baru disimpan hanya bila berhasil. Password disimpan di berkas .env komputer server ini,
        tidak pernah di daftar profil dan tidak pernah ditampilkan lagi.
      </p>
      <form className="at-form" onSubmit={submit} aria-label="Form server baru">
        <label>Nama profil
          <input value={f.name} onChange={set("name")} required placeholder="mis. server_hris" autoComplete="off" />
        </label>
        <label>Jenis database
          <select value={f.type} onChange={set("type")}>
            <option value="mysql">MySQL / MariaDB</option>
            <option value="sqlserver">SQL Server</option>
            <option value="postgresql">PostgreSQL (hanya baca)</option>
          </select>
        </label>
        <label>Alamat server
          <input value={f.host} onChange={set("host")} required placeholder={HOST_HINT[f.type]} autoComplete="off" />
        </label>
        <label>Port (kosongkan untuk bawaan)
          <input value={f.port} onChange={set("port")} inputMode="numeric" autoComplete="off" />
        </label>
        <label>Nama database yang dibaca
          <input value={f.database} onChange={set("database")} required={!creatorGiven} autoComplete="off" />
        </label>
        {creatorGiven && (
          <small>Boleh dikosongkan: server ini dipakai untuk membuat database baru, jadi belum perlu database yang sudah ada.</small>
        )}
        <label>Akun baca: user
          <input value={f.user} onChange={set("user")} required autoComplete="off" />
        </label>
        <label>Akun baca: password
          <input type="password" value={f.password} onChange={set("password")} required autoComplete="new-password" />
        </label>
        {!readOnly && (
          <>
            <small>Opsional, agar agent bisa membuat database (nama berawalan hai_):</small>
            <label>Akun pembuat database: user
              <input value={f.creatorUser} onChange={set("creatorUser")} autoComplete="off" />
            </label>
            <label>Akun pembuat database: password
              <input type="password" value={f.creatorPassword} onChange={set("creatorPassword")}
                required={Boolean(f.creatorUser.trim())} autoComplete="new-password" />
            </label>
            <small>Opsional, agar data bisa ditambah, diubah, dan dihapus:</small>
            <label>Akun ubah data: user
              <input value={f.writerUser} onChange={set("writerUser")} autoComplete="off" />
            </label>
            <label>Akun ubah data: password
              <input type="password" value={f.writerPassword} onChange={set("writerPassword")}
                required={Boolean(f.writerUser.trim())} autoComplete="new-password" />
            </label>
          </>
        )}
        {error && <div className="at-warning" role="alert">{error}</div>}
        {done && <p role="status">{done}</p>}
        <button className="at-button at-button-primary" disabled={busy}>
          {busy ? <LoaderCircle size={14} className="spin" /> : null} Uji dan simpan server
        </button>
      </form>
    </div>
  );
}

/** Wizard: look at the server with a one-time administrator login, then create the three accounts. */
function AutoForm({ onAdded }) {
  const [f, setF] = useState({ name: "", type: "mysql", host: "", port: "", adminUser: "root", adminPassword: "",
    database: "", creator: true, writer: true });
  const [found, setFound] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [done, setDone] = useState("");
  const set = (key) => (e) => setF({ ...f, [key]: e.target.value });
  const tick = (key) => (e) => setF({ ...f, [key]: e.target.checked });

  const body = () => ({
    name: f.name.trim(), type: f.type, host: f.host.trim(),
    ...(f.port.trim() ? { port: Number(f.port) } : {}),
    admin_user: f.adminUser.trim(), admin_password: f.adminPassword,
    ...(f.database.trim() ? { database: f.database.trim() } : {}),
    roles: ["reader", ...(f.creator ? ["creator"] : []), ...(f.writer ? ["writer"] : [])],
  });

  const work = (call, after) => async (event) => {
    event.preventDefault();
    setBusy(true); setError(""); setDone("");
    try {
      await after(await call(body()));
    } catch (e) {
      if (e?.body?.detail?.error === "step_up_required") {
        setError("Untuk keamanan, login ulang dulu (Keycloak), lalu ulangi langkah ini.");
        signInAgain();
      } else {
        setError(errorText(e));
      }
    } finally {
      setF((current) => ({ ...current, adminPassword: "" }));   // the administrator password is never kept
      setBusy(false);
    }
  };

  const inspect = work(inspectDatabaseServer, async (result) => {
    if (result.name_taken) throw new Error("Nama profil itu sudah dipakai. Pilih nama lain.");
    setFound(result);
  });
  const create = work(autoSetupDatabaseServer, async (added) => {
    setDone(`Server ${added.name} (${added.type}) siap. Akun dibuat: ${Object.values(added.accounts).join(", ")}. `
      + "Password acak tersimpan di .env dan tidak ditampilkan.");
    setFound(null);
    await onAdded?.(added);
  });

  const labelOf = { reader: "Akun baca", creator: "Akun pembuat database", writer: "Akun ubah data" };

  return (
    <div>
      <p>
        Cukup isi server dan <b>akun administrator</b> (root / sa). Password administrator dipakai <b>sekali</b>
        untuk membuat akun terbatas secara otomatis, lalu dibuang: tidak disimpan di berkas apa pun dan tidak
        dicatat. Yang tersimpan hanya password acak akun baru, di .env. Akun diberi nama sesuai profil.
      </p>
      <form className="at-form" onSubmit={found ? create : inspect} aria-label="Form server otomatis">
        <label>Nama profil
          <input value={f.name} onChange={set("name")} required placeholder="mis. server_hris" autoComplete="off" />
        </label>
        <label>Jenis database
          <select value={f.type} onChange={(e) => {
            setF({ ...f, type: e.target.value, adminUser: e.target.value === "sqlserver" ? "sa" : "root" });
            setFound(null);
          }}>
            <option value="mysql">MySQL / MariaDB</option>
            <option value="sqlserver">SQL Server</option>
          </select>
        </label>
        <label>Alamat server
          <input value={f.host} onChange={set("host")} required placeholder={HOST_HINT[f.type]} autoComplete="off" />
        </label>
        <label>Port (kosongkan untuk bawaan)
          <input value={f.port} onChange={set("port")} inputMode="numeric" autoComplete="off" />
        </label>
        <label>Administrator: user
          <input value={f.adminUser} onChange={set("adminUser")} required autoComplete="off" />
        </label>
        <label>Administrator: password
          <input type="password" value={f.adminPassword} onChange={set("adminPassword")} required
            autoComplete="new-password" />
        </label>
        {found && (
          <>
            <label>Database yang dibaca (opsional)
              <input list="found-databases" value={f.database} onChange={set("database")} autoComplete="off" />
              <datalist id="found-databases">
                {found.databases.map((d) => <option key={d} value={d} />)}
              </datalist>
            </label>
            <label><input type="checkbox" checked={f.creator} onChange={tick("creator")} /> Buat akun pembuat database (hai_)</label>
            <label><input type="checkbox" checked={f.writer} onChange={tick("writer")} /> Buat akun ubah data</label>
            <div className="at-config-summary" role="status">
              {Object.entries(found.plan).filter(([role]) => role === "reader" || f[role]).map(([role, p]) => (
                <span key={role}>
                  {labelOf[role]}: <b>{p.user}</b> — {p.exists ? "sudah ada, passwordnya diatur ulang" : "belum ada, akan dibuat"}
                </span>
              ))}
              {found.accounts.length > 0 && <span>Akun hai_ di server ini: {found.accounts.join(", ")}</span>}
            </div>
          </>
        )}
        {error && <div className="at-warning" role="alert">{error}</div>}
        {done && <p role="status">{done}</p>}
        <button className="at-button at-button-primary" disabled={busy}>
          {busy ? <LoaderCircle size={14} className="spin" /> : null} {found ? "Buat akun dan simpan server" : "Periksa server"}
        </button>
      </form>
    </div>
  );
}

export default function AddServerForm({ onAdded }) {
  const [mode, setMode] = useState("auto");
  return (
    <details className="at-panel" aria-label="Tambah server">
      <summary><b>Tambah server database</b></summary>
      <div className="at-form-row" role="radiogroup" aria-label="Cara menambah server">
        <label><input type="radio" name="server-mode" checked={mode === "auto"} onChange={() => setMode("auto")} />
          {" "}Otomatis (pakai akun administrator sekali)</label>
        <label><input type="radio" name="server-mode" checked={mode === "manual"} onChange={() => setMode("manual")} />
          {" "}Akun sudah ada (manual)</label>
      </div>
      {mode === "auto" ? <AutoForm onAdded={onAdded} /> : <ManualForm onAdded={onAdded} />}
    </details>
  );
}
