import { useEffect, useState } from "react";
import { AlertCircle, LoaderCircle } from "lucide-react";
import { Link, useNavigate } from "react-router-dom";
import {
  cancelDatabasePlan,
  connectDatabase,
  createDatabasePlan,
  getDatabaseCreationConfig,
  getDatabasePlan,
  listDatabaseConnections,
  refreshDatabaseConnection,
  runDatabasePlan,
} from "../api/client.js";
import { signInAgain } from "../auth/keycloak.js";
import AddServerForm from "./AddServerForm.jsx";
import "./autonomous-testing.css";

const PENDING_KEY = "hai.databaseCreation.pending";
const remember = (id) => { try { sessionStorage.setItem(PENDING_KEY, id); } catch { /* ask again */ } };
const takePending = () => {
  try {
    const value = sessionStorage.getItem(PENDING_KEY);
    sessionStorage.removeItem(PENDING_KEY);
    return value;
  } catch { return null; }
};
const errorCode = (e) => e?.body?.detail?.error;
const errorText = (e) => e?.body?.detail?.message || e?.message || "Terjadi kesalahan.";

export default function DatabaseCreation() {
  const [profiles, setProfiles] = useState([]);
  const [profile, setProfile] = useState("");
  const [question, setQuestion] = useState("");
  const [plan, setPlan] = useState(null);
  const [phrase, setPhrase] = useState("");
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState(null);
  const [result, setResult] = useState(null);
  const [notice, setNotice] = useState("");
  const [connected, setConnected] = useState(false);
  const navigate = useNavigate();

  useEffect(() => {
    let active = true;
    (async () => {
      try {
        const cfg = await getDatabaseCreationConfig();
        if (!active) return;
        setProfiles(cfg.profiles || []);
        if (cfg.profiles?.length === 1) setProfile(cfg.profiles[0].name);
      } catch (error) {
        if (active) setMessage({ text: errorText(error) });
      }
      const pending = takePending();
      if (pending && active) {
        try {
          const restored = await getDatabasePlan(pending);
          if (active && restored.status === "planned") {
            setPlan(restored);
            setNotice("Anda sudah login ulang. Tekan Jalankan untuk melanjutkan.");
          }
        } catch { /* expired: ask again */ }
      }
    })();
    return () => { active = false; };
  }, []);

  // A server was just added from the form: show it in the list, and pick it if it can create databases.
  const reloadProfiles = async (added) => {
    const cfg = await getDatabaseCreationConfig();
    setProfiles(cfg.profiles || []);
    if (added && (cfg.profiles || []).some((p) => p.name === added.name)) setProfile(added.name);
  };

  // A server was removed from the list: drop it from the choices (and from the selection if it was chosen).
  const serverRemoved = async (name) => {
    const cfg = await getDatabaseCreationConfig();
    setProfiles(cfg.profiles || []);
    setProfile((current) => (current === name ? "" : current));
  };

  const guarded = async (work) => {
    setBusy(true);
    setMessage(null);
    try { await work(); } catch (error) { setMessage({ text: errorText(error) }); } finally { setBusy(false); }
  };

  const makePlan = (event) => {
    event.preventDefault();
    return guarded(async () => {
      setResult(null); setPlan(null); setPhrase(""); setNotice(""); setConnected(false);
      const body = await createDatabasePlan({ question, ...(profile ? { profile } : {}) });
      if (body.status === "planned") setPlan(body);
      else setMessage({ text: body.reason });
    });
  };

  // The database exists now: connect it (or, when tables were added to one that is already connected, read its
  // schema again) and open the approval form with everything filled in. The user only has to press Simpan.
  const openInReports = async (created) => {
    const source = `${created.profile}/${created.database}`;
    try {
      const existing = ((await listDatabaseConnections()).connections || []).find((c) => c.file === source);
      let name;
      if (existing) {
        await refreshDatabaseConnection(existing.name);
        name = existing.name;
      } else {
        name = created.database.slice(0, 32);
        await connectDatabase({ name, profile: created.profile, database: created.database });
      }
      setConnected(true);
      navigate("/database-agent/reports", { state: {
        review: name,
        message: `${created.mode === "add_tables" ? "Tabel baru ditambahkan ke" : "Database baru dibuat:"} ${created.database}. `
          + "Sudah terhubung, dan pilihan di bawah sudah diisi (boleh difilter, tambah, ubah). Periksa lalu tekan Setujui.",
      } });
    } catch (error) {
      setMessage({ text: `Database sudah dibuat, tetapi belum bisa dihubungkan otomatis: ${errorText(error)} `
        + "Tekan Hubungkan database ini, atau hubungkan lewat Data Reports." });
    }
  };

  const run = () =>
    guarded(async () => {
      try {
        const created = await runDatabasePlan(plan.plan_id, phrase, plan.plan_id);
        setResult(created);
        setPlan(null);
        setNotice("");
        await openInReports(created);
      } catch (error) {
        if (errorCode(error) !== "step_up_required") throw error;
        remember(plan.plan_id);
        setMessage({ text: "Untuk keamanan, Anda akan diminta login ulang ke Keycloak..." });
        signInAgain();
      }
    });

  const cancel = () =>
    guarded(async () => {
      await cancelDatabasePlan(plan.plan_id);
      setPlan(null);
    });

  const connect = () =>
    guarded(async () => {
      await connectDatabase({ name: result.database.slice(0, 32), profile: result.profile, database: result.database });
      setConnected(true);
    });

  const phraseOk = plan && phrase.trim() === plan.confirm_phrase;

  return (
    <main className="at-page">
      <header className="at-header">
        <h1>Buat database</h1>
        <p>
          Minta agent membuat database baru beserta tabelnya, atau menambah tabel baru ke database yang sudah ada. Agent tidak menulis SQL: ia memilih rancangan
          terstruktur yang Anda periksa dulu. Hanya nama berawalan khusus yang bisa dibuat; database yang sudah
          ada tidak pernah ditimpa atau dihapus.
        </p>
      </header>

      {message && (
        <div className="at-warning" role="alert"><AlertCircle size={16} /><div>{message.text}</div></div>
      )}

      <AddServerForm onAdded={reloadProfiles} onRemoved={serverRemoved} />

      <section className="at-panel">
        {profiles.length === 0 ? (
          <p>
            Belum ada server yang boleh membuat database. Tambahkan server di atas (isi akun pembuat database),
            atau jalankan <code>python scripts\setup_mysql_creator_user.py &lt;profil&gt;</code> lalu restart
            backend (lihat docs/TRY_DATA_REPORTS_LOCALLY.md).
          </p>
        ) : (
          <form className="at-form" onSubmit={makePlan} aria-label="Rancang database">
            {profiles.length > 1 && (
              <label>
                Server
                <select value={profile} onChange={(e) => setProfile(e.target.value)} required>
                  <option value="">Pilih...</option>
                  {profiles.map((p) => <option key={p.name} value={p.name}>{p.type ? `${p.name} (${p.type})` : p.name}</option>)}
                </select>
              </label>
            )}
            {profiles.length === 1 && (
              <small>Server: <b>{profiles[0].name}{profiles[0].type ? ` (${profiles[0].type})` : ""}</b>. Untuk SQL Server,
                tambahkan servernya lewat panel <b>Tambah server database</b> di atas; setelah itu muncul pilihan server di sini.</small>
            )}
            <label>
              Perintah
              <textarea
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                rows={3}
                required
                placeholder="Contoh: buat database hai_toko dengan tabel bahan (nama, harga beli, tanggal masuk). Atau: tambah tabel pegawai di database hai_toko"
              />
            </label>
            <small>
              Nama database harus diawali <b>{profiles.find((p) => p.name === profile)?.prefix || profiles[0].prefix}</b>.
            </small>
            <button className="at-button at-button-primary" disabled={busy || !question.trim()}>
              {busy ? <LoaderCircle size={14} className="spin" /> : null} Buat rencana
            </button>
          </form>
        )}
      </section>

      {plan && (
        <section className="at-panel" aria-label="Rencana database">
          <h2>Periksa dulu. Belum ada yang dibuat.</h2>
          <div className="at-config-summary">
            {plan.mode === "add_tables" ? (
              <span>Menambah tabel ke database yang <b>sudah ada</b>: <b>{plan.plan.database}</b> (server {plan.profile}). Tabel dan data yang ada tidak disentuh.</span>
            ) : (
              <span>Database baru: <b>{plan.plan.database}</b> (server {plan.profile})</span>
            )}
            {plan.plan.tables.length === 0 && <span>Tanpa tabel.</span>}
            {plan.plan.tables.map((t) => (
              <span key={t.name}>
                Tabel <b>{t.name}</b>: {t.columns.map((c) => `${c.name} (${c.type}${c.nullable ? "" : ", wajib"})`).join(", ")}
              </span>
            ))}
          </div>
          {notice && <p role="status">{notice}</p>}
          <label>
            Ketik persis: <code>{plan.confirm_phrase}</code>
            <input value={phrase} onChange={(e) => setPhrase(e.target.value)} autoComplete="off" aria-label="Frasa konfirmasi" />
          </label>
          <small>Saat Jalankan ditekan, Anda mungkin diminta login ulang ke Keycloak.</small>
          <div className="at-form-row">
            <button type="button" className="at-button at-button-primary" disabled={busy || !phraseOk} onClick={run}>
              Jalankan
            </button>
            <button type="button" className="at-button at-button-secondary" disabled={busy} onClick={cancel}>
              Batal
            </button>
          </div>
        </section>
      )}

      {result && (
        <section className="at-panel" aria-live="polite">
          <b>Berhasil.</b>{" "}
          {result.mode === "add_tables"
            ? `Tabel ${result.tables.join(", ")} ditambahkan ke database ${result.database}.`
            : `Database ${result.database} dibuat${result.tables.length ? ` dengan tabel ${result.tables.join(", ")}` : ""}.`}
          <div className="at-form-row">
            {connected ? (
              <span>Terhubung. Buka tab <Link to="/database-agent/reports">Laporan</Link> untuk menyetujui tabelnya
                (centang <b>boleh diubah</b> bila ingin mengisinya).</span>
            ) : (
              <button type="button" className="at-button at-button-primary" disabled={busy} onClick={connect}>
                Hubungkan database ini
              </button>
            )}
          </div>
        </section>
      )}
    </main>
  );
}
