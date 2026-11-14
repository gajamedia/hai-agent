import { useEffect, useRef, useState } from "react";
import { AlertCircle, LoaderCircle, Paperclip, Send, X } from "lucide-react";
import { Link } from "react-router-dom";
import {
  cancelDataChangePlan,
  cancelDataEntry,
  cancelDatabasePlan,
  cancelReportPlan,
  createDataChangePlan,
  createDataChangePlanFromFile,
  createDataEntry,
  createDataEntryFromFile,
  createDatabasePlan,
  createDatabasePlanFromFile,
  createDatabasePlanFromImage,
  createReportPlan,
  decideDataChange,
  getDataChangeConfig,
  getDataChangePlan,
  getDataEntryDatabases,
  getDatabaseCreationConfig,
  getDatabasePlan,
  listDataChangeApprovals,
  listDatabaseConnections,
  routeAssistantFile,
  routeAssistantMessage,
  runDataChangePlan,
  runDataEntry,
  runDatabasePlan,
  runReportPlan,
} from "../api/client.js";
import { isImageFile } from "../api/fileKinds.js";
import { signInAgain } from "../auth/keycloak.js";
import { ApprovalCard, ChangeCard, DatabaseCard, EntryCard, ReportCard, ReportResult } from "./AssistantCards.jsx";
import FileMapping from "./FileMapping.jsx";
import "./autonomous-testing.css";

const PENDING_KEY = "hai.assistant.pending";
const errorCode = (e) => e?.body?.detail?.error;
const errorText = (e) => e?.body?.detail?.message || e?.message || "Terjadi kesalahan.";
const CONNECTIONS = { to: "/database-agent/connections", label: "Koneksi" };
// What a report can come with besides the table on screen (asked only when the person is looking at data).
const REPORT_FORMATS = { "Layar saja (JSON)": "", "Excel": "xlsx", "PDF": "pdf", "CSV": "csv", "HTML": "html" };
const OPERATION_DONE = { insert: "ditambahkan", update: "diubah", delete: "dihapus" };

// The conversation survives signing in again (the browser leaves for Keycloak and comes back, which reloads the
// page). Only the words are kept, for as long as this tab is open: cards hold rows of business data and buttons
// hold functions, so those are not stored (a waiting plan is fetched again by its id instead).
const HISTORY_KEY = "hai.assistant.history";
const KEEP_ITEMS = 60;
const loadHistory = () => {
  try {
    const saved = JSON.parse(sessionStorage.getItem(HISTORY_KEY) || "null");
    if (Array.isArray(saved) && saved.length) return saved;
  } catch { /* start fresh */ }
  return null;
};
const saveHistory = (items) => {
  try {
    const words = items.filter((i) => i.text && !i.choices).map(({ id, from, text, link, error }) => ({ id, from, text, link, error }));
    sessionStorage.setItem(HISTORY_KEY, JSON.stringify(words.slice(-KEEP_ITEMS)));
  } catch { /* private window: the conversation just starts again after a reload */ }
};

let nextId = 1;
const bot = (text, extra = {}) => ({ id: nextId++, from: "bot", text, ...extra });
const me = (text) => ({ id: nextId++, from: "me", text });

const GREETING = bot(
  "Tulis apa yang ingin dilakukan, dan lampirkan file bila perlu (gambar ERD hanya untuk membuat database). Saya memilihkan alurnya: membuat database, "
  + "mencatat, mengubah, atau melihat data. Semua yang menulis tetap menampilkan rencana dulu dan meminta "
  + "konfirmasi Anda.",
);

export default function Assistant() {
  const [items, setItems] = useState(() => {
    const saved = loadHistory();
    if (saved) nextId = Math.max(nextId, ...saved.map((i) => i.id || 0)) + 1;
    return saved || [GREETING];
  });
  const [text, setText] = useState("");
  const [file, setFile] = useState(null);
  const [busy, setBusy] = useState(false);
  // A question the assistant asked that the next typed message answers (the name of the database to make).
  const [asking, setAsking] = useState(null);
  const fileInput = useRef(null);
  const end = useRef(null);

  const push = (...more) => setItems((list) => [...list, ...more]);
  const replace = (id, patch) => setItems((list) => list.map((i) => (i.id === id ? { ...i, ...patch } : i)));
  const fail = (error) => push(bot(errorText(error), { error: true }));
  const guarded = async (work) => {
    setBusy(true);
    try { await work(); } catch (error) { fail(error); } finally { setBusy(false); }
  };

  useEffect(() => { end.current?.scrollIntoView?.({ block: "nearest" }); }, [items]);
  useEffect(() => { saveHistory(items); }, [items]);

  const newConversation = () => {
    setAsking(null);
    setItems([bot(GREETING.text)]);
  };

  // Back from signing in again: the plan that was waiting is shown again, ready for Jalankan.
  useEffect(() => {
    let saved = null;
    try { saved = sessionStorage.getItem(PENDING_KEY); sessionStorage.removeItem(PENDING_KEY); } catch { /* ask again */ }
    if (!saved) return;
    let pending;
    try { pending = JSON.parse(saved); } catch { pending = { kind: "database", id: saved }; }
    const load = pending.kind === "change" ? getDataChangePlan : getDatabasePlan;
    load(pending.id).then((view) => {
      if (view.status === "planned") {
        push(bot("Anda sudah login ulang. Tekan Jalankan untuk melanjutkan.", { card: { kind: pending.kind, view } }));
      }
    }).catch(() => { /* expired: ask again */ });
  }, []);

  // Changes that wait for this person's approval (only approvers get a list): shown when the page opens, so a
  // second person can approve here. Coming back from signing in again shows them once more.
  const loadApprovals = async () => {
    try {
      const approvals = (await listDataChangeApprovals())?.approvals || [];
      if (approvals.length) {
        push(bot(`Ada ${approvals.length} perubahan data yang menunggu persetujuan Anda.`),
          ...approvals.map((view) => bot(null, { card: { kind: "approval", view } })));
      }
    } catch { /* not an approver, or nothing to show */ }
  };
  useEffect(() => { loadApprovals(); }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const clearFile = () => {
    setFile(null);
    if (fileInput.current) fileInput.current.value = "";
  };

  const remember = (kind, id) => { try { sessionStorage.setItem(PENDING_KEY, JSON.stringify({ kind, id })); } catch { /* ask again */ } };

  // Several possible answers (which database, which table): buttons in the chat, then carry on with the pick.
  const offer = (prompt, names, resume) =>
    push(bot(prompt, { choices: names.map((n) => ({ label: n, pick: () => { push(me(n)); return guarded(() => resume(n)); } })) }));

  // -- Buat database ---------------------------------------------------------------------------------------------
  const planDatabase = async (ctx) => {
    const { question, database, table, attached } = ctx;
    const profiles = (await getDatabaseCreationConfig()).profiles || [];
    if (profiles.length === 0) {
      push(bot("Belum ada server yang boleh membuat database. Siapkan dulu di tab Buat database.", { link: { to: "/database-agent/create", label: "Buat database" } }));
      return;
    }
    const names = profiles.map((p) => p.name);
    const profile = ctx.profile || (names.length === 1 ? names[0] : null);
    if (!profile) { offer("Dibuat di server mana?", names, (v) => planDatabase({ ...ctx, profile: v })); return; }
    const body = attached && isImageFile(attached)
      ? await createDatabasePlanFromImage(attached, { database, table, profile, question })
      : attached
        ? await createDatabasePlanFromFile(attached, { database, table, profile, question })
        : await createDatabasePlan({ question, profile });
    if (body.status === "planned") push(bot("Ini rancangannya.", { card: { kind: "database", view: body } }));
    else push(bot(body.reason || "Permintaan itu tidak bisa dirancang."));
  };

  // -- Catat data ------------------------------------------------------------------------------------------------
  const planEntry = async (ctx) => {
    const { message, verdict, attached, mapping } = ctx;
    const databases = (await getDataEntryDatabases()).databases || [];
    if (databases.length === 0) {
      push(bot("Belum ada tabel yang disetujui untuk pencatatan. Minta administrator mencentang izin Tambah di tab Koneksi.", { link: CONNECTIONS }));
      return;
    }
    const names = databases.map((d) => d.name);
    const database = ctx.database || (names.includes(verdict.database) ? verdict.database : names.length === 1 ? names[0] : null);
    if (!database) { offer("Dicatat ke database mana?", names, (v) => planEntry({ ...ctx, database: v })); return; }
    let table;
    if (attached) {
      const tables = databases.find((d) => d.name === database).tables;
      table = ctx.table || (tables.includes(verdict.table) ? verdict.table : tables.length === 1 ? tables[0] : null);
      if (!table) { offer("Ke tabel mana?", tables, (v) => planEntry({ ...ctx, database, table: v })); return; }
    }
    const body = attached
      ? await createDataEntryFromFile(attached, { database, table, question: message, ...(mapping ? { mapping } : {}) })
      : await createDataEntry({ question: verdict.question || message, database });
    if (body.status === "planned") push(bot("Ini yang akan dicatat.", { card: { kind: "entry", view: body } }));
    else if (body.status === "needs_mapping") {
      push(bot("Nama kolom file belum cocok semua.", {
        card: { kind: "mapping", view: body, apply: (m) => guarded(() => planEntry({ ...ctx, database, table, mapping: m })) },
      }));
    } else push(bot(body.reason || "Tidak bisa disiapkan."));
  };

  // -- Ubah data -------------------------------------------------------------------------------------------------
  const planChange = async (ctx) => {
    const { message, verdict, attached, mapping } = ctx;
    const [cfg, list] = await Promise.all([getDataChangeConfig().catch(() => null), listDatabaseConnections()]);
    const databases = (list.connections || []).filter((c) => c.writable_tables?.length);
    if (databases.length === 0) {
      push(bot("Belum ada koneksi yang tabelnya disetujui untuk diubah. Atur di tab Koneksi (Ubah persetujuan).", { link: CONNECTIONS }));
      return;
    }
    const names = databases.map((d) => d.name);
    const database = ctx.database || (names.includes(verdict.database) ? verdict.database : names.length === 1 ? names[0] : null);
    if (!database) { offer("Diubah di database mana?", names, (v) => planChange({ ...ctx, database: v })); return; }
    const maxRows = cfg ? Math.min(cfg.default_max_rows, cfg.max_rows_ceiling) : 10;
    let table;
    if (attached) {
      const tables = databases.find((d) => d.name === database).writable_tables;
      table = ctx.table || (tables.includes(verdict.table) ? verdict.table : tables.length === 1 ? tables[0] : null);
      if (!table) { offer("Tabel mana yang diubah?", tables, (v) => planChange({ ...ctx, database, table: v })); return; }
    }
    const body = attached
      ? await createDataChangePlanFromFile(attached, { database, table, maxRows, question: message, ...(mapping ? { mapping } : {}) })
      : await createDataChangePlan({ question: verdict.question || message, max_rows: maxRows, database });
    if (body.status === "planned") push(bot("Ini perubahan yang akan dilakukan.", { card: { kind: "change", view: body } }));
    else if (body.status === "needs_mapping") {
      push(bot("Nama kolom file belum cocok semua.", {
        card: { kind: "mapping", view: body, required: body.match,
          apply: (m) => guarded(() => planChange({ ...ctx, database, table, mapping: m })) },
      }));
    } else push(bot(body.reason || "Tidak bisa direncanakan."));
  };

  // -- Lihat data ------------------------------------------------------------------------------------------------
  const planReport = async (ctx) => {
    const { message, verdict } = ctx;
    const active = ((await listDatabaseConnections()).connections || []).filter((c) => c.status === "active");
    if (active.length === 0) {
      push(bot("Belum ada database yang terhubung dan disetujui. Hubungkan dulu di tab Koneksi.", { link: CONNECTIONS }));
      return;
    }
    const names = active.map((c) => c.name);
    const database = ctx.database || (names.includes(verdict.database) ? verdict.database : names.length === 1 ? names[0] : null);
    if (!database) { offer("Dilihat dari database mana?", names, (v) => planReport({ ...ctx, database: v })); return; }
    // Said in the message ("dalam bentuk Excel"), or asked now: the plan is made for one format.
    const wanted = ctx.format !== undefined ? ctx.format : (verdict.format || undefined);
    if (wanted === undefined) {
      offer("Hasilnya dalam bentuk apa?", Object.keys(REPORT_FORMATS),
        (label) => planReport({ ...ctx, database, format: REPORT_FORMATS[label] }));
      return;
    }
    const body = await createReportPlan({ question: (verdict.question || message).trim(), database, ...(wanted ? { format: wanted } : {}) });
    if (body.status === "planned") push(bot("Ini rencana laporannya.", { card: { kind: "report", view: body } }));
    else push(bot(null, { card: { kind: "result", view: body } }));
  };

  // -- Routing ---------------------------------------------------------------------------------------------------
  const handle = async (message, attached) => {
    if (asking) {                                   // the answer to "what is the database called?"
      const pending = asking;
      setAsking(null);
      await planDatabase({ ...pending, database: message.trim() });
      return;
    }
    if (attached && isImageFile(attached)) {        // a picture is only for designing a database (an ERD)
      await planDatabase({ question: message, attached });
      return;
    }
    const verdict = attached ? await routeAssistantFile(attached, { message }) : await routeAssistantMessage(message);
    if (verdict.refused) { push(bot(verdict.refused)); return; }
    const ctx = { message, verdict, attached };
    if (verdict.intent === "create_database") {
      if (attached && !verdict.database) {
        setAsking({ question: message, table: verdict.table, attached });
        push(bot("Rancangan tabel diambil dari file itu. Nama database-nya apa? (harus diawali awalan server, mis. hai_toko)"));
        return;
      }
      await planDatabase({ question: verdict.question || message, database: verdict.database, table: verdict.table, attached });
    } else if (verdict.intent === "record_data") {
      await planEntry(ctx);
    } else if (verdict.intent === "change_data") {
      await planChange(ctx);
    } else if (verdict.intent === "view_data") {
      if (attached) push(bot("Lihat data belum menerima file. Tulis pertanyaan tentang data yang sudah ada di database."));
      else await planReport(ctx);
    } else if (verdict.denied) {
      push(bot("Saya paham maksud Anda, tetapi akun Anda tidak punya izin untuk alur itu. Hubungi admin bila perlu."));
    } else {
      push(bot("Saya belum yakin maksudnya. Apakah Anda ingin membuat database baru, mencatat data baru, mengubah data yang ada, atau melihat data? Tulis lebih rinci, atau lampirkan file."));
    }
  };

  const send = async (event) => {
    event.preventDefault();
    const message = text.trim();
    if (busy || (!message && !file)) return;
    const attached = file;
    push(me(attached ? `${message}${message ? "\n" : ""}📎 ${attached.name}` : message));
    setText("");
    clearFile();
    await guarded(() => handle(message, attached));
  };

  // -- Running and cancelling a card -----------------------------------------------------------------------------
  const stepUp = (error, kind, id) => {
    if (errorCode(error) !== "step_up_required") return false;
    remember(kind, id);
    push(bot("Untuk keamanan, Anda akan diminta login ulang ke Keycloak..."));
    signInAgain();
    return true;
  };

  const run = (item, extra) => guarded(async () => {
    const { kind, view } = item.card;
    try {
      if (kind === "database") {
        const made = await runDatabasePlan(view.plan_id, view.confirm_phrase, view.plan_id);
        replace(item.id, { card: null });
        push(bot(
          made.mode === "add_tables"
            ? `Tabel ${made.tables.join(", ")} ditambahkan ke database ${made.database} di server ${made.profile}.`
            : `Database ${made.database} dibuat di server ${made.profile}${made.tables.length ? ` dengan tabel ${made.tables.join(", ")}` : ""}. `
              + "Hubungkan dan setujui tabelnya di tab Koneksi agar bisa diisi dan dilihat.",
          { link: CONNECTIONS },
        ));
      } else if (kind === "entry") {
        const done = await runDataEntry(view.plan_id, view.plan_id, { confirmDuplicate: extra.confirmDuplicate });
        replace(item.id, { card: null });
        push(bot(`Tersimpan. ${done.rows_affected} baris dicatat pada tabel ${done.table}.`
          + (done.verification !== "verified" ? " Hasilnya belum dapat dipastikan; periksa datanya sebelum mengulang." : "")
          + " Salah catat? Batalkan di tab Catat data.", { link: { to: "/database-agent/entry", label: "Catat data" } }));
      } else if (kind === "change") {
        const phrase = view.confirm_phrase;
        const done = await runDataChangePlan(view.plan_id, phrase, view.plan_id);
        replace(item.id, { card: null });
        push(bot(`Berhasil. ${done.rows_affected} baris ${OPERATION_DONE[done.operation] || "diubah"} pada tabel ${done.table}.`
          + (done.approved_by ? ` Disetujui oleh ${done.approved_by}.` : "")));
      } else if (kind === "report") {
        const key = globalThis.crypto?.randomUUID?.() ?? `run-${Date.now()}`;
        const result = await runReportPlan(view.plan_id, key);
        replace(item.id, { card: null });
        push(bot(null, { card: { kind: "result", view: result } }));
      }
    } catch (error) {
      if (!stepUp(error, kind, view.plan_id)) throw error;
    }
  });

  const cancel = (item) => guarded(async () => {
    const { kind, view } = item.card;
    if (kind === "database") await cancelDatabasePlan(view.plan_id);
    else if (kind === "entry") await cancelDataEntry(view.plan_id);
    else if (kind === "change") await cancelDataChangePlan(view.plan_id);
    else if (kind === "report") await cancelReportPlan(view.plan_id);
    replace(item.id, { card: null });
    push(bot("Rencana dibatalkan. Tidak ada yang berubah."));
  });

  const decide = (item, decision) => guarded(async () => {
    try {
      await decideDataChange(item.card.view.plan_id, decision);
      replace(item.id, { card: null });
      push(bot(decision === "reject" ? "Perubahan ditolak." : "Perubahan disetujui. Pengaju bisa menjalankannya sekarang."));
    } catch (error) {
      if (errorCode(error) !== "step_up_required") throw error;
      push(bot("Untuk keamanan, Anda akan diminta login ulang ke Keycloak. Setelah kembali, daftar persetujuan muncul lagi."));
      signInAgain();
    }
  });

  const refresh = (item) => guarded(async () => {
    const view = await getDataChangePlan(item.card.view.plan_id);
    replace(item.id, { card: { kind: "change", view } });
  });

  const dropMapping = (item) => replace(item.id, { card: null });

  const renderCard = (item) => {
    const { card } = item;
    if (!card) return null;
    const common = { busy, onRun: (extra) => run(item, extra), onCancel: () => cancel(item) };
    if (card.kind === "database") return <DatabaseCard view={card.view} {...common} />;
    if (card.kind === "entry") return <EntryCard view={card.view} {...common} />;
    if (card.kind === "change") return <ChangeCard view={card.view} {...common} onRefresh={() => refresh(item)} />;
    if (card.kind === "report") return <ReportCard view={card.view} {...common} />;
    if (card.kind === "approval") return <ApprovalCard view={card.view} busy={busy} onDecide={(d) => decide(item, d)} />;
    if (card.kind === "result") return <ReportResult result={card.view} onError={fail} />;
    if (card.kind === "mapping") {
      return (
        <FileMapping key={card.view.headers.join("|")} state={card.view} busy={busy} required={card.required}
          note={card.required ? `Kolom pencari baris: ${card.required}; kolom file untuk itu harus dipilih.` : undefined}
          onApply={(m) => { dropMapping(item); card.apply(m); }} onCancel={() => dropMapping(item)} />
      );
    }
    return null;
  };

  return (
    <main className="at-page">
      <header className="at-header">
        <h1>Asisten</h1>
        <p>Satu tempat untuk semuanya: tulis permintaan dan lampirkan file, asisten memilih alur yang tepat.</p>
        {items.length > 1 && (
          <button type="button" className="at-button at-button-secondary" disabled={busy} onClick={newConversation}>
            Percakapan baru
          </button>
        )}
      </header>
      <section className="at-panel" aria-label="Percakapan" style={{ display: "grid", gap: 10 }}>
        {items.map((item) => (
          <div key={item.id} style={{ justifySelf: item.from === "me" ? "end" : "start", maxWidth: "min(760px, 100%)" }}>
            {(item.text || item.link) && (
              <p role={item.error ? "alert" : undefined} style={{ margin: 0, whiteSpace: "pre-wrap" }}>
                {item.error && <AlertCircle size={14} aria-hidden="true" />} {item.text}
                {item.link && <> <Link to={item.link.to}>{item.link.label}</Link></>}
              </p>
            )}
            {item.choices && (
              <div className="at-form-row">
                {item.choices.map((c) => (
                  <button key={c.label} type="button" className="at-button at-button-secondary" disabled={busy}
                    onClick={() => { replace(item.id, { choices: null }); c.pick(); }}>{c.label}</button>
                ))}
              </div>
            )}
            {renderCard(item)}
          </div>
        ))}
        {busy && <p aria-live="polite"><LoaderCircle size={14} className="spin" /> Memproses...</p>}
        <div ref={end} />
      </section>
      <form className="at-form" onSubmit={send} aria-label="Kirim pesan">
        {file && (
          <small>
            📎 {file.name}{" "}
            {isImageFile(file) && <b>(gambar dikirim utuh ke model AI untuk membuat database; jangan unggah yang memuat data pribadi) </b>}
            <button type="button" className="at-button at-button-secondary" onClick={clearFile} aria-label="Hapus lampiran">
              <X size={12} />
            </button>
          </small>
        )}
        <div className="at-form-row">
          <label style={{ cursor: "pointer" }}>
            <Paperclip size={16} aria-hidden="true" />
            <input ref={fileInput} type="file" accept=".csv,.tsv,.txt,.xlsx,.png,.jpg,.jpeg,.webp" aria-label="Lampirkan file"
              style={{ display: "none" }} onChange={(e) => setFile(e.target.files?.[0] || null)} />
          </label>
          <textarea value={text} onChange={(e) => setText(e.target.value)} rows={2} style={{ flex: 1 }}
            aria-label="Pesan" placeholder="Contoh: catat penjualan 10 pcs kopi 25 ribu" />
          <button className="at-button at-button-primary" disabled={busy || (!text.trim() && !file)}>
            <Send size={14} /> Kirim
          </button>
        </div>
      </form>
    </main>
  );
}
