import { useCallback, useEffect, useRef, useState } from "react";
import { AlertCircle, Database, LoaderCircle, Plug, Send, Unplug } from "lucide-react";
import { useLocation } from "react-router-dom";
import {
  approveDatabaseTables,
  cancelReportPlan,
  createEngineRequest,
  createReportPlan,
  downloadReport,
  connectDatabase,
  disconnectDatabase,
  getDatabaseSchema,
  listDatabaseConnections,
  listDatabaseProfiles,
  listEngineRequests,
  runReportPlan,
} from "../api/client.js";
import { describeCondition } from "./conditions.js";
import SourceChoices, { suggestName } from "./SourceChoices.jsx";
import "./autonomous-testing.css";

// A relation is the same pair of columns whichever side is written first.
const relationKey = (r) => [`${r.table}.${r.column}`, `${r.ref_table}.${r.ref_column}`].sort().join(" = ");
const relationText = (r) => `${r.table}.${r.column} = ${r.ref_table}.${r.ref_column}`;

// The server discovers the schema; the user only picks which tables/columns the
// agent may use and which columns may be filtered.
function ApprovalForm({ connection, schema, busy, onApprove, preset = false }) {
  // Starts from what is approved now, so changing an active connection adds to it
  // instead of starting from nothing. With `preset` (a database the agent has just made or extended), every
  // table that is not approved yet starts ticked: all columns readable and filterable and, when the server has
  // a writer account, Tambah and Ubah on every column except the key. Hapus is never pre-ticked. Nothing takes
  // effect until the user presses the approve button.
  const [selection, setSelection] = useState(() => {
    const base = Object.fromEntries(
      Object.entries(schema.approved || {}).map(([table, v]) => [
        table, {
          columns: v.columns, filterable: v.filterable_columns || [],
          writable: v.writable ? { operations: v.writable.operations || [], columns: v.writable.columns || [] } : null,
        },
      ]),
    );
    if (preset) {
      for (const table of schema.tables) {
        if (base[table.name]) continue;
        const names = table.columns.map((c) => c.name);
        const writable = schema.writer_available ? table.columns.filter((c) => !c.primary_key).map((c) => c.name) : [];
        base[table.name] = {
          columns: names, filterable: names,
          writable: writable.length ? { operations: ["insert", "update"], columns: writable } : null,
        };
      }
    }
    return base;
  });

  // Relations let the agent combine tables (join). The database's own foreign keys are offered, and a user can
  // add a relation the database does not define. A first-time approval starts with the foreign keys ticked.
  const [relations, setRelations] = useState(() => {
    const approved = schema.approved_relations || [];
    // Foreign keys start ticked until someone has approved relations once; after that, only their choice counts.
    const fresh = preset || !approved.length;
    const found = new Map();
    for (const r of [...approved, ...(fresh ? schema.foreign_keys || [] : [])]) found.set(relationKey(r), r);
    return [...found.values()];
  });
  const [offered, setOffered] = useState(() => {
    const found = new Map();
    // Suggestions are guesses from names, so they are listed last and never start ticked.
    for (const r of [
      ...(schema.suggested_relations || []), ...(schema.foreign_keys || []), ...(schema.approved_relations || []),
    ]) {
      found.set(relationKey(r), r);
    }
    return [...found.values()];
  });
  const [draft, setDraft] = useState({ table: "", column: "", ref_table: "", ref_column: "" });
  const usable = (r) =>
    Boolean(selection[r.table]?.columns.includes(r.column) && selection[r.ref_table]?.columns.includes(r.ref_column));
  const toggleRelation = (r) =>
    setRelations((cur) =>
      cur.some((x) => relationKey(x) === relationKey(r))
        ? cur.filter((x) => relationKey(x) !== relationKey(r))
        : [...cur, r],
    );
  const columnsOf = (table) => selection[table]?.columns || [];
  const draftReady = Object.values(draft).every(Boolean) && draft.table !== draft.ref_table;
  const addRelation = () => {
    if (!draftReady) return;
    const relation = { ...draft };
    setOffered((cur) => (cur.some((x) => relationKey(x) === relationKey(relation)) ? cur : [...cur, relation]));
    setRelations((cur) => (cur.some((x) => relationKey(x) === relationKey(relation)) ? cur : [...cur, relation]));
    setDraft({ table: "", column: "", ref_table: "", ref_column: "" });
  };
  const isForeignKey = (r) => (schema.foreign_keys || []).some((k) => relationKey(k) === relationKey(r));
  const relationLabel = (r) => {
    if (isForeignKey(r)) return "foreign key database";
    const approved = (schema.approved_relations || []).find((k) => relationKey(k) === relationKey(r));
    if (approved) return approved.source === "ai" ? "ditemukan AI, disetujui saat laporan dijalankan" : "dibuat manual";
    const guess = (schema.suggested_relations || []).find((k) => relationKey(k) === relationKey(r));
    return guess ? `saran dari nama: ${guess.reason}` : "dibuat manual";
  };

  const toggleTable = (table, columns) =>
    setSelection((cur) => {
      const next = { ...cur };
      if (next[table]) delete next[table];
      else next[table] = { columns: columns.map((c) => c.name), filterable: [], writable: null };
      return next;
    });

  const toggleColumn = (table, name) =>
    setSelection((cur) => {
      const entry = cur[table];
      if (!entry) return cur;
      const has = entry.columns.includes(name);
      const columns = has ? entry.columns.filter((c) => c !== name) : [...entry.columns, name];
      return {
        ...cur,
        [table]: {
          ...entry, columns, filterable: entry.filterable.filter((c) => columns.includes(c)),
          writable: entry.writable
            ? { ...entry.writable, columns: entry.writable.columns.filter((c) => columns.includes(c)) }
            : null,
        },
      };
    });

  const toggleOperation = (table, operation) =>
    setSelection((cur) => {
      const entry = cur[table];
      const writable = entry.writable || { operations: [], columns: [] };
      const operations = writable.operations.includes(operation)
        ? writable.operations.filter((o) => o !== operation)
        : [...writable.operations, operation];
      return { ...cur, [table]: { ...entry, writable: operations.length ? { ...writable, operations } : null } };
    });

  const toggleWritableColumn = (table, name) =>
    setSelection((cur) => {
      const entry = cur[table];
      const writable = entry.writable || { operations: [], columns: [] };
      const columns = writable.columns.includes(name)
        ? writable.columns.filter((c) => c !== name)
        : [...writable.columns, name];
      return { ...cur, [table]: { ...entry, writable: { ...writable, columns } } };
    });

  const toggleFilter = (table, name) =>
    setSelection((cur) => {
      const entry = cur[table];
      const has = entry.filterable.includes(name);
      const filterable = has ? entry.filterable.filter((c) => c !== name) : [...entry.filterable, name];
      return { ...cur, [table]: { ...entry, filterable } };
    });

  const submit = (event) => {
    event.preventDefault();
    const tables = {};
    for (const [table, entry] of Object.entries(selection)) {
      if (!entry.columns.length) continue;
      const found = schema.tables.find((t) => t.name === table);
      const key = found.columns.find((c) => c.primary_key && entry.columns.includes(c.name));
      tables[table] = {
        columns: entry.columns,
        filterable_columns: entry.filterable,
        ...(key ? { key_column: key.name } : {}),
        ...(entry.writable?.operations.length
          ? { writable: { operations: entry.writable.operations, columns: entry.writable.columns } }
          : {}),
      };
    }
    const chosen = relations
      .filter(usable)
      .map(({ table, column, ref_table, ref_column }) => ({ table, column, ref_table, ref_column }));
    onApprove(tables, chosen);
  };

  return (
    <form className="at-form" onSubmit={submit} aria-label="Setujui tabel">
      <div className="at-warning">
        <AlertCircle size={16} />
        <div>
          <b>Hanya pilih data yang boleh dilihat agent.</b>
          <p>Kolom yang dipilih tampil di katalog LLM. Jangan pilih kolom sensitif (PII).</p>
        </div>
      </div>
      {schema.tables.map((table) => {
        const entry = selection[table.name];
        return (
          <fieldset key={table.name} className="at-config-summary">
            <label className="at-check">
              <input
                type="checkbox"
                checked={Boolean(entry)}
                onChange={() => toggleTable(table.name, table.columns)}
              />
              <b>{table.name}</b>
            </label>
            {entry &&
              table.columns.map((column) => (
                <div key={column.name} className="at-form-row">
                  <label className="at-check">
                    <input
                      type="checkbox"
                      checked={entry.columns.includes(column.name)}
                      onChange={() => toggleColumn(table.name, column.name)}
                    />
                    {column.name} <small>{column.declared_type || "text"}</small>
                  </label>
                  <label className="at-check">
                    <input
                      type="checkbox"
                      disabled={!entry.columns.includes(column.name)}
                      checked={entry.filterable.includes(column.name)}
                      onChange={() => toggleFilter(table.name, column.name)}
                    />
                    boleh difilter
                  </label>
                  {schema.writer_available && !column.primary_key && (
                    <label className="at-check">
                      <input
                        type="checkbox"
                        disabled={!entry.columns.includes(column.name)}
                        checked={Boolean(entry.writable?.columns.includes(column.name))}
                        onChange={() => toggleWritableColumn(table.name, column.name)}
                      />
                      boleh diubah
                    </label>
                  )}
                </div>
              ))}
            {entry && schema.writer_available && (
              <div className="at-form-row" aria-label={`Izin ubah data ${table.name}`}>
                <b>Izin ubah data:</b>
                {[["insert", "Tambah"], ["update", "Ubah"], ["delete", "Hapus"]].map(([op, label]) => (
                  <label key={op} className="at-check">
                    <input
                      type="checkbox"
                      checked={Boolean(entry.writable?.operations.includes(op))}
                      onChange={() => toggleOperation(table.name, op)}
                    />
                    {label}
                  </label>
                ))}
              </div>
            )}
          </fieldset>
        );
      })}
      {Object.keys(selection).length > 1 && (
        <fieldset className="at-config-summary" aria-label="Relasi antar tabel">
          <b>Relasi antar tabel (agar agent bisa menggabungkan tabel)</b>
          <small>
            Centang relasi yang boleh dipakai. Agent hanya menggabungkan tabel lewat relasi yang Anda setujui di sini.
          </small>
          {!offered.some((r) => selection[r.table] && selection[r.ref_table]) && (
            <small>Tidak ada foreign key atau saran dari nama kolom untuk tabel yang dipilih. Tambahkan relasi di bawah bila ada.</small>
          )}
          {offered.filter((r) => selection[r.table] && selection[r.ref_table]).map((r) => (
            <label key={relationKey(r)} className="at-check">
              <input
                type="checkbox"
                disabled={!usable(r)}
                checked={relations.some((x) => relationKey(x) === relationKey(r)) && usable(r)}
                onChange={() => toggleRelation(r)}
              />
              {relationText(r)} <small>{relationLabel(r)}</small>
              {!usable(r) && <small> (pilih kedua tabel dan kolomnya dulu)</small>}
            </label>
          ))}
          <div className="at-form-row" aria-label="Tambah relasi">
            <b>Tambah relasi:</b>
            {[["table", "column", "Tabel"], ["ref_table", "ref_column", "dengan tabel"]].map(([tKey, cKey, label]) => (
              <span key={tKey} className="at-form-row">
                <select
                  aria-label={label}
                  value={draft[tKey]}
                  onChange={(e) => setDraft((d) => ({ ...d, [tKey]: e.target.value, [cKey]: "" }))}
                >
                  <option value="">{label}…</option>
                  {Object.keys(selection).map((t) => <option key={t} value={t}>{t}</option>)}
                </select>
                <select
                  aria-label={`Kolom ${label}`}
                  value={draft[cKey]}
                  disabled={!draft[tKey]}
                  onChange={(e) => setDraft((d) => ({ ...d, [cKey]: e.target.value }))}
                >
                  <option value="">kolom…</option>
                  {columnsOf(draft[tKey]).map((c) => <option key={c} value={c}>{c}</option>)}
                </select>
              </span>
            ))}
            <button type="button" className="at-button at-button-secondary" disabled={!draftReady} onClick={addRelation}>
              Tambah relasi
            </button>
          </div>
        </fieldset>
      )}
      <button className="at-button at-button-primary" disabled={busy || !Object.keys(selection).length}>
        Setujui untuk {connection}
      </button>
    </form>
  );
}

function ResultTable({ rows }) {
  if (!rows?.length) return null;
  const headers = Object.keys(rows[0]);
  return (
    <div style={{ overflowX: "auto" }}>
      <table aria-label="Hasil laporan" style={{ width: "100%", fontSize: 12, borderCollapse: "collapse" }}>
        <thead>
          <tr>
            {headers.map((h) => (
              <th key={h} style={{ textAlign: "left", padding: 6 }}>{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr key={i}>
              {headers.map((h) => (
                <td key={h} style={{ padding: 6 }}>{String(row[h] ?? "")}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// The agent may wait on a slow or stuck LLM provider before switching to the next one, so the page
// must say what is happening, count the seconds, and let the user stop waiting.
export const PLAN_TIMEOUT_MS = 4 * 60 * 1000;

const STATUS_TEXT = (c) =>
  c.status === "active"
    ? `aktif (${c.tables.join(", ")})`
    : c.status === "needs_reapproval"
      ? `perlu persetujuan ulang${c.reason ? `: ${c.reason}` : ""}`
      : c.status === "unavailable"
        ? `tidak tersedia${c.reason ? `: ${c.reason}` : ""}`
        : "menunggu persetujuan";

// ``part``: "connections" (connect databases, approve tables), "view" (ask for reports), or "all" (both).
const HEADERS = {
  all: ["DATA REPORTS", "Laporan dari database",
    "Hubungkan database SQLite dari folder data server, setujui tabelnya, lalu tanyakan dalam bahasa biasa. "
    + "Akses hanya-baca; agent hanya melihat data yang Anda setujui."],
  connections: ["KONEKSI", "Koneksi database",
    "Hubungkan database, lalu setujui tabel dan kolom yang boleh dilihat atau diisi agent. "
    + "Agent hanya menyentuh apa yang Anda setujui di sini."],
  view: ["LIHAT DATA", "Lihat data",
    "Tanyakan data dalam bahasa biasa dan dapatkan tabel, atau file Excel, PDF, HTML, dan CSV. "
    + "Akses hanya-baca; agent hanya melihat data yang sudah disetujui."],
};

export default function DataReports({ part = "all" }) {
  const [connections, setConnections] = useState([]);
  const [selected, setSelected] = useState("");
  const [schema, setSchema] = useState(null);
  const [name, setName] = useState("");
  const [filename, setFilename] = useState("");
  const [profiles, setProfiles] = useState([]);
  const [source, setSource] = useState("");
  const [database, setDatabase] = useState("");
  const location = useLocation();
  const [notice, setNotice] = useState(location.state?.message || "");
  const [preset, setPreset] = useState(Boolean(location.state?.review));
  const chosenProfile = profiles.find((p) => p.name === source);
  const [warnings, setWarnings] = useState([]);
  const [engineRequests, setEngineRequests] = useState([]);
  const [wantedEngine, setWantedEngine] = useState("");
  const [wantedReason, setWantedReason] = useState("");
  const [question, setQuestion] = useState("");
  const [result, setResult] = useState(null);
  const [plan, setPlan] = useState(null);
  const [format, setFormat] = useState("");
  const [planningSince, setPlanningSince] = useState(null);
  const [elapsed, setElapsed] = useState(0);
  const planAbort = useRef(null);
  const cancelledByUser = useRef(false);

  useEffect(() => {
    if (planningSince === null) {
      setElapsed(0);
      return undefined;
    }
    const timer = setInterval(() => setElapsed(Math.floor((Date.now() - planningSince) / 1000)), 1000);
    return () => clearInterval(timer);
  }, [planningSince]);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  const refresh = useCallback(async () => {
    try {
      const data = await listDatabaseConnections();
      setConnections(data.connections || []);
    } catch (err) {
      setError(err.message);
    }
  }, []);

  useEffect(() => {
    refresh();
    listEngineRequests()
      .then((data) => setEngineRequests(data.requests || []))
      .catch(() => setEngineRequests([]));
    listDatabaseProfiles()
      .then((data) => setProfiles(data.profiles || []))
      .catch(() => setProfiles([]));
  }, [refresh]);

  const guarded = async (action) => {
    setBusy(true);
    setError("");
    try {
      await action();
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  };

  const connect = (event) => {
    event.preventDefault();
    return guarded(async () => {
      const payload = source
        ? { name: name.trim(), profile: source, ...(database.trim() ? { database: database.trim() } : {}) }
        : { name: name.trim(), filename: filename.trim() };
      const data = await connectDatabase(payload);
      setWarnings(data.warnings || []);
      setSchema(data.schema);
      setSelected(data.name);
      setResult(null);
      await refresh();
    });
  };

  const approve = (tables, relations) =>
    guarded(async () => {
      if (relations?.length) await approveDatabaseTables(selected, tables, undefined, relations);
      else await approveDatabaseTables(selected, tables);
      setSchema(null);
      setPreset(false);
      setNotice("");
      await refresh();
    });

  const disconnect = (target) =>
    guarded(async () => {
      await disconnectDatabase(target);
      if (selected === target) {
        setSelected("");
        setSchema(null);
      }
      await refresh();
    });

  const review = (target) =>
    guarded(async () => {
      setSchema(await getDatabaseSchema(target));
      setSelected(target);
      setResult(null);
    });

  // Arriving from "Buat database": the new database is already connected; open its approval form at once.
  useEffect(() => {
    const target = location.state?.review;
    if (!target) return;
    window.history.replaceState({}, "");
    review(target);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const requestEngine = (event) => {
    event.preventDefault();
    return guarded(async () => {
      await createEngineRequest({ engine: wantedEngine.trim(), reason: wantedReason.trim() });
      setWantedEngine("");
      setWantedReason("");
      setEngineRequests((await listEngineRequests()).requests || []);
    });
  };

  const makePlan = async (event) => {
    event.preventDefault();
    setBusy(true);
    setError("");
    setResult(null);
    setPlan(null);
    const controller = new AbortController();
    planAbort.current = controller;
    cancelledByUser.current = false;
    const timeout = setTimeout(() => controller.abort(), PLAN_TIMEOUT_MS);
    setPlanningSince(Date.now());
    try {
      const data = await createReportPlan(
        { question: question.trim(), database: selected || undefined, format: format || undefined },
        controller.signal,
      );
      if (data.status === "planned") setPlan(data);
      else setResult(data);
    } catch (err) {
      if (controller.signal.aborted) {
        setError(
          cancelledByUser.current
            ? "Pembuatan rencana dibatalkan."
            : "Agent terlalu lama menjawab (lebih dari 4 menit). Periksa penyedia LLM di server dengan: "
                + "python scripts/check_llm_provider.py",
        );
      } else {
        setError(err.message);
      }
    } finally {
      clearTimeout(timeout);
      planAbort.current = null;
      setPlanningSince(null);
      setBusy(false);
    }
  };

  const cancelPlanning = () => {
    cancelledByUser.current = true;
    planAbort.current?.abort();
  };

  const runPlan = () =>
    guarded(async () => {
      // One key per click: a retried request replays the stored result instead of running twice.
      const key = globalThis.crypto?.randomUUID?.() ?? `run-${Date.now()}`;
      setResult(await runReportPlan(plan.plan_id, key));
      setPlan(null);
    });

  const saveFile = (info) =>
    guarded(async () => {
      const { blob, filename } = await downloadReport(info.url);
      const href = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = href;
      link.download = filename;
      document.body.appendChild(link);
      link.click();
      link.remove();
      URL.revokeObjectURL(href);
    });

  const dropPlan = () =>
    guarded(async () => {
      await cancelReportPlan(plan.plan_id);
      setPlan(null);
    });

  const active = connections.filter((c) => c.status === "active");
  const filters = result?.plan?.filters || {};

  return (
    <main className="at-shell">
      <header className="at-header">
        <div>
          <span className="at-eyebrow">{HEADERS[part][0]}</span>
          <h1>{HEADERS[part][1]}</h1>
          <p className="at-subtitle">{HEADERS[part][2]}</p>
        </div>
      </header>
      {warnings.length > 0 && (
        <div className="at-warning"><AlertCircle size={16} /><div>{warnings.join(" ")}</div></div>
      )}
      {notice && <div className="at-config-summary" role="status">{notice}</div>}
      {error && (
        <div className="at-alert" role="alert"><AlertCircle size={17} /><span>{error}</span></div>
      )}
      <div className="at-layout" style={part === "view" || (part === "connections" && !schema) ? { gridTemplateColumns: "minmax(0, 1fr)" } : undefined}>
        {part !== "view" && (
        <section className="at-panel">
          <div className="at-section-heading"><Database size={18} /><h2>Koneksi</h2></div>
          <form className="at-form" onSubmit={connect} aria-label="Hubungkan database">
            <label>
              Nama koneksi
              <input value={name} onChange={(e) => setName(e.target.value)} placeholder="sales" />
            </label>
            {profiles.length > 0 && (
              <label>
                Sumber
                <select value={source} onChange={(e) => { setSource(e.target.value); setDatabase(""); }}>
                  <option value="">File SQLite di folder data</option>
                  {profiles.map((p) => (
                    <option key={p.name} value={p.name}>{`${p.name} (${p.type})`}</option>
                  ))}
                </select>
              </label>
            )}
            <SourceChoices
              source={source}
              selected={source ? (database || undefined) : filename}
              onPick={(item) => {
                if (source) setDatabase(item.default ? "" : item.name);
                else setFilename(item.name);
                if (!name.trim() && suggestName(item.name)) setName(suggestName(item.name));
              }}
            />
            {(chosenProfile?.prefix || chosenProfile?.any_database) && (
              <label>
                Database <small>opsional; kosong = database bawaan profil</small>
                <input
                  value={database}
                  onChange={(e) => setDatabase(e.target.value)}
                  placeholder={chosenProfile.any_database ? "nama_database" : `${chosenProfile.prefix}nama_database`}
                  aria-label="Database buatan agent"
                />
                <small>
                  {chosenProfile.any_database
                    ? "Tulis nama database mana pun di server ini yang bisa dibaca akun baca-saja (juga yang tidak dibuat aplikasi ini)."
                    : `Untuk database yang dibuat agent, tulis namanya (diawali ${chosenProfile.prefix}).`}
                  {" "}Nama koneksi di atas hanya label; ia tidak memilih database.
                </small>
              </label>
            )}
            {!source && (
              <label>
                Nama file <small>relatif terhadap folder data server</small>
                <input
                  value={filename}
                  onChange={(e) => setFilename(e.target.value)}
                  placeholder="sales.sqlite3"
                />
              </label>
            )}
            <button
              className="at-button at-button-primary"
              disabled={busy || !name.trim() || (!source && !filename.trim())}
            >
              <Plug size={14} /> Hubungkan
            </button>
          </form>
          <details style={{ marginTop: 14 }}>
            <summary>Database Anda belum didukung?</summary>
            <form className="at-form" onSubmit={requestEngine} aria-label="Minta engine baru">
              <label>
                Nama database
                <input value={wantedEngine} onChange={(e) => setWantedEngine(e.target.value)} placeholder="Oracle 19c" />
              </label>
              <label>
                Alasan
                <input value={wantedReason} onChange={(e) => setWantedReason(e.target.value)} placeholder="Data keuangan kami ada di sana" />
              </label>
              <button className="at-button at-button-secondary" disabled={busy || !wantedEngine.trim() || !wantedReason.trim()}>
                Kirim permintaan
              </button>
            </form>
            <ul style={{ listStyle: "none", padding: 0 }}>
              {engineRequests.map((r) => (
                <li key={r.id}>{r.engine} · {r.status}{r.note ? ` (${r.note})` : ""}</li>
              ))}
            </ul>
          </details>
          <ul style={{ listStyle: "none", padding: 0, marginTop: 14 }}>
            {connections.map((c) => (
              <li key={c.name} style={{ display: "flex", justifyContent: "space-between", padding: "6px 0" }}>
                <span>
                  <b>{c.name}</b> ·{" "}
                  {STATUS_TEXT(c)}
                  {c.file && <small style={{ display: "block", opacity: 0.7 }}>sumber: {c.file}</small>}
                </span>
                {c.status !== "unavailable" && (
                  <button type="button" className="at-button at-button-secondary" disabled={busy} onClick={() => review(c.name)}>
                    {c.status === "active" ? "Ubah persetujuan" : "Setujui"}
                  </button>
                )}
                <button
                  type="button"
                  className="at-button at-button-secondary"
                  disabled={busy}
                  onClick={() => disconnect(c.name)}
                  aria-label={`Putuskan ${c.name}`}
                >
                  <Unplug size={14} />
                </button>
              </li>
            ))}
          </ul>
        </section>
        )}
        {(part !== "connections" || schema) && (
        <section className="at-panel">
          {schema ? (
            <ApprovalForm key={selected} connection={selected} schema={schema} busy={busy} onApprove={approve} preset={preset} />
          ) : (
            <>
              <div className="at-section-heading"><Send size={18} /><h2>Tanya laporan</h2></div>
              <form className="at-form" onSubmit={makePlan} aria-label="Tanya laporan">
                <label>
                  Database
                  <select value={selected} onChange={(e) => setSelected(e.target.value)}>
                    <option value="">{active.length === 1 ? "(otomatis)" : "Pilih database"}</option>
                    {active.map((c) => (
                      <option key={c.name} value={c.name}>{c.name}</option>
                    ))}
                  </select>
                </label>
                <label>
                  Pertanyaan
                  <input
                    value={question}
                    onChange={(e) => setQuestion(e.target.value)}
                    placeholder="Total penjualan widget per wilayah"
                  />
                </label>
                <label>
                  Hasil
                  <select value={format} onChange={(e) => setFormat(e.target.value)}>
                    <option value="">Tampilan di layar (file JSON tersedia)</option>
                    <option value="xlsx">Tampilan di layar + file Excel</option>
                    <option value="pdf">Tampilan di layar + file PDF</option>
                    <option value="html">Tampilan di layar + file HTML</option>
                    <option value="csv">Tampilan di layar + file CSV</option>
                  </select>
                </label>
                <button
                  className="at-button at-button-primary"
                  disabled={busy || !question.trim() || !active.length}
                >
                  {busy ? <LoaderCircle size={14} className="spin" /> : <Send size={14} />} Buat rencana
                </button>
              </form>
              {planningSince !== null && (
                <div className="at-notice" role="status" aria-live="polite" style={{ marginTop: 14 }}>
                  <LoaderCircle size={16} className="spin" />
                  <span>
                    Agent sedang menyusun rencana... <b>{elapsed} detik</b>. Biasanya 10 sampai 60 detik; bisa
                    lebih lama bila penyedia LLM pertama macet dan harus beralih ke yang kedua.
                  </span>
                  <button type="button" className="at-button at-button-secondary" onClick={cancelPlanning}>
                    Batalkan
                  </button>
                </div>
              )}
              {plan && (
                <div className="at-config-summary" style={{ marginTop: 16 }} aria-label="Rencana laporan">
                  <b>Rencana (belum ada data yang dibaca)</b>
                  <span>Tabel: {plan.plan.table}</span>
                  {plan.plan.proposed_relations?.length > 0 && (
                    <div className="at-warning" aria-label="Relasi usulan AI">
                      <AlertCircle size={16} />
                      <div>
                        <b>Periksa: AI menduga hubungan antar tabel</b>
                        {plan.plan.proposed_relations.map((r) => (
                          <p key={`${r.table}.${r.column}`}>
                            Isi kolom <b>{r.column}</b> di tabel <b>{r.table}</b> sama dengan kolom <b>{r.ref_column}</b>{" "}
                            di tabel <b>{r.ref_table}</b>.
                          </p>
                        ))}
                        <p>
                          Ini dugaan dari nama tabel dan kolom, bukan definisi database. Bila benar, tekan Jalankan:
                          hubungan ini akan diingat untuk pertanyaan berikutnya (bisa dicabut di Ubah persetujuan).
                          Bila ragu, periksa isi hasilnya dulu atau batalkan.
                        </p>
                      </div>
                    </div>
                  )}
                  {plan.plan.joins?.length > 0 && (
                    <span>
                      Gabungkan tabel: {plan.plan.joins
                        .map((j) => `${j.table} (${j.type === "left" ? "semua baris tabel utama" : "hanya yang cocok"}, ${j.on.join(" = ")})`)
                        .join("; ")}
                    </span>
                  )}
                  <span>File: {String(plan.plan.format || "json").toUpperCase()}</span>
                  {plan.plan.kind === "list" ? (
                    <>
                      <span>Jenis: daftar baris (data apa adanya, tanpa ringkasan)</span>
                      <span>Kolom: {plan.plan.columns.join(", ")}</span>

                      <span>
                        Urutan: {plan.plan.order_by?.length
                          ? plan.plan.order_by
                              .map((o) => `${o.column} ${o.direction === "desc" ? "menurun" : "menaik"}`)
                              .join(", ")
                          : "bawaan database"}
                      </span>
                      <span>Maksimal: {plan.plan.limit} baris</span>
                    </>
                  ) : (
                    <>
                      <span>Kelompokkan per: {plan.plan.group_by.join(", ")}</span>
                      <span>
                        Hitung: {plan.plan.metrics.map((m) => `${m.operation}(${m.column || "*"})`).join(", ")}
                      </span>
                    </>
                  )}
                  <span>
                    Filter: {(() => {
                      const parts = [
                        ...Object.entries(plan.plan.filters).map(([k, v]) => `${k} = ${v}`),
                        ...(plan.plan.conditions || []).map(describeCondition),
                      ];
                      return parts.length ? parts.join(", ") : "tidak ada";
                    })()}
                  </span>
                  <div className="at-form-row">
                    <button type="button" className="at-button at-button-primary" disabled={busy} onClick={runPlan}>
                      Jalankan
                    </button>
                    <button type="button" className="at-button at-button-secondary" disabled={busy} onClick={dropPlan}>
                      Batal
                    </button>
                  </div>
                </div>
              )}
              {result && (
                <div style={{ marginTop: 16 }} aria-live="polite">
                  {result.status === "succeeded" && (
                    <>
                      {result.kind === "list" ? (
                        <>
                          <p>
                            Daftar baris dari {result.plan?.table}
                            {Object.keys(filters).length ? `, filter ${JSON.stringify(filters)}` : ""}:
                            menampilkan {result.rows?.length} dari {result.row_count} baris.
                          </p>
                          {result.limit_reached && (
                            <div className="at-warning">
                              <AlertCircle size={16} />
                              <div>
                                <b>Dibatasi {result.plan?.limit} baris pertama.</b>
                                <p>Masih ada baris lain di database. Persempit dengan filter, atau minta jumlah baris lebih besar.</p>
                              </div>
                            </div>
                          )}
                          {result.row_count > (result.rows?.length || 0) && (
                            <p className="muted">
                              Layar menampilkan {result.rows.length} baris pertama. File yang diunduh memuat semua{" "}
                              {result.row_count} baris.
                            </p>
                          )}
                        </>
                      ) : (
                        <p>
                          Laporan selesai: {result.plan?.table} per {result.plan?.group_by?.join(", ")}
                          {Object.keys(filters).length ? `, filter ${JSON.stringify(filters)}` : ""}.
                        </p>
                      )}
                      <ResultTable rows={result.rows} />
                      {result.download && (
                        <button
                          type="button"
                          className="at-button at-button-secondary"
                          style={{ marginTop: 12 }}
                          disabled={busy}
                          onClick={() => saveFile(result.download)}
                        >
                          Unduh {String(result.download.filename || "file").split(".").pop().toUpperCase()}
                        </button>
                      )}
                    </>
                  )}
                  {result.status === "unsupported" && (
                    <div className="at-warning">
                      <AlertCircle size={16} />
                      <div>
                        <b>Belum didukung</b>
                        <p>{result.reason}</p>
                        {result.needs_join && (
                          <p>
                            Untuk menggabungkan tabel, buka tab <b>Koneksi</b>, lalu <b>Ubah persetujuan</b> pada database ini, pastikan
                            kedua tabel disetujui, lalu centang relasinya di kotak <b>Relasi antar tabel</b>
                            (foreign key, saran dari nama kolom, atau tambahkan sendiri). Setelah itu ulangi
                            pertanyaannya.
                          </p>
                        )}
                      </div>
                    </div>
                  )}
                  {result.status === "failed" && (
                    <div className="at-alert">
                      <AlertCircle size={16} />
                      <span>Gagal ({result.error_code}): {result.reason || "lihat log server"}</span>
                    </div>
                  )}
                </div>
              )}
            </>
          )}
        </section>
        )}
      </div>
    </main>
  );
}
