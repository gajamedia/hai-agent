import { useCallback, useEffect, useState } from "react";
import { AlertCircle, ArrowLeft, Database, LoaderCircle, Plug, Send, Unplug } from "lucide-react";
import { Link } from "react-router-dom";
import {
  approveDatabaseTables,
  askTableReport,
  connectDatabase,
  disconnectDatabase,
  getDatabaseSchema,
  listDatabaseConnections,
  listDatabaseProfiles,
} from "../api/client.js";
import "./autonomous-testing.css";

// The server discovers the schema; the user only picks which tables/columns the
// agent may use and which columns may be filtered.
function ApprovalForm({ connection, schema, busy, onApprove }) {
  const [selection, setSelection] = useState({});

  const toggleTable = (table, columns) =>
    setSelection((cur) => {
      const next = { ...cur };
      if (next[table]) delete next[table];
      else next[table] = { columns: columns.map((c) => c.name), filterable: [] };
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
        [table]: { columns, filterable: entry.filterable.filter((c) => columns.includes(c)) },
      };
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
      };
    }
    onApprove(tables);
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
                </div>
              ))}
          </fieldset>
        );
      })}
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

const STATUS_TEXT = (c) =>
  c.status === "active"
    ? `aktif (${c.tables.join(", ")})`
    : c.status === "needs_reapproval"
      ? `perlu persetujuan ulang${c.reason ? `: ${c.reason}` : ""}`
      : c.status === "unavailable"
        ? `tidak tersedia${c.reason ? `: ${c.reason}` : ""}`
        : "menunggu persetujuan";

export default function DataReports() {
  const [connections, setConnections] = useState([]);
  const [selected, setSelected] = useState("");
  const [schema, setSchema] = useState(null);
  const [name, setName] = useState("");
  const [filename, setFilename] = useState("");
  const [profiles, setProfiles] = useState([]);
  const [source, setSource] = useState("");
  const [warnings, setWarnings] = useState([]);
  const [question, setQuestion] = useState("");
  const [result, setResult] = useState(null);
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
        ? { name: name.trim(), profile: source }
        : { name: name.trim(), filename: filename.trim() };
      const data = await connectDatabase(payload);
      setWarnings(data.warnings || []);
      setSchema(data.schema);
      setSelected(data.name);
      setResult(null);
      await refresh();
    });
  };

  const approve = (tables) =>
    guarded(async () => {
      await approveDatabaseTables(selected, tables);
      setSchema(null);
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

  const ask = (event) => {
    event.preventDefault();
    return guarded(async () => {
      setResult(await askTableReport({ question: question.trim(), database: selected || undefined }));
    });
  };

  const active = connections.filter((c) => c.status === "active");
  const filters = result?.plan?.filters || {};

  return (
    <main className="at-shell">
      <Link className="at-back" to="/"><ArrowLeft size={14} /> Kembali</Link>
      <header className="at-header">
        <div>
          <span className="at-eyebrow">DATA REPORTS</span>
          <h1>Laporan dari database</h1>
          <p className="at-subtitle">
            Hubungkan database SQLite dari folder data server, setujui tabelnya, lalu tanyakan dalam
            bahasa biasa. Akses hanya-baca; agent hanya melihat data yang Anda setujui.
          </p>
        </div>
      </header>
      {warnings.length > 0 && (
        <div className="at-warning"><AlertCircle size={16} /><div>{warnings.join(" ")}</div></div>
      )}
      {error && (
        <div className="at-alert" role="alert"><AlertCircle size={17} /><span>{error}</span></div>
      )}
      <div className="at-layout">
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
                <select value={source} onChange={(e) => setSource(e.target.value)}>
                  <option value="">File SQLite di folder data</option>
                  {profiles.map((p) => (
                    <option key={p.name} value={p.name}>{`${p.name} (${p.type})`}</option>
                  ))}
                </select>
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
          <ul style={{ listStyle: "none", padding: 0, marginTop: 14 }}>
            {connections.map((c) => (
              <li key={c.name} style={{ display: "flex", justifyContent: "space-between", padding: "6px 0" }}>
                <span>
                  <b>{c.name}</b> ·{" "}
                  {STATUS_TEXT(c)}
                </span>
                {c.status !== "active" && c.status !== "unavailable" && (
                  <button type="button" className="at-button" disabled={busy} onClick={() => review(c.name)}>
                    Setujui
                  </button>
                )}
                <button
                  type="button"
                  className="at-button"
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
        <section className="at-panel">
          {schema ? (
            <ApprovalForm connection={selected} schema={schema} busy={busy} onApprove={approve} />
          ) : (
            <>
              <div className="at-section-heading"><Send size={18} /><h2>Tanya laporan</h2></div>
              <form className="at-form" onSubmit={ask} aria-label="Tanya laporan">
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
                <button
                  className="at-button at-button-primary"
                  disabled={busy || !question.trim() || !active.length}
                >
                  {busy ? <LoaderCircle size={14} /> : <Send size={14} />} Buat laporan
                </button>
              </form>
              {result && (
                <div style={{ marginTop: 16 }} aria-live="polite">
                  {result.status === "succeeded" && (
                    <>
                      <p>
                        Laporan selesai: {result.plan?.table} per {result.plan?.group_by?.join(", ")}
                        {Object.keys(filters).length ? `, filter ${JSON.stringify(filters)}` : ""}.
                      </p>
                      <ResultTable rows={result.rows} />
                    </>
                  )}
                  {result.status === "unsupported" && (
                    <div className="at-warning">
                      <AlertCircle size={16} />
                      <div><b>Belum didukung</b><p>{result.reason}</p></div>
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
      </div>
    </main>
  );
}
