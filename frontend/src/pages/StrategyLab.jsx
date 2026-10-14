
import { useMemo } from "react";
import {
  AlertCircle,
  CircleHelp,
  FlaskConical,
  LoaderCircle,
  RefreshCw,
  ShieldAlert,
  Brain,
  Play,
  Database,
} from "lucide-react";
import "./strategy.css";

const formatNumber = (value, digits = 2) => {
  if (value === null || value === undefined || value === "") {
    return "—";
  }

  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed.toFixed(digits) : "—";
};

const formatPercent = (value) => {
  if (value === null || value === undefined || value === "") {
    return "—";
  }

  const parsed = Number(value);

  if (!Number.isFinite(parsed)) {
    return "—";
  }

  return `${Math.round(parsed * 100)}%`;
};

const formatCount = (value) => {
  if (value === null || value === undefined || value === "") {
    return 0;
  }

  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : 0;
};

const formatLabel = (value, fallback = "—") =>
  typeof value === "string" && value.trim() ? value : fallback;

function StatusTag({ status }) {
  const value = String(status || "unknown")
    .toLowerCase()
    .replace(/[^a-z0-9_-]/g, "-");

  return (
    <span className={`strategy-status status-${value}`}>
      {value}
    </span>
  );
}

function EmptyState({ icon: Icon = CircleHelp, title, description }) {
  return (
    <div className="strategy-no-data">
      <Icon size={20} />
      <div>
        <b>{title}</b>
        <p>{description}</p>
      </div>
    </div>
  );
}

function MetricCard({ label, value, description }) {
  return (
    <article className="strategy-metric">
      <small>{label}</small>
      <strong className="strategy-long-value">{value}</strong>
      <span>{description}</span>
    </article>
  );
}

function SourceNote({ children }) {
  return (
    <div className="strategy-source-note">
      <ShieldAlert size={17} />
      <p>{children}</p>
    </div>
  );
}

function StrategyRegistry({
  title,
  strategies = [],
  persistence,
  emptyTitle,
  emptyDescription,
}) {
  const activeCount = useMemo(
    () =>
      strategies.filter(
        (strategy) =>
          String(strategy.status || "").toLowerCase() === "active"
      ).length,
    [strategies]
  );

  return (
    <section className="panel strategy-panel">
      <div className="panel-head">
        <h3>{title}</h3>
        <span className="tag">{strategies.length} strategi</span>
      </div>

      <div className="strategy-registry-summary">
        <span>
          <b>{strategies.length}</b> total
        </span>
        <span>
          <b>{activeCount}</b> active
        </span>
        <span>
          Persistence: <b>{formatLabel(persistence, "unknown")}</b>
        </span>
      </div>

      {strategies.length > 0 ? (
        <div className="table-scroll">
          <table className="telemetry-table strategy-table">
            <thead>
              <tr>
                <th>Strategi</th>
                <th>Status</th>
                <th>Goal</th>
                <th>Attempts</th>
                <th>Success rate</th>
                <th>Avg. actions</th>
                <th>Avg. progress</th>
                <th>Reflection penalty</th>
              </tr>
            </thead>

            <tbody>
              {strategies.map((strategy, index) => {
                const steps = Array.isArray(strategy.steps)
                  ? strategy.steps
                  : [];

                const experiences = Array.isArray(
                  strategy.source_experiences
                )
                  ? strategy.source_experiences
                  : [];

                return (
                  <tr
                    key={`${strategy.name || "strategy"}-${strategy.goal || ""}-${index}`}
                  >
                    <td>
                      <b>{formatLabel(strategy.name, "Unnamed")}</b>

                      {strategy.is_currently_selected && (
                        <span className="strategy-current-label">
                          Currently selected
                        </span>
                      )}

                      <details className="strategy-steps-details">
                        <summary>Langkah ({steps.length})</summary>

                        {steps.length > 0 ? (
                          <ol>
                            {steps.map((step, stepIndex) => (
                              <li key={`${step}-${stepIndex}`}>
                                <code>{step}</code>
                              </li>
                            ))}
                          </ol>
                        ) : (
                          <p className="muted">
                            Tidak ada langkah yang tercatat.
                          </p>
                        )}
                      </details>

                      <small className="strategy-submetric">
                        {experiences.length} source experiences
                      </small>
                    </td>

                    <td>
                      <StatusTag status={strategy.status} />
                    </td>

                    <td>{formatLabel(strategy.goal)}</td>

                    <td>
                      {formatCount(strategy.attempts)}
                      <small className="strategy-submetric">
                        {formatCount(strategy.success_count)} sukses ·{" "}
                        {formatCount(strategy.failure_count)} gagal
                      </small>
                    </td>

                    <td>{formatPercent(strategy.success_rate)}</td>

                    <td>{formatNumber(strategy.average_actions)}</td>

                    <td>
                      {formatNumber(strategy.average_progress)}
                      <small className="strategy-submetric">
                        {formatCount(strategy.progress_count)} pengukuran
                      </small>
                    </td>

                    <td>{formatNumber(strategy.reflection_penalty)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      ) : (
        <EmptyState
          title={emptyTitle}
          description={emptyDescription}
        />
      )}
    </section>
  );
}

function EvolutionEvidence({ evidence = [] }) {
  return (
    <section className="panel strategy-panel">
      <div className="panel-head">
        <h3>Evolution evidence</h3>
        <span className="tag">{evidence.length} record</span>
      </div>

      {evidence.length > 0 ? (
        <div className="table-scroll">
          <table className="telemetry-table strategy-table">
            <thead>
              <tr>
                <th>Strategi hasil evolusi</th>
                <th>Sumber strategi</th>
                <th>Generasi</th>
                <th>Status evidence</th>
                <th>Trials</th>
                <th>Success rate</th>
                <th>Avg. progress</th>
                <th>Source experiences</th>
              </tr>
            </thead>

            <tbody>
              {evidence.map((item, index) => {
                const sourceIds = Array.isArray(
                  item.source_experience_ids
                )
                  ? item.source_experience_ids
                  : [];

                return (
                  <tr
                    key={`${item.strategy_name || "evidence"}-${index}`}
                  >
                    <td>
                      <b>{formatLabel(item.strategy_name)}</b>
                    </td>

                    <td>
                      {formatLabel(item.source_strategy_name)}
                    </td>

                    <td>{item.generation ?? "—"}</td>

                    <td>
                      <StatusTag status={item.status} />
                    </td>

                    <td>
                      {formatCount(item.trial_count)}
                      <small className="strategy-submetric">
                        {formatCount(item.success_count)} sukses ·{" "}
                        {formatCount(item.failure_count)} gagal
                      </small>
                    </td>

                    <td>{formatPercent(item.success_rate)}</td>

                    <td>{formatNumber(item.average_progress)}</td>

                    <td>
                      <details>
                        <summary>{sourceIds.length} ID</summary>

                        {sourceIds.length > 0 ? (
                          <ul>
                            {sourceIds.map((id) => (
                              <li key={id}>
                                <code>{id}</code>
                              </li>
                            ))}
                          </ul>
                        ) : (
                          <span className="muted">
                            Tidak ada ID tambahan.
                          </span>
                        )}
                      </details>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      ) : (
        <EmptyState
          icon={FlaskConical}
          title="Belum ada evolution evidence"
          description="Belum ada record bukti evolusi strategi pada runtime ini. Statistik keberhasilan strategi tidak otomatis berarti proses evolusi sudah dilakukan."
        />
      )}
    </section>
  );
}

function SelectedStrategyDetail({ strategies = [], activeStrategyName }) {
  const selected = strategies.find(
    (strategy) => strategy.is_currently_selected
  );

  return (
    <section className="panel strategy-panel">
      <div className="panel-head">
        <h3>Active strategy detail</h3>
        <span className="tag">
          {selected ? "Strategi terpilih tersedia" : "Tidak ada strategi terpilih"}
        </span>
      </div>

      {selected ? (
        <>
          <p>
            <b>{selected.name}</b> — {selected.goal}
          </p>

          {selected.steps?.length ? (
            <ol className="strategy-plan-steps">
              {selected.steps.map((step, index) => (
                <li key={`${step}-${index}`}>
                  <code>{step}</code>
                </li>
              ))}
            </ol>
          ) : (
            <p className="muted">Strategi tidak memiliki langkah tercatat.</p>
          )}

          <p className="muted">
            Status pemilihan berasal dari metadata API. Halaman ini tidak
            memilih atau menjalankan strategi.
          </p>
        </>
      ) : (
        <p className="muted">
          {activeStrategyName
            ? `Agent melaporkan active_strategy: ${activeStrategyName}. Tidak ada record registry yang ditandai currently selected.`
            : "Agent tidak melaporkan active_strategy saat ini. Registry masih dapat berisi strategi yang tersimpan."}
        </p>
      )}
    </section>
  );
}

function RuntimeSection({
  title,
  description,
  icon: Icon,
  runtime,
  isSimulation = false,
}) {
  const strategies = Array.isArray(runtime?.strategies)
    ? runtime.strategies
    : [];

  const evidence = Array.isArray(runtime?.evolution_evidence)
    ? runtime.evolution_evidence
    : [];

  return (
    <section className="strategy-runtime-section">
      <div className="strategy-runtime-heading">
        <div className="strategy-runtime-icon">
          <Icon size={20} />
        </div>

        <div>
          <small>RUNTIME SOURCE</small>
          <h3>{title}</h3>
          <p>{description}</p>
        </div>
      </div>

      <div className="strategy-metrics">
        <MetricCard
          label="STRATEGIES"
          value={runtime?.strategy_count ?? strategies.length}
          description="Record dalam registry runtime ini"
        />

        <MetricCard
          label="ACTIVE"
          value={
            strategies.filter(
              (strategy) =>
                String(strategy.status || "").toLowerCase() === "active"
            ).length
          }
          description="Strategi dengan status active"
        />

        <MetricCard
          label="EVOLUTION EVIDENCE"
          value={runtime?.evolution_evidence_count ?? evidence.length}
          description="Record bukti evolusi"
        />

        <MetricCard
          label="PERSISTENCE"
          value={formatLabel(
            runtime?.strategy_persistence,
            "unknown"
          )}
          description="Status penyimpanan strategi"
        />
      </div>

      <SourceNote>
        Sumber: <b>{formatLabel(runtime?.source, "unknown")}</b>. Agent:{" "}
        <b>{formatLabel(runtime?.agent_name, "unknown")}</b>. Strategy
        persistence:{" "}
        <b>{formatLabel(runtime?.strategy_persistence, "unknown")}</b>.
        Evolution evidence persistence:{" "}
        <b>
          {formatLabel(runtime?.evolution_evidence_persistence, "unknown")}
        </b>.
        {isSimulation &&
          " Strategi simulasi berasal dari pengalaman runtime simulasi, bukan bukti bahwa cognitive planner memilih strategi tersebut."}
      </SourceNote>

      <StrategyRegistry
        title={`${title} registry`}
        strategies={strategies}
        persistence={runtime?.strategy_persistence}
        emptyTitle={`Registry ${title} masih kosong`}
        emptyDescription={`Belum ada strategi yang dilaporkan oleh ${title}.`}
      />

      <EvolutionEvidence evidence={evidence} />

      {!isSimulation && (
        <SelectedStrategyDetail
          strategies={strategies}
          activeStrategyName={runtime?.active_strategy_name}
        />
      )}
    </section>
  );
}

export default function StrategyLab({
  data,
  loading = false,
  error = null,
  updatedAt = null,
  onRefresh,
}) {
  const simulation = data?.simulation || null;

  const cognitiveStrategies = Array.isArray(data?.strategies)
    ? data.strategies
    : [];

  const simulationStrategies = Array.isArray(simulation?.strategies)
    ? simulation.strategies
    : [];

  const totalStrategies =
    cognitiveStrategies.length + simulationStrategies.length;

  return (
    <section id="strategies" className="strategy-section">
      <div className="section-head">
        <FlaskConical size={18} />

        <div>
          <small>STRATEGY OBSERVABILITY</small>
          <h2>Strategy Lab</h2>
          <p>
            Observabilitas strategi, hasil pengalaman, dan evolution
            evidence dari cognitive runtime serta simulation runtime.
            Halaman ini tidak memilih atau menjalankan strategi.
          </p>
        </div>

        <button
          type="button"
          className="ghost"
          onClick={onRefresh}
          disabled={loading || typeof onRefresh !== "function"}
        >
          {loading ? (
            <LoaderCircle className="spin" size={14} />
          ) : (
            <RefreshCw size={14} />
          )}

          {loading ? "Memuat..." : "Refresh"}
        </button>
      </div>

      {error && (
        <div className="alert error strategy-alert">
          <AlertCircle size={17} />

          <div>
            <b>Strategy API error</b>
            <pre>{String(error)}</pre>
          </div>
        </div>
      )}

      {loading && !data && (
        <div className="empty strategy-empty">
          <LoaderCircle className="spin" size={25} />
          <h3>Memuat strategi</h3>
          <p>Mengambil registry dari kedua runtime.</p>
        </div>
      )}

      {!loading && !data && !error && (
        <div className="empty strategy-empty">
          <CircleHelp size={25} />
          <h3>Data strategi belum tersedia</h3>
          <p>Belum ada respons dari Strategy API.</p>
        </div>
      )}

      {data && (
        <>
          <div className="strategy-metrics">
            <MetricCard
              label="COGNITIVE STRATEGIES"
              value={data.strategy_count ?? cognitiveStrategies.length}
              description="Registry milik cognitive runtime"
            />

            <MetricCard
              label="SIMULATION STRATEGIES"
              value={simulation?.strategy_count ?? simulationStrategies.length}
              description="Registry milik simulation runtime"
            />

            <MetricCard
              label="TOTAL STRATEGIES"
              value={totalStrategies}
              description="Jumlah record dari kedua runtime"
            />

            <MetricCard
              label="EVOLUTION EVIDENCE"
              value={
                (data.evolution_evidence_count ??
                  data.evolution_evidence?.length ??
                  0) +
                (simulation?.evolution_evidence_count ??
                  simulation?.evolution_evidence?.length ??
                  0)
              }
              description="Total evidence dari kedua runtime"
            />
          </div>

          <div className="strategy-runtime-overview">
            <article className="strategy-runtime-card">
              <Brain size={20} />
              <div>
                <b>Cognitive Runtime</b>
                <span>
                  {formatLabel(data.agent_name, "Unknown agent")}
                </span>
                <small>
                  {formatLabel(data.source, "unknown")} ·{" "}
                  {cognitiveStrategies.length} strategi
                </small>
              </div>
            </article>

            <article className="strategy-runtime-card">
              <Play size={20} />
              <div>
                <b>Simulation Runtime</b>
                <span>
                  {formatLabel(
                    simulation?.agent_name,
                    "Simulation agent unavailable"
                  )}
                </span>
                <small>
                  {simulation
                    ? `${formatLabel(simulation.source, "unknown")} · ${simulationStrategies.length} strategi`
                    : "Data simulation runtime tidak tersedia pada respons API."}
                </small>
              </div>
            </article>

            <article className="strategy-runtime-card">
              <Database size={20} />
              <div>
                <b>Data persistence</b>
                <span>
                  Cognitive:{" "}
                  {formatLabel(
                    data.evolution_evidence_persistence,
                    "unknown"
                  )}
                </span>
                <small>
                  Simulation:{" "}
                  {formatLabel(
                    simulation?.strategy_persistence,
                    "unknown"
                  )}
                </small>
              </div>
            </article>
          </div>

          <RuntimeSection
            title="Cognitive Runtime"
            description="Strategi dan evolution evidence yang dilaporkan oleh agent kognitif."
            icon={Brain}
            runtime={data}
          />

          {simulation ? (
            <RuntimeSection
              title="Simulation Runtime"
              description="Strategi yang tercatat dari pengalaman eksekusi pada runtime simulasi."
              icon={Play}
              runtime={simulation}
              isSimulation
            />
          ) : (
            <section className="panel strategy-panel">
              <EmptyState
                title="Simulation runtime belum tersedia"
                description="Respons Strategy API belum menyertakan properti simulation. Periksa endpoint backend dan pastikan frontend menggunakan respons API terbaru."
              />
            </section>
          )}

          <p className="strategy-updated">
            {updatedAt
              ? `Terakhir diperbarui: ${updatedAt.toLocaleTimeString()}`
              : "Waktu pembaruan belum tersedia."}
            {" · "}
            Data mencerminkan respons runtime saat ini. Status persistence
            menunjukkan konfigurasi penyimpanan, bukan jaminan bahwa setiap
            perubahan telah tersimpan tanpa error.
          </p>
        </>
      )}
    </section>
  );
}
