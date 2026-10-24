import { useEffect, useRef, useState } from "react";
import {
  AlertCircle,
  ArrowRight,
  BrainCircuit,
  CheckCircle2,
  LoaderCircle,
  RotateCcw,
  ShieldCheck,
  ArrowDownToLine,
} from "lucide-react";
import { runCognitive } from "../api/client.js";
import "./assistant-composer.css";

const EMPTY_PROMPT = "";
const EMPTY_GOAL = "";

function pretty(value) {
  return JSON.stringify(value, null, 2);
}

function listOrEmpty(value) {
  return Array.isArray(value) ? value : [];
}

/**
 * A friendly natural-language entry point for the existing cognitive API.
 * This component deliberately does not execute generated plan steps.
 */
export default function AssistantComposer({ onPrepareExecution }) {
  const [observation, setObservation] = useState(EMPTY_PROMPT);
  const [goal, setGoal] = useState(EMPTY_GOAL);
  const [includeTrace, setIncludeTrace] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [response, setResponse] = useState(null);
  const controllerRef = useRef(null);

  useEffect(() => () => {
    controllerRef.current?.abort();
  }, []);

  async function submit(event) {
    event.preventDefault();
    setError("");
    setResponse(null);

    const normalizedObservation = observation.trim();
    const normalizedGoal = goal.trim();

    if (!normalizedObservation) {
      setError("Jelaskan observasi atau situasi yang perlu dianalisis.");
      return;
    }

    setBusy(true);
    const controller = new AbortController();
    controllerRef.current = controller;

    try {
      const payload = {
        observation: normalizedObservation,
        goal: normalizedGoal || null,
        include_trace: includeTrace,
        metadata: { source: "assistant_workspace" },
      };
      const result = await runCognitive(payload, controller.signal);
      setResponse(result);
    } catch (exception) {
      if (exception.name !== "AbortError") {
        setError(
          `${exception.message || "Cognitive request gagal."}${
            exception.body ? `\n${pretty(exception.body)}` : ""
          }`,
        );
      }
    } finally {
      if (controllerRef.current === controller) {
        controllerRef.current = null;
        setBusy(false);
      }
    }
  }

  function prepareExecution() {
    if (!response) return;

    const normalizedGoal = goal.trim() || observation.trim();
    if (!normalizedGoal) {
      setError("Isi tujuan atau instruksi sebelum menyiapkan eksekusi.");
      return;
    }

    if (typeof onPrepareExecution !== "function") {
      setError("Execution Workspace belum terhubung pada halaman ini.");
      return;
    }

    const plan = response?.result?.plan;

    if (!plan || !Array.isArray(plan.steps) || plan.steps.length === 0) {
      setError("Cognitive Pass belum menghasilkan plan yang valid untuk dieksekusi.");
      return;
    }

    onPrepareExecution({
      goalDescription: normalizedGoal,
      plan,
    });
  }

  function reset() {
    if (busy) return;
    setObservation("");
    setGoal("");
    setError("");
    setResponse(null);
  }

  const result = response?.result || {};
  const plan = result.plan || {};
  const interpretation = result.interpretation || {};
  const reasoning = result.reasoning || {};
  const steps = listOrEmpty(plan.steps);

  return (
    <section className="assistant-composer" id="assistant">
      <div className="assistant-composer-heading">
        <div className="assistant-composer-icon">
          <BrainCircuit size={20} />
        </div>
        <div>
          <p className="assistant-eyebrow">NATURAL-LANGUAGE WORKSPACE</p>
          <h2>Ask hAI Agent</h2>
          <p>
            Jelaskan situasi dan tujuan. Agent akan menjalankan cognitive pass
            dan menyajikan interpretasi serta rencana yang dihasilkan backend.
          </p>
        </div>
      </div>

      <div className="assistant-composer-grid">
        <form className="panel assistant-form" onSubmit={submit}>
          <label className="assistant-field">
            <span>Observasi / instruksi</span>
            <textarea
              value={observation}
              onChange={(event) => setObservation(event.target.value)}
              placeholder="Contoh: Kotak masih tertutup, kunci berada di dalam kotak, dan pintu terkunci."
              rows={5}
              maxLength={10000}
              disabled={busy}
              required
            />
          </label>

          <label className="assistant-field">
            <span>Tujuan (opsional)</span>
            <input
              value={goal}
              onChange={(event) => setGoal(event.target.value)}
              placeholder="Contoh: Buka pintu dengan aman"
              maxLength={1000}
              disabled={busy}
            />
          </label>

          <label className="assistant-trace-toggle">
            <input
              type="checkbox"
              checked={includeTrace}
              onChange={(event) => setIncludeTrace(event.target.checked)}
              disabled={busy}
            />
            <span>
              <b>Sertakan runtime trace</b>
              <small>Meminta trace jika tersedia pada cognitive runtime.</small>
            </span>
          </label>

          {error && (
            <div className="assistant-error" role="alert">
              <AlertCircle size={17} />
              <pre>{error}</pre>
            </div>
          )}

          <div className="assistant-form-actions">
            <button
              type="button"
              className="ghost"
              onClick={reset}
              disabled={busy}
            >
              <RotateCcw size={14} /> Reset
            </button>
            <button type="submit" className="primary" disabled={busy || !observation.trim()}>
              {busy ? (
                <LoaderCircle className="spin" size={15} />
              ) : (
                <ArrowRight size={15} />
              )}
              {busy ? "Menganalisis..." : "Jalankan cognitive pass"}
            </button>
          </div>
          <p className="assistant-disclaimer">
            <ShieldCheck size={14} />
            Cognitive pass tidak mengeksekusi aksi. Eksekusi simulasi tetap
            harus dimulai secara eksplisit dari Execution Workspace.
          </p>
        </form>

        <div className="assistant-output">
          {!response && !busy && (
            <div className="assistant-empty">
              <BrainCircuit size={27} />
              <h3>Hasil akan muncul di sini</h3>
              <p>
                Respons dan rencana akan berasal dari API aktif, bukan data
                contoh yang dibuat oleh antarmuka.
              </p>
            </div>
          )}

          {busy && (
            <div className="assistant-empty">
              <LoaderCircle className="spin" size={26} />
              <h3>Agent sedang menganalisis</h3>
              <p>Menunggu hasil dari cognitive runtime.</p>
            </div>
          )}

          {response && (
            <div className="stack">
              <section className="panel assistant-result-card">
                <div className="panel-head">
                  <h3><CheckCircle2 size={17} /> Cognitive response diterima</h3>
                  <span className="tag">
                    {response.cycle != null ? `Cycle ${response.cycle}` : "Cycle n/a"}
                  </span>
                </div>
                <p className="assistant-result-summary">
                  {interpretation.summary || reasoning.summary || "Backend tidak mengembalikan ringkasan."}
                </p>
                <div className="assistant-result-meta">
                  <span>{steps.length} plan step</span>
                  <span>
                    Confidence:{" "}
                    {typeof plan.confidence === "number"
                      ? `${Math.round(plan.confidence * 100)}%`
                      : "n/a"}
                  </span>
                </div>
              </section>

              <section className="panel">
                <div className="panel-head">
                  <h3>Generated plan</h3>
                  <span className="tag">{steps.length} steps</span>
                </div>
                {plan.rationale && <p className="muted">{plan.rationale}</p>}
                {steps.length ? (
                  <ol className="assistant-plan-list">
                    {steps.map((step, index) => (
                      <li key={`${step.index ?? index}-${step.action ?? "step"}`}>
                        <span className="step-number">{step.index ?? index + 1}</span>
                        <div>
                          <b>{step.action || "Action belum bernama"}</b>
                          {step.purpose && <p>{step.purpose}</p>}
                          {step.success_criteria && (
                            <small className="good">{step.success_criteria}</small>
                          )}
                          {step.risk && <small className="risk">Risk: {String(step.risk)}</small>}
                        </div>
                      </li>
                    ))}
                  </ol>
                ) : (
                  <p className="muted">Plan tidak tersedia atau tidak memiliki langkah.</p>
                )}
                <p className="assistant-disclaimer">
                  Rencana ini belum dieksekusi dan belum membuktikan bahwa goal tercapai.
                </p>
                <div className="assistant-handoff">
                  <button
                    type="button"
                    className="primary"
                    onClick={prepareExecution}
                    disabled={busy || typeof onPrepareExecution !== "function"}
                  >
                    <ArrowDownToLine size={15} />
                    Siapkan goal di Execution Workspace
                  </button>
                  <p className="assistant-disclaimer">
                    Ini hanya meneruskan goal. Target state JSON tetap harus ditinjau
                    secara manual. Execution Workspace akan menjalankan alur Plan &amp; Run
                    tersendiri; rencana cognitive pass ini tidak dieksekusi langsung.
                  </p>
                </div>
              </section>

              <details className="panel assistant-raw">
                <summary>Respons JSON lengkap</summary>
                <pre>{pretty(response)}</pre>
              </details>
            </div>
          )}
        </div>
      </div>
    </section>
  );
}
