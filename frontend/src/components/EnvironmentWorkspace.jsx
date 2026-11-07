import { useEffect, useMemo, useState } from "react";
import { AlertCircle, CheckCircle2, CircleHelp, LoaderCircle, RefreshCw, Server, ShieldCheck } from "lucide-react";
import {
  getEnvironmentInstanceCapabilities,
  getEnvironmentInstances,
  getEnvironmentProviders,
} from "../api/client.js";

const asItems = (value) => Array.isArray(value?.items) ? value.items : [];

export default function EnvironmentWorkspace() {
  const [providers, setProviders] = useState([]);
  const [instances, setInstances] = useState([]);
  const [selectedProviderId, setSelectedProviderId] = useState("");
  const [selectedInstanceId, setSelectedInstanceId] = useState("");
  const [capabilityReport, setCapabilityReport] = useState(null);
  const [selectedCapabilityIds, setSelectedCapabilityIds] = useState([]);
  const [loading, setLoading] = useState(true);
  const [capabilitiesLoading, setCapabilitiesLoading] = useState(false);
  const [error, setError] = useState("");
  const [capabilityError, setCapabilityError] = useState("");

  async function refreshDirectory(signal) {
    setLoading(true);
    setError("");
    try {
      const [providerData, instanceData] = await Promise.all([
        getEnvironmentProviders(signal),
        getEnvironmentInstances(signal),
      ]);
      const nextProviders = asItems(providerData);
      const nextInstances = asItems(instanceData).filter(
        (item) => String(item.status).toLowerCase() === "active",
      );
      setProviders(nextProviders);
      setInstances(nextInstances);

      const providerStillExists = nextProviders.some(
        (item) => item.provider_id === selectedProviderId,
      );
      const nextProviderId = providerStillExists
        ? selectedProviderId
        : nextInstances[0]?.provider_id || nextProviders[0]?.provider_id || "";
      const matchingInstances = nextInstances.filter(
        (item) => item.provider_id === nextProviderId,
      );
      const instanceStillExists = matchingInstances.some(
        (item) => item.instance_id === selectedInstanceId,
      );
      setSelectedProviderId(nextProviderId);
      setSelectedInstanceId(
        instanceStillExists ? selectedInstanceId : matchingInstances[0]?.instance_id || "",
      );
    } catch (exception) {
      if (exception.name !== "AbortError") {
        setError(`${exception.message}${exception.body ? `\n${JSON.stringify(exception.body, null, 2)}` : ""}`);
      }
    } finally {
      if (!signal?.aborted) setLoading(false);
    }
  }

  useEffect(() => {
    const controller = new AbortController();
    refreshDirectory(controller.signal);
    return () => controller.abort();
  }, []);

  useEffect(() => {
    if (!selectedInstanceId) {
      setCapabilityReport(null);
      setSelectedCapabilityIds([]);
      setCapabilityError("");
      return undefined;
    }

    const controller = new AbortController();
    setCapabilitiesLoading(true);
    setCapabilityError("");
    setCapabilityReport(null);
    setSelectedCapabilityIds([]);

    getEnvironmentInstanceCapabilities(selectedInstanceId, controller.signal)
      .then((data) => {
        if (!controller.signal.aborted) setCapabilityReport(data);
      })
      .catch((exception) => {
        if (exception.name !== "AbortError" && !controller.signal.aborted) {
          setCapabilityError(`${exception.message}${exception.body ? `\n${JSON.stringify(exception.body, null, 2)}` : ""}`);
        }
      })
      .finally(() => {
        if (!controller.signal.aborted) setCapabilitiesLoading(false);
      });

    return () => controller.abort();
  }, [selectedInstanceId]);

  const providerInstances = useMemo(
    () => instances.filter((item) => item.provider_id === selectedProviderId),
    [instances, selectedProviderId],
  );

  const availableIds = useMemo(
    () => new Set(capabilityReport?.available_capability_ids || []),
    [capabilityReport],
  );

  const candidates = useMemo(() => {
    const counts = new Map();
    for (const candidate of capabilityReport?.candidates || []) {
      if (availableIds.has(candidate.capability_id) && candidate.state === "available") {
        counts.set(candidate.action_name, (counts.get(candidate.action_name) || 0) + 1);
      }
    }
    return (capabilityReport?.candidates || []).map((candidate) => ({
      ...candidate,
      available: availableIds.has(candidate.capability_id) && candidate.state === "available",
      ambiguous: (counts.get(candidate.action_name) || 0) > 1,
    }));
  }, [availableIds, capabilityReport]);

  function toggleCapability(capability) {
    if (!capability.available || capability.ambiguous) return;
    setSelectedCapabilityIds((previous) =>
      previous.includes(capability.capability_id)
        ? previous.filter((id) => id !== capability.capability_id)
        : [...previous, capability.capability_id],
    );
  }

  const selectedProvider = providers.find((item) => item.provider_id === selectedProviderId);

  return (
    <section className="panel environment-workspace" aria-labelledby="environment-workspace-title">
      <div className="panel-head">
        <div>
          <h3 id="environment-workspace-title"><Server size={17} /> Dynamic Environment Workspace</h3>
          <p className="muted">Discovery environment dan capability dari backend tepercaya.</p>
        </div>
        <button type="button" className="ghost" onClick={() => refreshDirectory()} disabled={loading}>
          {loading ? <LoaderCircle className="spin" size={14} /> : <RefreshCw size={14} />}
          Refresh
        </button>
      </div>

      <div className="environment-workspace-notice" role="note">
        <ShieldCheck size={16} />
        <span>Discovery dan pemilihan metadata saja. Panel ini belum mengeksekusi capability atau mengubah authorization backend.</span>
      </div>

      {error && <div className="alert error" role="alert"><AlertCircle size={16} /><pre>{error}</pre></div>}
      {loading && providers.length === 0 && (
        <div className="empty"><LoaderCircle className="spin" size={22} /><h3>Memuat environment</h3><p>Mengambil provider dan instance dari API.</p></div>
      )}

      {!loading && !error && providers.length === 0 && (
        <div className="empty">
          <CircleHelp size={22} />
          <h3>Belum ada environment provider</h3>
          <p>Provider harus didaftarkan oleh aplikasi tepercaya. Frontend tidak dapat mendaftarkan provider.</p>
        </div>
      )}

      {providers.length > 0 && (
        <div className="environment-workspace-grid">
          <label className="execution-field">
            <span>Environment provider</span>
            <select
              value={selectedProviderId}
              onChange={(event) => {
                const nextProviderId = event.target.value;
                setSelectedProviderId(nextProviderId);
                const nextInstance = instances.find((item) => item.provider_id === nextProviderId);
                setSelectedInstanceId(nextInstance?.instance_id || "");
              }}
              disabled={loading}
            >
              {providers.map((provider) => (
                <option key={provider.provider_id} value={provider.provider_id}>
                  {provider.provider_id} · {provider.descriptor?.environment_type || "unknown"}
                </option>
              ))}
            </select>
          </label>

          <label className="execution-field">
            <span>Active environment instance</span>
            <select
              value={selectedInstanceId}
              onChange={(event) => setSelectedInstanceId(event.target.value)}
              disabled={loading || providerInstances.length === 0}
            >
              {providerInstances.length === 0 && <option value="">Tidak ada active instance</option>}
              {providerInstances.map((instance) => (
                <option key={instance.instance_id} value={instance.instance_id}>
                  {instance.instance_id}
                </option>
              ))}
            </select>
          </label>
        </div>
      )}

      {selectedProvider && (
        <div className="environment-workspace-meta">
          <span className="tag">{selectedProvider.descriptor?.mode || "mode unknown"}</span>
          <span className="tag">{selectedProvider.descriptor?.domain || "domain unknown"}</span>
          <span className="tag">{selectedProvider.active_instance_count ?? 0} active instance</span>
        </div>
      )}

      {capabilityError && <div className="alert error" role="alert"><AlertCircle size={16} /><pre>{capabilityError}</pre></div>}
      {capabilitiesLoading && <div className="empty"><LoaderCircle className="spin" size={20} /><p>Memuat capability instance...</p></div>}

      {!capabilitiesLoading && selectedInstanceId && !capabilityError && capabilityReport && (
        <>
          <div className="panel-head environment-capability-heading">
            <div>
              <h4>Discovered capabilities</h4>
              <p className="muted">
                {capabilityReport.environment_type} · registry version {capabilityReport.registry_version}
              </p>
            </div>
            <span className="tag">{selectedCapabilityIds.length} dipilih</span>
          </div>
          {candidates.length > 0 ? (
            <div className="environment-capability-list">
              {candidates.map((candidate) => (
                <label
                  className={`environment-capability ${!candidate.available || candidate.ambiguous ? "is-disabled" : ""}`}
                  key={candidate.capability_id}
                >
                  <input
                    type="checkbox"
                    checked={selectedCapabilityIds.includes(candidate.capability_id)}
                    disabled={!candidate.available || candidate.ambiguous}
                    onChange={() => toggleCapability(candidate)}
                  />
                  <span className="environment-capability-body">
                    <b>{candidate.action_name}</b>
                    <small>{candidate.description || candidate.capability_id}</small>
                    <small>
                      ID: <code>{candidate.capability_id}</code> · Risk: {candidate.risk_level || "unspecified"}
                    </small>
                    <small>Permissions: {(candidate.permissions || []).join(", ") || "none declared"}</small>
                    {!candidate.available && <small className="bad">Tidak tersedia: {candidate.state}</small>}
                    {candidate.ambiguous && <small className="bad">Binding ambigu: lebih dari satu capability memakai action name ini.</small>}
                    {candidate.available && !candidate.ambiguous && <small className="good"><CheckCircle2 size={12} /> Tersedia untuk discovery</small>}
                  </span>
                </label>
              ))}
            </div>
          ) : (
            <div className="empty"><CircleHelp size={20} /><p>Tidak ada kandidat capability untuk instance ini.</p></div>
          )}
          <p className="footnote">
            Pemilihan di sini hanya menyimpan ID capability di UI. Binding dan izin final tetap harus divalidasi backend pada saat integrasi eksekusi.
          </p>
        </>
      )}

      {!capabilitiesLoading && !selectedInstanceId && providers.length > 0 && !error && (
        <div className="empty"><CircleHelp size={20} /><p>Provider ini belum memiliki active instance.</p></div>
      )}
    </section>
  );
}
