import { useState } from 'react';
import { Check, Copy } from 'lucide-react';
export default function JsonPanel({ title, value }) {
  const [copied, setCopied] = useState(false);
  if (value == null) return <section className="panel"><h3>{title}</h3><p className="muted">Belum ada data.</p></section>;
  const json = JSON.stringify(value, null, 2);
  async function copy() { try { await navigator.clipboard.writeText(json); setCopied(true); setTimeout(() => setCopied(false), 1200); } catch {} }
  return <section className="panel"><div className="panel-head"><h3>{title}</h3><button className="ghost" onClick={copy}>{copied ? <Check size={14}/> : <Copy size={14}/>} {copied ? 'Tersalin' : 'Salin'}</button></div><pre className="json-view">{json}</pre></section>;
}
