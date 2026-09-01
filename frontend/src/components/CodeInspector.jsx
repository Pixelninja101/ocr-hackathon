import React, { useState } from "react";
import {
  Copy, Check, Download, ShieldCheck, AlertTriangle,
  Lock, Link, Wifi, User, CreditCard, MessageSquare,
  MapPin, Code2, Hash, FileText, Mail, Phone, AlertCircle,
  ChevronDown, ChevronUp, Eye, EyeOff, Layers
} from "lucide-react";
import { parseQrPayload } from "../utils/qrPayloadParser";

// ─────────────────────────────────────────────────────────────────────────────
// Icon map for parsed type
// ─────────────────────────────────────────────────────────────────────────────
const TYPE_ICONS = {
  shield: ShieldCheck,
  lock: Lock,
  link: Link,
  wifi: Wifi,
  contact: User,
  payment: CreditCard,
  sms: MessageSquare,
  email: Mail,
  phone: Phone,
  geo: MapPin,
  code: Code2,
  hash: Hash,
  text: FileText,
  warning: AlertCircle,
};

const TYPE_COLORS = {
  aadhaar_old: "type-aadhaar-old",
  aadhaar_secure: "type-aadhaar-secure",
  url: "type-url",
  upi: "type-upi",
  wifi: "type-wifi",
  vcard: "type-contact",
  mecard: "type-contact",
  email: "type-email",
  phone: "type-phone",
  sms: "type-sms",
  geo: "type-geo",
  json: "type-code",
  xml: "type-code",
  numeric: "type-numeric",
  text: "type-text",
  empty: "type-text",
};

// ─────────────────────────────────────────────────────────────────────────────
// Parsed Content Renderer
// ─────────────────────────────────────────────────────────────────────────────
function ParsedContent({ parsed, data }) {
  const [showSensitive, setShowSensitive] = useState({});
  const IconComponent = TYPE_ICONS[parsed.icon] || FileText;
  const colorClass = TYPE_COLORS[parsed.type] || "type-text";

  const toggleSensitive = (key) =>
    setShowSensitive((prev) => ({ ...prev, [key]: !prev[key] }));

  return (
    <div className={`parsed-content ${colorClass}`}>
      {/* Type Header */}
      <div className="parsed-type-header">
        <div className="parsed-type-icon-wrap">
          <IconComponent size={18} />
        </div>
        <span className="parsed-type-label">{parsed.label}</span>
      </div>

      {/* Encrypted / Advisory Notice */}
      {parsed.notice && (
        <div className="parsed-notice">
          <Lock size={13} className="parsed-notice-icon" />
          <p>{parsed.notice}</p>
        </div>
      )}

      {/* Map Link for geo */}
      {parsed.mapUrl && (
        <a
          href={parsed.mapUrl}
          target="_blank"
          rel="noopener noreferrer"
          className="parsed-map-link"
        >
          <MapPin size={12} /> View on OpenStreetMap
        </a>
      )}

      {/* URL quick-launch */}
      {parsed.type === "url" && data && (
        <a
          href={data.trim()}
          target="_blank"
          rel="noopener noreferrer"
          className="parsed-map-link"
        >
          <Link size={12} /> Open URL
        </a>
      )}

      {/* Fields Table */}
      {parsed.fields.length > 0 && (
        <div className="parsed-fields-table">
          {parsed.fields.map((field, i) => (
            <div key={i} className="parsed-field-row">
              <span className="parsed-field-key">{field.key}</span>
              <span className="parsed-field-value">
                {field.sensitive ? (
                  <span className="sensitive-value-wrap">
                    <span className={showSensitive[field.key] ? "" : "sensitive-masked"}>
                      {showSensitive[field.key] ? field.value : "••••••••••••"}
                    </span>
                    <button
                      className="btn-toggle-sensitive"
                      onClick={(e) => { e.stopPropagation(); toggleSensitive(field.key); }}
                      title={showSensitive[field.key] ? "Hide" : "Reveal"}
                    >
                      {showSensitive[field.key] ? <EyeOff size={11} /> : <Eye size={11} />}
                    </button>
                  </span>
                ) : (
                  field.value
                )}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Main CodeInspector
// ─────────────────────────────────────────────────────────────────────────────
export default function CodeInspector({ code, index, isSelected, onSelect }) {
  const [copied, setCopied] = useState(false);
  const [rawOpen, setRawOpen] = useState(false);
  const [viewMode, setViewMode] = useState("raw");

  const {
    id = index + 1,
    decoded,
    data,
    bbox,
    center,
    width,
    height,
    area,
    decode_method,
    attempts = 1,
  } = code;

  const parsed = decoded && data ? parseQrPayload(data) : null;

  const handleCopy = () => {
    if (!data) return;
    navigator.clipboard.writeText(data);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleDownload = () => {
    if (!data) return;
    const blob = new Blob([data], { type: "text/plain;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `qr_payload_${id}.txt`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const getHexView = (str) => {
    if (!str) return "";
    const bytes = new TextEncoder().encode(str);
    let hexStr = "";
    for (let i = 0; i < bytes.length; i++) {
      hexStr += bytes[i].toString(16).padStart(2, "0").toUpperCase() + " ";
      if ((i + 1) % 16 === 0) hexStr += "\n";
      else if ((i + 1) % 8 === 0) hexStr += "  ";
    }
    return hexStr;
  };

  const charCount = data ? data.length : 0;
  const byteCount = data ? new TextEncoder().encode(data).length : 0;

  // Badge label: use parsed type label if available, else legacy detection
  const badgeLabel = parsed ? parsed.label : (() => {
    if (!data) return "Undecoded";
    const t = data.trim();
    if (t.startsWith("http")) return "URL";
    if (t.startsWith("{")) return "JSON";
    if (t.startsWith("<")) return "XML Data";
    if (/^\d+$/.test(t)) return "Numeric / Compressed";
    return "Text Payload";
  })();

  return (
    <div
      className={`code-inspector-card ${isSelected ? "card-active" : ""} ${decoded ? "card-decoded" : "card-undecoded"}`}
      onClick={onSelect}
      id={`qr-card-${index}`}
    >
      {/* Card Header */}
      <div className="card-header">
        <div className="card-title-group">
          <span className="qr-index-badge">#{id}</span>
          {decoded ? (
            <span className="badge badge-emerald">
              <ShieldCheck size={12} /> Decoded
            </span>
          ) : (
            <span className="badge badge-amber">
              <AlertTriangle size={12} /> Undecodable
            </span>
          )}
          <span className="badge badge-cyan">{badgeLabel}</span>
          {decode_method && (
            <span className="badge badge-subtle" title={`Decoded via ${decode_method}`}>
              <Layers size={11} /> {decode_method.replace(/_/g, " ")}
            </span>
          )}
        </div>

        {decoded && (
          <div className="card-actions" onClick={(e) => e.stopPropagation()}>
            <button
              onClick={handleCopy}
              className={`btn-action ${copied ? "btn-copied" : ""}`}
              title="Copy QR payload to clipboard"
            >
              {copied ? <Check size={13} /> : <Copy size={13} />}
              <span>{copied ? "Copied" : "Copy"}</span>
            </button>
            <button onClick={handleDownload} className="btn-action" title="Download payload as file">
              <Download size={13} />
            </button>
          </div>
        )}
      </div>

      {/* Payload Display Area */}
      <div className="payload-container">
        {decoded ? (
          <>
            {/* ── Parsed / Interpreted View ── */}
            {parsed && (
              <ParsedContent parsed={parsed} data={data} />
            )}

            {/* ── Collapsible Raw Data Section ── */}
            <details
              className="raw-data-collapsible"
              open={rawOpen}
              onToggle={(e) => setRawOpen(e.target.open)}
              onClick={(e) => e.stopPropagation()}
            >
              <summary className="raw-data-summary">
                <span className="raw-data-summary-label">
                  {rawOpen ? <ChevronUp size={13} /> : <ChevronDown size={13} />}
                  Raw Data
                </span>
                <span className="payload-stats-inline">
                  {charCount} chars &bull; {byteCount} bytes &bull; {attempts} attempt{attempts > 1 ? "s" : ""}
                </span>
              </summary>

              <div className="raw-data-body">
                {/* View Mode Tabs */}
                <div className="payload-toolbar" onClick={(e) => e.stopPropagation()}>
                  <div className="view-mode-tabs">
                    <button className={`tab-btn ${viewMode === "raw" ? "tab-active" : ""}`} onClick={() => setViewMode("raw")}>
                      Raw Text
                    </button>
                    <button className={`tab-btn ${viewMode === "hex" ? "tab-active" : ""}`} onClick={() => setViewMode("hex")}>
                      Hex Bytes
                    </button>
                  </div>
                </div>
                <div className="payload-content-box">
                  {viewMode === "raw" ? (
                    <pre className="payload-text mono-text">{data}</pre>
                  ) : (
                    <pre className="payload-text hex-view mono-text">{getHexView(data)}</pre>
                  )}
                </div>
              </div>
            </details>
          </>
        ) : (
          <div className="undecoded-notice">
            <AlertTriangle size={18} className="text-amber" />
            <div>
              <p className="undecoded-title">QR Code Detected ({attempts} Multi-Stage Attempts Evaluated)</p>
              <p className="undecoded-desc">
                OpenCV identified standard finder patterns at this coordinate, but error correction could not decode data through direct, perspective-rectified, or thresholded variants.
              </p>
            </div>
          </div>
        )}
      </div>

      {/* Geometric Bounding Box Properties */}
      <div className="geometry-section">
        <div className="geometry-grid">
          <div className="geo-item">
            <span className="geo-label">Center:</span>
            <span className="geo-val mono-text">
              ({center ? `${center[0]}, ${center[1]}` : "N/A"})
            </span>
          </div>
          <div className="geo-item">
            <span className="geo-label">Dimensions:</span>
            <span className="geo-val mono-text">
              {width ? `${Math.round(width)} × ${Math.round(height)} px` : "N/A"}
            </span>
          </div>
          <div className="geo-item">
            <span className="geo-label">Area:</span>
            <span className="geo-val mono-text">
              {area ? `${Math.round(area)} px²` : "N/A"}
            </span>
          </div>
        </div>

        {bbox && bbox.length === 4 && (
          <details className="bbox-details" onClick={(e) => e.stopPropagation()}>
            <summary className="bbox-summary">
              <span>Polygon Coordinates (4 Vertices)</span>
            </summary>
            <div className="bbox-points-list mono-text">
              {bbox.map((pt, pIdx) => (
                <span key={pIdx} className="point-chip">
                  P{pIdx + 1}: [{pt[0]}, {pt[1]}]
                </span>
              ))}
            </div>
          </details>
        )}
      </div>
    </div>
  );
}
