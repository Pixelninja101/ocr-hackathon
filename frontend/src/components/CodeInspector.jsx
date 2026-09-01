import React, { useState } from 'react';
import { Copy, Check, Download, ShieldCheck, AlertTriangle, Cpu, Layers } from 'lucide-react';

export default function CodeInspector({ code, index, isSelected, onSelect }) {
  const [copied, setCopied] = useState(false);
  const [viewMode, setViewMode] = useState('raw'); // 'raw' | 'hex'

  const { id = index + 1, decoded, data, bbox, center, width, height, area, decode_method, attempts = 1 } = code;

  const handleCopy = () => {
    if (!data) return;
    navigator.clipboard.writeText(data);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleDownload = () => {
    if (!data) return;
    const blob = new Blob([data], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `qr_payload_${id}.txt`;
    a.click();
    URL.revokeObjectURL(url);
  };

  // Helper to detect content type
  const detectContentType = (str) => {
    if (!str) return 'Empty';
    const trimmed = str.trim();
    if (trimmed.startsWith('http://') || trimmed.startsWith('https://')) return 'URL';
    if (trimmed.startsWith('{') && trimmed.endsWith('}')) return 'JSON';
    if (trimmed.startsWith('<?xml') || trimmed.startsWith('<')) return 'XML Data';
    if (/^\d+$/.test(trimmed)) return 'Numeric / Compressed';
    return 'Text Payload';
  };

  const contentType = decoded ? detectContentType(data) : 'Undecoded';
  const charCount = data ? data.length : 0;
  const byteCount = data ? new TextEncoder().encode(data).length : 0;

  // Convert string to hex representation for binary inspection
  const getHexView = (str) => {
    if (!str) return '';
    const bytes = new TextEncoder().encode(str);
    let hexStr = '';
    for (let i = 0; i < bytes.length; i++) {
      hexStr += bytes[i].toString(16).padStart(2, '0').toUpperCase() + ' ';
      if ((i + 1) % 16 === 0) hexStr += '\n';
      else if ((i + 1) % 8 === 0) hexStr += '  ';
    }
    return hexStr;
  };

  return (
    <div
      className={`code-inspector-card ${
        isSelected ? 'card-active' : ''
      } ${decoded ? 'card-decoded' : 'card-undecoded'}`}
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
          <span className="badge badge-cyan">{contentType}</span>
          {decode_method && (
            <span className="badge badge-subtle" title={`Decoded via ${decode_method}`}>
              <Layers size={11} /> {decode_method.replace(/_/g, ' ')}
            </span>
          )}
        </div>

        {decoded && (
          <div className="card-actions" onClick={(e) => e.stopPropagation()}>
            <button
              onClick={handleCopy}
              className={`btn-action ${copied ? 'btn-copied' : ''}`}
              title="Copy QR payload to clipboard"
            >
              {copied ? <Check size={13} /> : <Copy size={13} />}
              <span>{copied ? 'Copied' : 'Copy'}</span>
            </button>
            <button
              onClick={handleDownload}
              className="btn-action"
              title="Download payload as file"
            >
              <Download size={13} />
            </button>
          </div>
        )}
      </div>

      {/* Payload Display Area */}
      <div className="payload-container">
        {decoded ? (
          <>
            <div className="payload-toolbar">
              <div className="view-mode-tabs" onClick={(e) => e.stopPropagation()}>
                <button
                  className={`tab-btn ${viewMode === 'raw' ? 'tab-active' : ''}`}
                  onClick={() => setViewMode('raw')}
                >
                  Raw Text
                </button>
                <button
                  className={`tab-btn ${viewMode === 'hex' ? 'tab-active' : ''}`}
                  onClick={() => setViewMode('hex')}
                >
                  Hex Bytes
                </button>
              </div>

              <div className="payload-stats">
                <span>{charCount} chars</span>
                <span>•</span>
                <span>{byteCount} bytes</span>
                <span>•</span>
                <span>{attempts} attempt{attempts > 1 ? 's' : ''}</span>
              </div>
            </div>

            <div className="payload-content-box">
              {viewMode === 'raw' ? (
                <pre className="payload-text mono-text">{data}</pre>
              ) : (
                <pre className="payload-text hex-view mono-text">{getHexView(data)}</pre>
              )}
            </div>
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
              ({center ? `${center[0]}, ${center[1]}` : 'N/A'})
            </span>
          </div>
          <div className="geo-item">
            <span className="geo-label">Dimensions:</span>
            <span className="geo-val mono-text">
              {width ? `${Math.round(width)} × ${Math.round(height)} px` : 'N/A'}
            </span>
          </div>
          <div className="geo-item">
            <span className="geo-label">Area:</span>
            <span className="geo-val mono-text">
              {area ? `${Math.round(area)} px²` : 'N/A'}
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
