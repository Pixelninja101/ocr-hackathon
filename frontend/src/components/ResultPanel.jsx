import React from 'react';
import { ShieldCheck, AlertCircle, AlertTriangle, QrCode, FileText, CheckCircle2, Copy } from 'lucide-react';
import CodeInspector from './CodeInspector';

export default function ResultPanel({
  scanResult,
  selectedCodeIndex,
  onSelectCode,
}) {
  if (!scanResult) return null;

  const { success, qr_detected, qr_count, codes = [], warnings = [], errors = [], metadata = {} } = scanResult;

  const decodedCount = codes.filter((c) => c.decoded).length;
  const undecodedCount = codes.filter((c) => !c.decoded).length;

  return (
    <div className="results-panel">
      {/* Overview Statistics Header */}
      <div className="results-summary-card glass-panel">
        <div className="summary-main">
          <div className="summary-stat-group">
            <div className="stat-icon-box">
              <QrCode size={22} className={qr_detected ? 'text-emerald' : 'text-muted'} />
            </div>
            <div>
              <h2 className="summary-title">
                {qr_detected
                  ? `${qr_count} QR Code${qr_count > 1 ? 's' : ''} Detected`
                  : 'No QR Code Detected'}
              </h2>
              <p className="summary-sub">
                {qr_detected
                  ? `${decodedCount} successfully decoded${undecodedCount > 0 ? `, ${undecodedCount} undecodable` : ''}`
                  : 'Document analyzed with OpenCV multi-stage detector'}
              </p>
            </div>
          </div>

          <div className="summary-badges">
            {qr_detected ? (
              decodedCount === qr_count ? (
                <span className="badge badge-emerald">
                  <CheckCircle2 size={13} /> Complete Scan
                </span>
              ) : decodedCount > 0 ? (
                <span className="badge badge-amber">
                  <AlertTriangle size={13} /> Partial Decode
                </span>
              ) : (
                <span className="badge badge-rose">
                  <AlertCircle size={13} /> Decode Failed
                </span>
              )
            ) : (
              <span className="badge badge-amber">
                <AlertCircle size={13} /> No Target Found
              </span>
            )}
          </div>
        </div>

        {/* Document & Engine Metadata Strip */}
        <div className="metadata-strip">
          <div className="meta-cell">
            <span className="meta-cell-label">File:</span>
            <span className="meta-cell-val" title={metadata.filename}>
              {metadata.filename || 'Document'}
            </span>
          </div>
          <div className="meta-cell">
            <span className="meta-cell-label">Type:</span>
            <span className="meta-cell-val uppercase">{metadata.file_type || 'image'}</span>
          </div>
          {metadata.page_count > 1 && (
            <div className="meta-cell">
              <span className="meta-cell-label">Page:</span>
              <span className="meta-cell-val">
                {metadata.current_page} of {metadata.page_count}
              </span>
            </div>
          )}
          <div className="meta-cell">
            <span className="meta-cell-label">Engine:</span>
            <span className="meta-cell-val">OpenCV QRCodeDetector</span>
          </div>
        </div>
      </div>

      {/* Warnings & Errors Banners */}
      {errors && errors.length > 0 && (
        <div className="alert-banner alert-error glass-panel">
          <AlertCircle size={18} className="alert-icon" />
          <div className="alert-content">
            <h4 className="alert-title">Processing Error</h4>
            {errors.map((err, i) => (
              <p key={i} className="alert-text">{err}</p>
            ))}
          </div>
        </div>
      )}

      {warnings && warnings.length > 0 && !errors.length && (
        <div className="alert-banner alert-warning glass-panel">
          <AlertTriangle size={18} className="alert-icon" />
          <div className="alert-content">
            <h4 className="alert-title">Detection Notice</h4>
            {warnings.map((w, i) => (
              <p key={i} className="alert-text">{w}</p>
            ))}
          </div>
        </div>
      )}

      {/* Codes List */}
      {qr_detected && codes.length > 0 ? (
        <div className="codes-list-section">
          <h3 className="section-heading">Detected QR Payloads & Coordinates</h3>
          <div className="codes-cards-list">
            {codes.map((code, idx) => (
              <CodeInspector
                key={idx}
                code={code}
                index={idx}
                isSelected={selectedCodeIndex === idx}
                onSelect={() => onSelectCode(idx)}
              />
            ))}
          </div>
        </div>
      ) : !errors.length && (
        <div className="no-qr-card glass-panel">
          <QrCode size={36} className="no-qr-icon" />
          <h3 className="no-qr-title">No QR Code Found in Document</h3>
          <p className="no-qr-desc">
            The OpenCV detection engine evaluated the document using standard multi-QR finder patterns and adaptive contrast equalization. No valid QR markers were identified.
          </p>
        </div>
      )}
    </div>
  );
}
