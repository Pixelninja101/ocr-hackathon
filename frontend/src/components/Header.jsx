import React, { useEffect, useState } from 'react';
import { QrCode, Shield, Activity, RefreshCw } from 'lucide-react';

export default function Header({ onReset, isScanning }) {
  const [backendHealth, setBackendHealth] = useState({ status: 'checking', engine: '' });

  const checkHealth = async () => {
    try {
      const res = await fetch('/api/health');
      if (res.ok) {
        const data = await res.json();
        setBackendHealth({ status: 'connected', engine: data.engine || 'OpenCV QRCodeDetector' });
      } else {
        setBackendHealth({ status: 'disconnected', engine: '' });
      }
    } catch {
      setBackendHealth({ status: 'disconnected', engine: '' });
    }
  };

  useEffect(() => {
    checkHealth();
    const interval = setInterval(checkHealth, 10000);
    return () => clearInterval(interval);
  }, []);

  return (
    <header className="header-container glass-panel">
      <div className="header-brand">
        <div className="brand-icon-wrapper">
          <QrCode className="brand-icon" size={26} />
        </div>
        <div className="brand-text">
          <div className="brand-title-row">
            <h1 className="brand-title">QR Detection & Decoding</h1>
            <span className="badge badge-cyan">Standalone Core</span>
          </div>
          <p className="brand-subtitle">OpenCV & PyMuPDF In-Memory Document Analysis</p>
        </div>
      </div>

      <div className="header-actions">
        {/* Backend status indicator */}
        <div className="status-indicator">
          {backendHealth.status === 'connected' ? (
            <div className="status-pill status-online">
              <span className="pulse-dot"></span>
              <span className="status-text">Engine Ready: {backendHealth.engine}</span>
            </div>
          ) : backendHealth.status === 'checking' ? (
            <div className="status-pill status-checking">
              <Activity size={14} className="spin" />
              <span className="status-text">Connecting to Backend...</span>
            </div>
          ) : (
            <div className="status-pill status-offline">
              <span className="status-dot-offline"></span>
              <span className="status-text">Backend Offline</span>
            </div>
          )}
        </div>

        {onReset && (
          <button
            onClick={onReset}
            className="btn btn-secondary"
            title="Reset scanner and load a new document"
            disabled={isScanning}
            id="reset-scan-btn"
          >
            <RefreshCw size={15} className={isScanning ? 'spin' : ''} />
            <span>New Scan</span>
          </button>
        )}
      </div>
    </header>
  );
}
