import React, { useRef, useState, useEffect } from 'react';
import { ZoomIn, ZoomOut, Maximize2, ChevronLeft, ChevronRight, Eye, ShieldCheck, AlertTriangle } from 'lucide-react';

export default function QRVisualizer({
  imageSrc,
  codes = [],
  metadata = {},
  selectedCodeIndex = null,
  onSelectCode,
  currentPage = 1,
  totalPdfPages = 1,
  onPageChange,
  isScanning = false,
}) {
  const [zoomLevel, setZoomLevel] = useState(1);
  const [naturalDimensions, setNaturalDimensions] = useState({ width: 0, height: 0 });
  const containerRef = useRef(null);
  const imageRef = useRef(null);

  // Read natural image dimensions when loaded
  const handleImageLoad = (e) => {
    const { naturalWidth, naturalHeight } = e.target;
    setNaturalDimensions({
      width: metadata.image_width || naturalWidth,
      height: metadata.image_height || naturalHeight,
    });
  };

  const handleZoomIn = () => setZoomLevel((prev) => Math.min(prev + 0.25, 3.5));
  const handleZoomOut = () => setZoomLevel((prev) => Math.max(prev - 0.25, 0.5));
  const handleResetZoom = () => setZoomLevel(1);

  const viewWidth = metadata.image_width || naturalDimensions.width || 800;
  const viewHeight = metadata.image_height || naturalDimensions.height || 600;

  return (
    <div className="visualizer-container glass-panel">
      {/* Visualizer Top Control Bar */}
      <div className="visualizer-toolbar">
        <div className="toolbar-left">
          <span className="toolbar-label">
            <Eye size={14} /> Document View
          </span>
          {totalPdfPages > 1 && (
            <div className="pdf-page-controls">
              <button
                className="btn-page-nav"
                disabled={currentPage <= 1 || isScanning}
                onClick={() => onPageChange(currentPage - 1)}
                title="Previous Page"
              >
                <ChevronLeft size={14} />
              </button>
              <span className="page-indicator">
                Page {currentPage} of {totalPdfPages}
              </span>
              <button
                className="btn-page-nav"
                disabled={currentPage >= totalPdfPages || isScanning}
                onClick={() => onPageChange(currentPage + 1)}
                title="Next Page"
              >
                <ChevronRight size={14} />
              </button>
            </div>
          )}
        </div>

        <div className="toolbar-zoom-controls">
          <button className="btn-icon-sm" onClick={handleZoomOut} title="Zoom Out">
            <ZoomOut size={14} />
          </button>
          <span className="zoom-label">{Math.round(zoomLevel * 100)}%</span>
          <button className="btn-icon-sm" onClick={handleZoomIn} title="Zoom In">
            <ZoomIn size={14} />
          </button>
          <button className="btn-icon-sm" onClick={handleResetZoom} title="Reset Zoom">
            <Maximize2 size={14} />
          </button>
        </div>
      </div>

      {/* Main Interactive Canvas Area */}
      <div className="visualizer-canvas-viewport" ref={containerRef}>
        <div
          className="canvas-zoom-wrapper"
          style={{ transform: `scale(${zoomLevel})`, transformOrigin: 'top center' }}
        >
          {isScanning && <div className="scanner-beam"></div>}

          {imageSrc && (
            <div className="image-overlay-wrapper">
              <img
                ref={imageRef}
                src={imageSrc}
                alt="Document Preview"
                className="document-image"
                onLoad={handleImageLoad}
              />

              {/* Exact coordinate SVG Overlay */}
              {viewWidth > 0 && viewHeight > 0 && (
                <svg
                  className="qr-svg-overlay"
                  viewBox={`0 0 ${viewWidth} ${viewHeight}`}
                  preserveAspectRatio="none"
                >
                  {codes.map((code, idx) => {
                    const isSelected = selectedCodeIndex === idx;
                    const isDecoded = code.decoded;
                    const pts = code.bbox;
                    if (!pts || pts.length < 4) return null;

                    const pointsStr = pts.map((p) => `${p[0]},${p[1]}`).join(' ');
                    const minX = Math.min(...pts.map((p) => p[0]));
                    const minY = Math.min(...pts.map((p) => p[1]));

                    return (
                      <g
                        key={idx}
                        className={`qr-polygon-group ${isSelected ? 'selected' : ''}`}
                        onClick={() => onSelectCode(idx)}
                      >
                        {/* Glow / Highlight fill & stroke */}
                        <polygon
                          points={pointsStr}
                          className={`qr-bounding-box ${
                            isDecoded ? 'box-decoded' : 'box-undecoded'
                          } ${isSelected ? 'box-active' : ''}`}
                        />

                        {/* Corner markers */}
                        {pts.map((p, pIdx) => (
                          <circle
                            key={pIdx}
                            cx={p[0]}
                            cy={p[1]}
                            r={Math.max(4, Math.min(viewWidth, viewHeight) * 0.006)}
                            className={`qr-corner-dot ${
                              isDecoded ? 'dot-decoded' : 'dot-undecoded'
                            }`}
                          />
                        ))}

                        {/* Bounding box label tag */}
                        <foreignObject
                          x={minX}
                          y={Math.max(0, minY - 32)}
                          width="160"
                          height="32"
                          className="qr-svg-label-object"
                        >
                          <div
                            className={`qr-svg-label ${
                              isDecoded ? 'label-decoded' : 'label-undecoded'
                            } ${isSelected ? 'label-active' : ''}`}
                          >
                            {isDecoded ? (
                              <ShieldCheck size={11} />
                            ) : (
                              <AlertTriangle size={11} />
                            )}
                            <span>QR #{idx + 1} {isDecoded ? 'Decoded' : 'Unreadable'}</span>
                          </div>
                        </foreignObject>
                      </g>
                    );
                  })}
                </svg>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Visualizer Footer Info */}
      <div className="visualizer-footer">
        <span className="footer-meta">
          Resolution: {viewWidth} × {viewHeight} px
        </span>
        <div className="footer-legend">
          <span className="legend-item">
            <span className="legend-dot dot-emerald"></span> Decoded ({codes.filter((c) => c.decoded).length})
          </span>
          <span className="legend-item">
            <span className="legend-dot dot-amber"></span> Undecoded ({codes.filter((c) => !c.decoded).length})
          </span>
        </div>
      </div>
    </div>
  );
}
