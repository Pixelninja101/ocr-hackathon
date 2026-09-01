import React, { useState, useEffect } from 'react';
import Header from './components/Header';
import FileUpload from './components/FileUpload';
import CameraScanner from './components/CameraScanner';
import QRVisualizer from './components/QRVisualizer';
import ResultPanel from './components/ResultPanel';
import './App.css';

export default function App() {
  const [file, setFile] = useState(null);
  const [localImageUrl, setLocalImageUrl] = useState(null);
  const [scanResult, setScanResult] = useState(null);
  const [isScanning, setIsScanning] = useState(false);
  const [selectedCodeIndex, setSelectedCodeIndex] = useState(null);
  const [currentPage, setCurrentPage] = useState(1);
  const [isCameraOpen, setIsCameraOpen] = useState(false);
  const [apiError, setApiError] = useState(null);

  // Clean up object URLs to prevent memory leaks
  useEffect(() => {
    return () => {
      if (localImageUrl) {
        URL.revokeObjectURL(localImageUrl);
      }
    };
  }, [localImageUrl]);

  const handleFileSelect = (selectedFile) => {
    if (localImageUrl) {
      URL.revokeObjectURL(localImageUrl);
      setLocalImageUrl(null);
    }

    setFile(selectedFile);
    setCurrentPage(1);
    setSelectedCodeIndex(null);
    setApiError(null);

    // Create local preview object URL for standard images
    const isPdf = selectedFile.type === 'application/pdf' || selectedFile.name.toLowerCase().endsWith('.pdf');
    if (!isPdf) {
      const url = URL.createObjectURL(selectedFile);
      setLocalImageUrl(url);
    }

    executeScan(selectedFile, 0);
  };

  const executeScan = async (targetFile, pageNumber = 0) => {
    setIsScanning(true);
    setApiError(null);

    try {
      const formData = new FormData();
      formData.append('file', targetFile);
      formData.append('page', pageNumber.toString());

      const response = await fetch('/api/qr/scan', {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();

      if (!response.ok && !data.success) {
        setApiError(data.errors?.[0] || `Scan request failed with HTTP ${response.status}`);
        setScanResult(data);
      } else {
        setScanResult(data);
        if (data.codes && data.codes.length > 0) {
          setSelectedCodeIndex(0);
        } else {
          setSelectedCodeIndex(null);
        }
      }
    } catch (err) {
      console.error('API Scan Error:', err);
      setApiError('Unable to connect to the backend QR detection service. Ensure the server is running on http://127.0.0.1:8000.');
    } finally {
      setIsScanning(false);
    }
  };

  const handlePageChange = (newPage) => {
    if (!file || newPage < 1) return;
    setCurrentPage(newPage);
    setSelectedCodeIndex(null);
    executeScan(file, newPage - 1);
  };

  const handleReset = () => {
    if (localImageUrl) {
      URL.revokeObjectURL(localImageUrl);
    }
    setFile(null);
    setLocalImageUrl(null);
    setScanResult(null);
    setSelectedCodeIndex(null);
    setCurrentPage(1);
    setApiError(null);
  };

  // Determine active visual image source:
  // For PDF: use the high-res base64 rendered preview from metadata
  // For standard images: use the local object URL
  const activeImageSrc =
    scanResult?.metadata?.preview_image || localImageUrl || null;

  return (
    <div className="app-layout">
      {/* Navigation Header */}
      <Header onReset={file ? handleReset : null} isScanning={isScanning} />

      {/* Main Content Area */}
      <main className="app-main">
        {!file ? (
          /* Upload State */
          <div className="upload-view-container">
            <FileUpload
              onFileSelect={handleFileSelect}
              onOpenCamera={() => setIsCameraOpen(true)}
              isScanning={isScanning}
            />
          </div>
        ) : (
          /* Scanned & Visualizing State */
          <div className="app-main-grid">
            {/* Left Column: Visualizer & Document Canvas */}
            <div className="panel-column">
              <QRVisualizer
                imageSrc={activeImageSrc}
                codes={scanResult?.codes || []}
                metadata={scanResult?.metadata || {}}
                selectedCodeIndex={selectedCodeIndex}
                onSelectCode={(idx) => setSelectedCodeIndex(idx)}
                currentPage={scanResult?.metadata?.current_page || currentPage}
                totalPdfPages={scanResult?.metadata?.page_count || 1}
                onPageChange={handlePageChange}
                isScanning={isScanning}
              />
            </div>

            {/* Right Column: Decoded Payloads & Analytics */}
            <div className="panel-column">
              <ResultPanel
                scanResult={scanResult}
                selectedCodeIndex={selectedCodeIndex}
                onSelectCode={(idx) => setSelectedCodeIndex(idx)}
              />
            </div>
          </div>
        )}

        {/* Global Connection / API Error Banner */}
        {apiError && !scanResult && (
          <div className="alert-banner alert-error glass-panel mt-3">
            <div className="alert-content">
              <h4 className="alert-title">Connection Error</h4>
              <p className="alert-text">{apiError}</p>
            </div>
          </div>
        )}
      </main>

      {/* Live Camera Scanner Modal */}
      <CameraScanner
        isOpen={isCameraOpen}
        onClose={() => setIsCameraOpen(false)}
        onCapture={handleFileSelect}
      />
    </div>
  );
}
