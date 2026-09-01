import React, { useRef, useState } from 'react';
import { UploadCloud, FileText, Image as ImageIcon, Camera, Sparkles, AlertCircle } from 'lucide-react';

const MAX_FILE_SIZE_MB = 15;
const ACCEPTED_TYPES = ['.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff', '.tif', '.pdf'];

export default function FileUpload({ onFileSelect, onOpenCamera, isScanning }) {
  const [isDragOver, setIsDragOver] = useState(false);
  const [validationError, setValidationError] = useState(null);
  const fileInputRef = useRef(null);

  const handleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragOver(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragOver(false);
  };

  const validateAndPassFile = (file) => {
    setValidationError(null);
    if (!file) return;

    // Check size limit
    if (file.size > MAX_FILE_SIZE_MB * 1024 * 1024) {
      setValidationError(`File size (${(file.size / (1024 * 1024)).toFixed(2)} MB) exceeds ${MAX_FILE_SIZE_MB} MB limit.`);
      return;
    }

    // Check file extension
    const ext = '.' + file.name.split('.').pop().toLowerCase();
    if (!ACCEPTED_TYPES.includes(ext)) {
      setValidationError(`Unsupported file type '${ext}'. Please upload an image or PDF file.`);
      return;
    }

    onFileSelect(file);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragOver(false);

    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      validateAndPassFile(e.dataTransfer.files[0]);
    }
  };

  const handleFileInputChange = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      validateAndPassFile(e.target.files[0]);
    }
  };

  return (
    <div className="upload-section">
      <div
        className={`dropzone glass-panel ${isDragOver ? 'dropzone-active' : ''} ${isScanning ? 'dropzone-scanning' : ''}`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => !isScanning && fileInputRef.current?.click()}
        id="file-dropzone"
      >
        <input
          ref={fileInputRef}
          type="file"
          accept={ACCEPTED_TYPES.join(',')}
          onChange={handleFileInputChange}
          style={{ display: 'none' }}
          disabled={isScanning}
          id="file-input-hidden"
        />

        <div className="dropzone-icon-container">
          <div className="icon-pulse-ring"></div>
          <UploadCloud className="dropzone-icon" size={44} />
        </div>

        <div className="dropzone-text-block">
          <h3 className="dropzone-title">
            {isScanning ? 'Processing Document...' : 'Upload Image or PDF Document'}
          </h3>
          <p className="dropzone-subtitle">
            Drag & drop document here, or <span className="text-highlight">browse files</span>
          </p>
        </div>

        <div className="dropzone-specs">
          <span className="spec-badge">
            <ImageIcon size={13} /> PNG, JPG, WEBP, BMP, TIFF
          </span>
          <span className="spec-badge">
            <FileText size={13} /> PDF Documents
          </span>
          <span className="spec-badge">Max {MAX_FILE_SIZE_MB}MB</span>
        </div>

        {validationError && (
          <div className="validation-error-banner" onClick={(e) => e.stopPropagation()}>
            <AlertCircle size={16} />
            <span>{validationError}</span>
          </div>
        )}
      </div>

      <div className="upload-actions-bar">
        <button
          type="button"
          onClick={onOpenCamera}
          className="btn btn-camera"
          disabled={isScanning}
          id="open-camera-btn"
        >
          <Camera size={16} />
          <span>Scan with Device Camera</span>
        </button>
      </div>
    </div>
  );
}
