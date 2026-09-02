import { useState, useRef } from 'react';

export default function ImageUploader({ onImageUpload, isLoading }) {
  const [dragActive, setDragActive] = useState(false);
  const inputRef = useRef(null);

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  };

  const handleChange = (e) => {
    e.preventDefault();
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0]);
    }
  };

  const handleFile = (file) => {
    // Only accept images
    if (!file.type.match('image.*')) {
      alert("Please upload an image file.");
      return;
    }
    
    // Create preview URL
    const previewUrl = URL.createObjectURL(file);
    onImageUpload(file, previewUrl);
  };

  return (
    <div className="glass-panel" style={{ maxWidth: '600px', margin: '0 auto' }}>
      <div 
        className={`uploader-container ${dragActive ? 'drag-active' : ''}`}
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
        onClick={() => !isLoading && inputRef.current.click()}
      >
        <input 
          ref={inputRef}
          type="file" 
          accept="image/*" 
          onChange={handleChange} 
          style={{ display: 'none' }} 
        />
        
        <div className="uploader-icon">
          <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
            <polyline points="17 8 12 3 7 8"></polyline>
            <line x1="12" y1="3" x2="12" y2="15"></line>
          </svg>
        </div>
        
        <h3 style={{ fontSize: '1.5rem', marginBottom: '0.5rem' }}>Upload Vial Image</h3>
        <p style={{ color: 'var(--text-secondary)', marginBottom: '1.5rem' }}>
          Drag and drop or click to select an image from the MVTec dataset
        </p>
        
        <button className="btn-primary" disabled={isLoading} onClick={(e) => {
          e.stopPropagation();
          inputRef.current.click();
        }}>
          {isLoading ? (
            <><span className="loader"></span> Processing...</>
          ) : (
            'Select Image'
          )}
        </button>
      </div>
    </div>
  );
}
