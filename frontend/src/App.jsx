import { useState } from 'react';
import ImageUploader from './components/ImageUploader';
import InspectionResults from './components/InspectionResults';
import './index.css';

function App() {
  const [isLoading, setIsLoading] = useState(false);
  const [originalImage, setOriginalImage] = useState(null);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleImageUpload = async (file, previewUrl) => {
    setIsLoading(true);
    setError(null);
    setOriginalImage(previewUrl);
    setResult(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`API error: ${response.statusText}`);
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      console.error("Failed to inspect image:", err);
      setError("Failed to connect to the inspection server. Ensure the backend API is running.");
    } finally {
      setIsLoading(false);
    }
  };

  const handleReset = () => {
    setOriginalImage(null);
    setResult(null);
    setError(null);
  };

  return (
    <>
      <header className="header">
        <h1>VIGIL</h1>
        <p>Visual Inspection & Guided Intelligence Layer</p>
      </header>

      <main>
        {error && (
          <div style={{
            background: 'rgba(239, 68, 68, 0.1)',
            border: '1px solid var(--danger)',
            color: '#fca5a5',
            padding: '1rem',
            borderRadius: '8px',
            marginBottom: '2rem',
            textAlign: 'center'
          }}>
            {error}
          </div>
        )}

        {!result ? (
          <ImageUploader onImageUpload={handleImageUpload} isLoading={isLoading} />
        ) : (
          <InspectionResults result={result} originalImage={originalImage} onReset={handleReset} />
        )}
      </main>
    </>
  );
}

export default App;
