import React, { useEffect, useState } from 'react';
import './index.css';

function App() {
  const [results, setResults] = useState(null);

  useEffect(() => {
    fetch('http://localhost:8000/experiments/patchcore_baseline/results.json')
      .then(res => res.json())
      .then(data => setResults(data))
      .catch(err => console.error("Error fetching results:", err));
  }, []);

  return (
    <div className="app-container">
      <header className="header">
        <h1>VIGIL</h1>
        <p>Anomaly Detection Dashboard</p>
      </header>

      {results ? (
        <>
          <div className="dashboard-grid">
            <div className="glass-panel metric-card">
              <h3>Image AUROC</h3>
              <div className="value">{(results.image_auroc * 100).toFixed(2)}<span>%</span></div>
            </div>
            <div className="glass-panel metric-card">
              <h3>Pixel AUROC</h3>
              <div className="value">{(results.pixel_auroc * 100).toFixed(2)}<span>%</span></div>
            </div>
            <div className="glass-panel metric-card">
              <h3>PRO (Per-Region Overlap)</h3>
              <div className="value">{(results.pixel_pro * 100).toFixed(2)}<span>%</span></div>
            </div>
          </div>

          <section className="gallery-section glass-panel">
            <h2>Visualizations & Analytics</h2>
            <div className="gallery-grid">
              {[
                { id: 'roc_curve_image', title: 'ROC Curve' },
                { id: 'score_distribution', title: 'Score Distribution' },
                { id: 'per_variant_auroc', title: 'AUROC per Variant' },
                { id: 'anomaly_maps_bad', title: 'Sample Anomaly Maps' }
              ].map((img) => (
                <div key={img.id} className="gallery-item">
                  <img 
                    src={`http://localhost:8000/experiments/patchcore_baseline/visualizations/${img.id}.png`} 
                    alt={img.title} 
                    onError={(e) => { e.target.style.display = 'none'; }}
                  />
                  <div className="label">{img.title}</div>
                </div>
              ))}
            </div>
          </section>
        </>
      ) : (
        <div style={{textAlign: 'center', marginTop: '50px', fontSize: '1.2rem', color: '#94a3b8'}}>Loading results...</div>
      )}
    </div>
  );
}

export default App;
