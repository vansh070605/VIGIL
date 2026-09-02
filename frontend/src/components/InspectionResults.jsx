export default function InspectionResults({ result, originalImage, onReset }) {
  if (!result) return null;

  const { score, is_anomalous, heatmap_base64 } = result;
  
  const scoreBadgeClass = is_anomalous ? 'score-anomalous' : 'score-normal';
  const statusText = is_anomalous ? 'DEFECT DETECTED' : 'NORMAL';

  return (
    <div className="glass-panel" style={{ animation: 'fadeInUp 0.6s ease-out' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem' }}>
        <h2 style={{ fontSize: '1.8rem', fontWeight: '600' }}>Inspection Results</h2>
        <button onClick={onReset} className="btn-primary" style={{ marginTop: 0, padding: '0.5rem 1rem' }}>
          Inspect Another
        </button>
      </div>

      <div style={{ textAlign: 'center' }}>
        <div className={`score-badge ${scoreBadgeClass}`}>
          <span style={{ marginRight: '0.5rem' }}>{statusText}</span>
          <span style={{ opacity: 0.8, fontSize: '0.9em' }}>| Anomaly Score: {score.toFixed(4)}</span>
        </div>
      </div>

      <div className="results-grid">
        <div className="image-card">
          <h3>Original Image</h3>
          <div className="img-wrapper">
            <img src={originalImage} alt="Original Vial" />
          </div>
        </div>

        <div className="image-card">
          <h3>Anomaly Heatmap Overlay</h3>
          <div className="img-wrapper">
            <img src={heatmap_base64} alt="Anomaly Heatmap" />
          </div>
        </div>
      </div>
    </div>
  );
}
