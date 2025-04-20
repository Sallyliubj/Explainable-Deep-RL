import React from 'react';
import './Results.css';

export default function ResultCard({ result = {}, index }) {
  const {
    recommendation = '',
    q_value = 0,
    explanation = '',
    feature_importance = [],
  } = result;

  return (
    <div className="result-card">
      <h3>Patient {index + 1}</h3>
      <p><strong>Recommendation:</strong> {recommendation} (Q-value: {q_value.toFixed(4)})</p>
      <p><strong>Explanation:</strong> {explanation}</p>

      <div className="bar-chart">
        {feature_importance.map((feat, i) => (
          <div key={i} className="bar-container">
            <div className="bar-label">{feat.feature}</div>
            <div className="bar-fill-wrapper">
              <div className="bar-fill shap" style={{ width: `${(feat.shap * 100).toFixed(1)}%` }}>
                SHAP {feat.shap.toFixed(2)}
              </div>
              <div className="bar-fill attn" style={{ width: `${(feat.attn * 100).toFixed(1)}%` }}>
                ATTN {feat.attn.toFixed(2)}
              </div>
              <div className="bar-fill hybrid" style={{ width: `${(feat.hybrid * 100).toFixed(1)}%` }}>
                HYBRID {feat.hybrid.toFixed(2)}
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}