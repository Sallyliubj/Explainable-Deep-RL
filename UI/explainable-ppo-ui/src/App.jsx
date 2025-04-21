// src/App.jsx
import React, { useState } from 'react';
import './App.css';
import ResultsList from './components/ResultsList';


const defaultPatient = {
  gender: 'M',
  age: 65,
  wbc: 10,
  lactate: 1.5,
  creatinine: 1,
  los: 5.0,
  expire_flag: 0,
  has_staph: false,
  has_strep: false,
  has_e_coli: false,
  has_pseudomonas: false,
  has_klebsiella: false,
  pneumonia: false,
  uti: false,
  sepsis: false,
  skin_soft_tissue: false,
  intra_abdominal: false,
  meningitis: false
};



function App() {
  const [patients, setPatients] = useState([defaultPatient]);
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [expandedStates, setExpandedStates] = useState([true]);

  const handleChange = (index, field, value) => {
    const updated = [...patients];
    updated[index][field] = value;
    setPatients(updated);
  };

  const handleCheckbox = (index, field) => {
    const updated = [...patients];
    updated[index][field] = !updated[index][field];
    setPatients(updated);
  };

  const addPatient = () => setPatients([...patients, { ...defaultPatient }]);

  const removePatient = (index) => {
    const updated = [...patients];
    updated.splice(index, 1);
    setPatients(updated);
  };

  const toggleExpanded = (index) => {
    const updated = [...expandedStates];
    updated[index] = !updated[index];
    setExpandedStates(updated);
  };


  const handleSubmit = async () => {
    setLoading(true);
    try {
      const res = await fetch('http://localhost:5001/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ patients }),
      });
      const data = await res.json();
      setResults(data.results);
    } catch (err) {
      console.error('Error:', err);
    }
    setLoading(false);
  };

  return (
      <div className="App">
        <h1>Antibiotic Recommendation Interface</h1>
        {patients.map((p, i) => {
            const expanded = expandedStates[i];
            return (
            <div key={i} className="patient-form">
              <h2>
                Patient {i + 1}
                {patients.length > 1 && (
                  <button
                    onClick={() => removePatient(i)}
                    style={{
                      marginLeft: '1rem',
                      padding: '0.25rem 0.5rem',
                      backgroundColor: '#e74c3c',
                      color: 'white',
                      border: 'none',
                      borderRadius: '4px',
                      cursor: 'pointer'
                    }}
                  >
                    Delete
                  </button>
                )}
              </h2>
              <div>
                <label>Gender:
                  <select value={p.gender} onChange={e => handleChange(i, 'gender', e.target.value)}>
                    <option value="M">Male</option>
                    <option value="F">Female</option>
                  </select>
                </label>
                <label>Age:
                  <input type="number" value={p.age}
                         onChange={e => handleChange(i, 'age', parseFloat(e.target.value))}/>
                </label>
                <label>Length of Stay (LOS):
                  <input type="number" step="0.1" value={p.los}
                         onChange={e => handleChange(i, 'los', parseFloat(e.target.value))}/>
                </label>
                <label>Expired in Hospital:
                  <select value={p.expire_flag}
                          onChange={e => handleChange(i, 'expire_flag', parseInt(e.target.value))}>
                    <option value={0}>No</option>
                    <option value={1}>Yes</option>
                  </select>
                </label>
                <label>WBC:
                  <input type="number" value={p.wbc}
                         onChange={e => handleChange(i, 'wbc', parseFloat(e.target.value))}/>
                </label>
                <label>Lactate:
                  <input type="number" step="0.1" value={p.lactate}
                         onChange={e => handleChange(i, 'lactate', parseFloat(e.target.value))}/>
                </label>
                <label>Creatinine:
                  <input type="number" step="0.1" value={p.creatinine}
                         onChange={e => handleChange(i, 'creatinine', parseFloat(e.target.value))}/>
                </label>
              </div>
              <div className="checkboxes">
                {['has_staph', 'has_strep', 'has_e_coli', 'has_pseudomonas', 'has_klebsiella', 'pneumonia', 'uti', 'sepsis', 'skin_soft_tissue', 'intra_abdominal', 'meningitis'].map(field => (
                    <label key={field}>
                      <input
                          type="checkbox"
                          checked={p[field]}
                          onChange={() => handleCheckbox(i, field)}
                      /> {field.replace(/_/g, ' ').toUpperCase()}
                    </label>
                ))}
              </div>
              {results[i] && (
                  <div className="result-card">
                    <h3 style={{cursor: 'pointer'}} onClick={() => toggleExpanded(i)}>
                      Recommendation {expanded ? '🔼' : '🔽'}
                    </h3>
                    {expanded && (
                        <>
                          <p>
                            <strong>Recommendation:</strong> {results[i].recommendation} (Q-value: {results[i].q_value.toFixed(4)})
                          </p>
                          <p><strong>Explanation:</strong> {results[i].explanation}</p>

                          <div className="bar-chart">
                            {[...results[i].feature_importance]
                                .sort((a, b) => b.hybrid - a.hybrid)
                                .map((feat, j) => (
                                    <div key={j} className="bar-container">
                                      <div className="bar-label">{feat.feature}</div>
                                      <div className="bar-fill-wrapper">
                                        <div className="bar-fill hybrid"
                                             style={{width: `${(feat.hybrid * 100).toFixed(1)}%`}}>
                                          HYBRID {feat.hybrid.toFixed(2)}
                                        </div>
                                      </div>
                                    </div>
                                ))}
                          </div>
                        </>
                    )}
                  </div>
              )}
            </div>
        )}
        )}
        <div className="buttons">
          <button onClick={addPatient}>Add Another Patient</button>
          <button onClick={handleSubmit} disabled={loading}>{loading ? 'Processing...' : 'Submit'}</button>
        </div>

        {/*<div className="results">*/}
        {/*  <ResultsList results={results}/>*/}
        {/*</div>*/}

      </div>
  );
}

export default App;