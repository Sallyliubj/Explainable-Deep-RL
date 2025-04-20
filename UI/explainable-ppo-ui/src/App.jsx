// src/App.jsx
import React, { useState } from 'react';
import './App.css';
import ResultsList from './components/ResultsList';


const defaultPatient = {
  gender: 'M',
  wbc: 10,
  lactate: 1.5,
  creatinine: 1,
  has_staph: false,
  has_strep: false,
  has_e_coli: false,
  has_pseudomonas: false,
  has_klebsiella: false,
  pneumonia: false,
  uti: false,
  sepsis: false,
};

function App() {
  const [patients, setPatients] = useState([defaultPatient]);
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);

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

  const handleSubmit = async () => {
    setLoading(true);
    try {
      const res = await fetch('http://localhost:5000/predict', {
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
        {patients.map((p, i) => (
            <div key={i} className="patient-form">
              <h2>Patient {i + 1}</h2>
              <div>
                <label>Gender:
                  <select value={p.gender} onChange={e => handleChange(i, 'gender', e.target.value)}>
                    <option value="M">Male</option>
                    <option value="F">Female</option>
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
                {['has_staph', 'has_strep', 'has_e_coli', 'has_pseudomonas', 'has_klebsiella', 'pneumonia', 'uti', 'sepsis'].map(field => (
                    <label key={field}>
                      <input
                          type="checkbox"
                          checked={p[field]}
                          onChange={() => handleCheckbox(i, field)}
                      /> {field.replace(/_/g, ' ').toUpperCase()}
                    </label>
                ))}
              </div>
            </div>
        ))}
        <div className="buttons">
          <button onClick={addPatient}>Add Another Patient</button>
          <button onClick={handleSubmit} disabled={loading}>{loading ? 'Processing...' : 'Submit'}</button>
        </div>

        <div className="results">
          <ResultsList results={results} />
        </div>

      </div>
  );
}

export default App;