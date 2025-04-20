import React from 'react';
import ResultCard from './ResultCard';
import './Results.css';

const ResultsList = ({ results = [] }) => {
  return (
    <div className="results-list">
      {results.map((result, index) => (
        <ResultCard key={index} result={result} index={index} />
      ))}
    </div>
  );
};

export default ResultsList;
