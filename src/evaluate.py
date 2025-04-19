import numpy as np
from utils.reward import calculate_reward

def evaluate_ppo(agent, X_test, y_test, prescription_contexts, idx_to_antibiotic):
    """
    Evaluate the trained PPO agent.
    
    Args:
        agent: Trained PPO agent
        X_test: Test features
        y_test: Test labels
        prescription_contexts: List of prescription contexts
        idx_to_antibiotic: Dictionary mapping indices to antibiotic names
        
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    predictions = []
    total_reward = 0
    appropriate_coverage = 0
    narrow_spectrum_used = 0
    broad_spectrum_used = 0
    
    broad_spectrum = ['MEROPENEM', 'PIPERACILLIN', 'VANCOMYCIN']
    
    for i, state in enumerate(X_test):
        action, _ = agent.get_action(state, training=False)
        predictions.append(action)
        
        predicted_antibiotic = idx_to_antibiotic.get(action, "UNKNOWN")
        actual_antibiotic = idx_to_antibiotic.get(y_test[i], "UNKNOWN")
        
        # Default outcome and patient data
        patient_outcome = {'expire_flag': 0, 'los': 5}
        patient_data = {}
        
        if i < len(prescription_contexts):
            context = prescription_contexts[i]
            patient_data = {
                'has_staph': 1 if 'staph' in str(context.get('organisms', [])).lower() else 0,
                'has_strep': 1 if 'strep' in str(context.get('organisms', [])).lower() else 0,
                'has_e.coli': 1 if 'e.coli' in str(context.get('organisms', [])).lower() else 0,
                'has_pseudomonas': 1 if 'pseudomonas' in str(context.get('organisms', [])).lower() else 0,
                'has_klebsiella': 1 if 'klebsiella' in str(context.get('organisms', [])).lower() else 0,
                'pneumonia': context.get('pneumonia', False),
                'uti': context.get('uti', False),
                'sepsis': context.get('sepsis', False),
                'wbc': context.get('wbc', 10),
                'lactate': context.get('lactate', 1.5),
                'creatinine': context.get('creatinine', 1.0)
            }
            
            # Calculate clinical metrics
            has_organism = False
            covered = False
            
            for org_key, org_name in [('has_staph', 'Staphylococcus'),
                                    ('has_strep', 'Streptococcus'),
                                    ('has_e.coli', 'E. coli'),
                                    ('has_pseudomonas', 'Pseudomonas'),
                                    ('has_klebsiella', 'Klebsiella')]:
                if patient_data.get(org_key, 0) == 1:
                    has_organism = True
                    if ((org_key == 'has_staph' and predicted_antibiotic in ['VANCOMYCIN', 'CEFAZOLIN']) or
                        (org_key == 'has_strep' and predicted_antibiotic in ['AMOXICILLIN', 'CEFTRIAXONE']) or
                        (org_key == 'has_e.coli' and predicted_antibiotic in ['CIPROFLOXACIN', 'CEFTRIAXONE']) or
                        (org_key == 'has_pseudomonas' and predicted_antibiotic in ['PIPERACILLIN', 'MEROPENEM', 'CIPROFLOXACIN']) or
                        (org_key == 'has_klebsiella' and predicted_antibiotic in ['CEFTRIAXONE', 'MEROPENEM'])):
                        covered = True
            
            if has_organism and covered:
                appropriate_coverage += 1
            
            if predicted_antibiotic in broad_spectrum:
                broad_spectrum_used += 1
            else:
                narrow_spectrum_used += 1
        
        reward = calculate_reward(predicted_antibiotic, actual_antibiotic, patient_outcome, patient_data)
        total_reward += reward
    
    accuracy = np.mean(predictions == y_test)
    appropriate_coverage_rate = appropriate_coverage / max(1, sum(1 for context in prescription_contexts[:len(X_test)]
                                             if any(org in str(context.get('organisms', [])).lower()
                                                    for org in ['staph', 'strep', 'e.coli', 'pseudomonas', 'klebsiella'])))
    narrow_spectrum_rate = narrow_spectrum_used / max(1, (narrow_spectrum_used + broad_spectrum_used))
    
    return {
        'accuracy': accuracy,
        'avg_reward': total_reward / len(X_test),
        'appropriate_coverage_rate': appropriate_coverage_rate,
        'narrow_spectrum_rate': narrow_spectrum_rate
    } 