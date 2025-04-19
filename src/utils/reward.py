def calculate_reward(prescribed_antibiotic, actual_antibiotic, patient_outcome, patient_data):
    """
    Calculate reward for the prescribed antibiotic based on various factors.
    
    Args:
        prescribed_antibiotic (str): The antibiotic prescribed by the agent
        actual_antibiotic (str): The actual antibiotic prescribed in the data
        patient_outcome (dict): Dictionary containing patient outcome information
        patient_data (dict): Dictionary containing patient clinical data
        
    Returns:
        float: Calculated reward value
    """
    # Base reward for matching the actual prescription
    if prescribed_antibiotic == actual_antibiotic:
        base_reward = 1.0
    else:
        # Check if they're in the same antibiotic class/family
        penicillins = ['AMOXICILLIN', 'PIPERACILLIN']
        cephalosporins = ['CEFAZOLIN', 'CEFEPIME', 'CEFTRIAXONE']
        fluoroquinolones = ['CIPROFLOXACIN', 'LEVOFLOXACIN']
        
        # If in same class, partial reward
        if (prescribed_antibiotic in penicillins and actual_antibiotic in penicillins) or \
           (prescribed_antibiotic in cephalosporins and actual_antibiotic in cephalosporins) or \
           (prescribed_antibiotic in fluoroquinolones and actual_antibiotic in fluoroquinolones):
            base_reward = 0.5
        else:
            base_reward = -0.5
    
    # Outcome-based adjustment
    good_outcome = (patient_outcome.get('expire_flag', 0) == 0) and (patient_outcome.get('los', 10) < 7)
    outcome_factor = 1.5 if good_outcome else 0.5
    
    # Organism-specific logic
    has_organism = False
    for org_key in ['has_staph', 'has_strep', 'has_e.coli', 'has_pseudomonas', 'has_klebsiella']:
        if org_key in patient_data and patient_data[org_key] == 1:
            has_organism = True
            # Check organism-specific appropriateness
            if org_key == 'has_staph' and prescribed_antibiotic in ['VANCOMYCIN', 'CEFAZOLIN']:
                base_reward += 0.5
            elif org_key == 'has_strep' and prescribed_antibiotic in ['AMOXICILLIN', 'CEFTRIAXONE']:
                base_reward += 0.5
            elif org_key == 'has_e.coli' and prescribed_antibiotic in ['CIPROFLOXACIN', 'CEFTRIAXONE']:
                base_reward += 0.5
            elif org_key == 'has_pseudomonas' and prescribed_antibiotic in ['PIPERACILLIN', 'MEROPENEM', 'CIPROFLOXACIN']:
                base_reward += 0.5
            elif org_key == 'has_klebsiella' and prescribed_antibiotic in ['CEFTRIAXONE', 'MEROPENEM']:
                base_reward += 0.5
    
    # Infection type appropriateness
    if patient_data.get('pneumonia', 0) == 1:
        if prescribed_antibiotic in ['CEFTRIAXONE', 'LEVOFLOXACIN', 'AZITHROMYCIN']:
            base_reward += 0.3
    elif patient_data.get('uti', 0) == 1:
        if prescribed_antibiotic in ['CIPROFLOXACIN', 'CEFTRIAXONE', 'NITROFURANTOIN']:
            base_reward += 0.3
    elif patient_data.get('sepsis', 0) == 1:
        if prescribed_antibiotic in ['VANCOMYCIN', 'PIPERACILLIN', 'MEROPENEM']:
            base_reward += 0.3
    
    # Penalize broad-spectrum use when not needed
    if prescribed_antibiotic in ['MEROPENEM', 'PIPERACILLIN', 'VANCOMYCIN']:
        # If no severe illness markers, penalize broad spectrum
        if not patient_data.get('sepsis', 0) and patient_data.get('wbc', 15) < 15:
            base_reward -= 0.2
    
    # Final reward calculation
    final_reward = base_reward * outcome_factor
    
    return final_reward 