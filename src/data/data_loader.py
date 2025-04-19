import pandas as pd

def load_mimic_data(base_path):
    """
    Load MIMIC-III datasets from the specified base path.
    
    Args:
        base_path (str): Base path to the MIMIC-III dataset directory
        
    Returns:
        tuple: Loaded dataframes (microbiologyevents, prescriptions, patients, labevents, diagnoses, icustays, chartevents)
    """
    print("Loading data...")
    
    # Load datasets
    microbiologyevents = pd.read_csv(f"{base_path}/MICROBIOLOGYEVENTS.csv")
    prescriptions = pd.read_csv(f"{base_path}/PRESCRIPTIONS.csv")
    patients = pd.read_csv(f"{base_path}/PATIENTS.csv")
    labevents = pd.read_csv(f"{base_path}/LABEVENTS.csv")
    diagnoses = pd.read_csv(f"{base_path}/DIAGNOSES_ICD.csv")
    icustays = pd.read_csv(f"{base_path}/ICUSTAYS.csv", parse_dates=['intime', 'outtime'])
    chartevents = pd.read_csv(f"{base_path}/CHARTEVENTS.csv", low_memory=False)

    print("Data loaded successfully")
    
    return (microbiologyevents, prescriptions, patients, labevents, 
            diagnoses, icustays, chartevents)

# List of antibiotics used in the study
ANTIBIOTICS_LIST = [
    'AMOXICILLIN', 'AZITHROMYCIN', 'CEFAZOLIN', 'CEFEPIME', 'CEFTRIAXONE',
    'CIPROFLOXACIN', 'CLINDAMYCIN', 'DOXYCYCLINE', 'ERYTHROMYCIN', 'GENTAMICIN',
    'LEVOFLOXACIN', 'MEROPENEM', 'METRONIDAZOLE', 'PIPERACILLIN', 'VANCOMYCIN'
] 