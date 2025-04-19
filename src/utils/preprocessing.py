from datetime import datetime, timedelta
import pandas as pd
import numpy as np

def calculate_age(row):
    """Calculate patient age at admission time."""
    try:
        years_diff = row['intime'].year - row['dob'].year
        if (row['intime'].month, row['intime'].day) < (row['dob'].month, row['dob'].day):
            years_diff -= 1
        if years_diff > 100 or years_diff < 0:
            return 65
        return years_diff
    except:
        return 65

def get_events_before_prescription(row, micro_data, lab_data):
    """Extract microbiology and lab events before a prescription"""
    try:
        patient_micro = micro_data[micro_data['subject_id'] == row['subject_id']]
        patient_micro = patient_micro[patient_micro['hadm_id'] == row['hadm_id']]
        
        # Convert times to datetime safely
        rx_time = pd.to_datetime(row['startdate']) if 'startdate' in row else None
        if rx_time is None and 'charttime' in row:
            rx_time = pd.to_datetime(row['charttime'])
        
        if rx_time is None:
            return [], {}, {}
            
        # Convert chartdate safely, handling errors
        patient_micro['chartdate'] = pd.to_datetime(patient_micro['chartdate'], errors='coerce')
        patient_micro = patient_micro.dropna(subset=['chartdate'])
        
        # Get events in the 72 hours before prescription
        start_time = rx_time - timedelta(hours=72)
        patient_micro = patient_micro[(patient_micro['chartdate'] <= rx_time) & 
                                    (patient_micro['chartdate'] >= start_time)]
        
        # Get organism and antibiotic susceptibility information
        organisms = patient_micro['org_name'].dropna().unique().tolist()
        resistance_profile = {}
        
        for _, micro_row in patient_micro.iterrows():
            if isinstance(micro_row['interpretation'], str):
                ab_name = micro_row['ab_name'] if isinstance(micro_row['ab_name'], str) else 'unknown'
                resistance_profile[ab_name] = micro_row['interpretation']
        
        # Get lab values for the patient
        patient_labs = lab_data[lab_data['subject_id'] == row['subject_id']]
        patient_labs = patient_labs[patient_labs['hadm_id'] == row['hadm_id']]
        
        patient_labs['charttime'] = pd.to_datetime(patient_labs['charttime'], errors='coerce')
        patient_labs = patient_labs.dropna(subset=['charttime'])
        
        # Get lab values in the 24 hours before prescription
        lab_start_time = rx_time - timedelta(hours=24)
        patient_labs = patient_labs[(patient_labs['charttime'] <= rx_time) & 
                                  (patient_labs['charttime'] >= lab_start_time)]
        
        # Important lab values for infection
        lab_values = {}
        important_labs = {
            50811: 'wbc',       # White Blood Cell Count
            50813: 'lactate',   # Lactate
            50912: 'creatinine' # Creatinine 
        }
        
        for lab_id, lab_name in important_labs.items():
            lab_subset = patient_labs[patient_labs['itemid'] == lab_id]
            if not lab_subset.empty:
                lab_values[lab_name] = pd.to_numeric(lab_subset['valuenum'], errors='coerce').mean()
            else:
                lab_values[lab_name] = np.nan
                
        return organisms, resistance_profile, lab_values
    except Exception as e:
        print(f"Error in get_events_before_prescription: {e}")
        return [], {}, {}

def get_patient_diagnoses(row, diagnoses_data):
    """Extract diagnoses for a patient admission"""
    try:
        patient_dx = diagnoses_data[diagnoses_data['subject_id'] == row['subject_id']]
        patient_dx = patient_dx[patient_dx['hadm_id'] == row['hadm_id']]
        
        # Check for common infections by ICD-9 codes
        infection_codes = {
            'pneumonia': ['480', '481', '482', '483', '484', '485', '486'],
            'uti': ['590', '595', '599.0'],
            'sepsis': ['038', '995.91', '995.92'],
            'skin_soft_tissue': ['680', '681', '682'],
            'intra_abdominal': ['540', '541', '566', '567'],
            'meningitis': ['320', '321', '322']
        }
        
        diagnoses = {}
        
        for infection, codes in infection_codes.items():
            has_infection = False
            
            for code in codes:
                if patient_dx['icd9_code'].astype(str).str.startswith(code).any():
                    has_infection = True
                    break
                    
            diagnoses[infection] = has_infection
            
        return diagnoses
    except Exception as e:
        print(f"Error in get_patient_diagnoses: {e}")
        return {infection: False for infection in infection_codes.keys()} 