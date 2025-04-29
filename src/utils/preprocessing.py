# This preprocessing code was adapted from https://www.kaggle.com/code/yepvaishz/rl-research
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer

from data.data_loader import load_mimic_data, ANTIBIOTICS_LIST

import os
import pickle
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
    

def prepare_prescription_contexts(antibiotic_prescriptions, microbiologyevents, labevents, diagnoses, patient_icu, sample_size=500):
    """Prepare prescription contexts for training."""
    print("Preparing prescription contexts...")
    prescription_contexts = []
    
    sample_prescriptions = antibiotic_prescriptions.sample(min(sample_size, len(antibiotic_prescriptions)), random_state=42)
    
    for idx, row in sample_prescriptions.iterrows():
        try:
            organisms, resistance, labs = get_events_before_prescription(row, microbiologyevents, labevents)
            diagnoses_info = get_patient_diagnoses(row, diagnoses)
            
            patient_info = patient_icu[
                (patient_icu['subject_id'] == row['subject_id']) & 
                (patient_icu['hadm_id'] == row['hadm_id'])
            ]
            
            if not patient_info.empty:
                context = {
                    'subject_id': row['subject_id'],
                    'hadm_id': row['hadm_id'],
                    'icustay_id': row['icustay_id'] if 'icustay_id' in row else None,
                    'age': patient_info['age'].values[0] if not patient_info.empty else 65,
                    'gender': patient_info['gender'].values[0] if not patient_info.empty else 'M',
                    'organisms': organisms,
                    'resistance': resistance,
                    'wbc': labs.get('wbc', None),
                    'lactate': labs.get('lactate', None),
                    'creatinine': labs.get('creatinine', None),
                    'pneumonia': diagnoses_info.get('pneumonia', False),
                    'uti': diagnoses_info.get('uti', False),
                    'sepsis': diagnoses_info.get('sepsis', False),
                    'skin_soft_tissue': diagnoses_info.get('skin_soft_tissue', False),
                    'intra_abdominal': diagnoses_info.get('intra_abdominal', False),
                    'meningitis': diagnoses_info.get('meningitis', False),
                    'prescribed_antibiotic': row['drug_name_generic'] if not pd.isna(row['drug_name_generic']) else row['drug'],
                    'los': patient_info['los'].values[0] if not patient_info.empty else 5.0,
                    'expire_flag': patient_info['expire_flag'].values[0] if not patient_info.empty else 0
                }
                
                prescription_contexts.append(context)
                
                if len(prescription_contexts) % 50 == 0:
                    print(f"Processed {len(prescription_contexts)} prescriptions")
                    
        except Exception as e:
            print(f"Error processing row {idx}: {e}")
            
    return prescription_contexts


def preprocess_and_save_data(data_path='dataset/mimic-iii-clinical-database'):

    if not os.path.exists('dataset/train/X_train.npy') \
        or not os.path.exists('dataset/test/X_test.npy') \
        or not os.path.exists('dataset/train/y_train.npy') \
        or not os.path.exists('dataset/test/y_test.npy')\
        or not os.path.exists('dataset/train/prescription_contexts.pkl') \
        or not os.path.exists('dataset/train/idx_to_antibiotic.pkl') \
        or not os.path.exists('dataset/train/num_cols.pkl') \
        or not os.path.exists('dataset/train/cat_cols.pkl') \
        or not os.path.exists('dataset/train/X_original.pkl') \
        or not os.path.exists('dataset/train/state_size.txt') \
        or not os.path.exists('dataset/train/n_actions.txt'): 
        
        print("Data not found. Processing data...\n")
        # Load data
        (microbiologyevents, prescriptions, patients, labevents, 
        diagnoses, icustays, chartevents) = load_mimic_data(data_path)
        
        # Filter antibiotic prescriptions
        antibiotic_prescriptions = prescriptions[
            prescriptions['drug_name_generic'].str.upper().str.contains('|'.join(ANTIBIOTICS_LIST), na=False) |
            prescriptions['drug'].str.upper().str.contains('|'.join(ANTIBIOTICS_LIST), na=False)
        ].copy()
        
        # Prepare patient ICU data
        patients['dob'] = pd.to_datetime(patients['dob'])
        patients['dod'] = pd.to_datetime(patients['dod'])
        
        patient_icu = pd.merge(
            patients[['subject_id', 'gender', 'dob', 'expire_flag']], 
            icustays[['subject_id', 'hadm_id', 'icustay_id', 'intime', 'outtime', 'los']], 
            on='subject_id'
        )
        
        patient_icu['age'] = patient_icu.apply(calculate_age, axis=1)
        patient_icu = patient_icu[patient_icu['age'] >= 18]
        
        # Prepare prescription contexts
        prescription_contexts = prepare_prescription_contexts(
            antibiotic_prescriptions, microbiologyevents, labevents, diagnoses, patient_icu
        )
        
        # Convert to DataFrame and prepare features
        prescription_df = pd.DataFrame(prescription_contexts)
        
        # Create antibiotic_class column by extracting the antibiotic name from prescribed_antibiotic
        def extract_antibiotic_class(prescribed_antibiotic):
            if pd.isna(prescribed_antibiotic):
                return 'UNKNOWN'
            prescribed_upper = prescribed_antibiotic.upper()
            for antibiotic in ANTIBIOTICS_LIST:
                if antibiotic in prescribed_upper:
                    return antibiotic
            return 'UNKNOWN'
        
        prescription_df['antibiotic_class'] = prescription_df['prescribed_antibiotic'].apply(extract_antibiotic_class)
        
        # Remove rows with unknown antibiotic class
        prescription_df = prescription_df[prescription_df['antibiotic_class'] != 'UNKNOWN']
        
        if len(prescription_df) == 0:
            raise ValueError("No valid prescriptions found after filtering. Check the antibiotic names and data.")
        
        # Feature engineering
        for col in ['wbc', 'lactate', 'creatinine']:
            if col in prescription_df.columns:
                median_val = prescription_df[col].median()
                if pd.isna(median_val):
                    median_val = 0.0
                prescription_df[col] = prescription_df[col].fillna(median_val)
            else:
                prescription_df[col] = 0.0
        
        # Convert organism lists to binary features
        def has_organism(row, organism_type):
            if isinstance(row.get('organisms'), list):
                return any(organism_type.lower() in str(org).lower() for org in row['organisms'])
            return False
        
        common_organisms = ['staph', 'strep', 'e.coli', 'pseudomonas', 'klebsiella']
        for org in common_organisms:
            prescription_df[f'has_{org}'] = prescription_df.apply(lambda row: has_organism(row, org), axis=1)
        
        # Prepare features for model
        columns_to_drop = ['subject_id', 'hadm_id', 'icustay_id', 'prescribed_antibiotic', 
                        'organisms', 'resistance']
        X_columns = [col for col in prescription_df.columns if col not in columns_to_drop + ['antibiotic_class']]
        X = prescription_df[X_columns].copy()
        y = prescription_df['antibiotic_class']
        
        # Convert boolean columns to int
        bool_cols = X.select_dtypes(include=['bool']).columns
        X[bool_cols] = X[bool_cols].astype(int)
        
        # Handle categorical and numerical features
        cat_cols = ['gender']
        num_cols = [col for col in X.columns if col not in cat_cols]
        
        for col in num_cols:
            X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
        
        X = X.fillna(0)
        
        # Setup preprocessing
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), num_cols),
                ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
            ])
        
        # Process features
        X_processed = preprocessor.fit_transform(X)
        X_processed = np.array(X_processed.toarray() if hasattr(X_processed, 'toarray') else X_processed)
        
        # Prepare action space
        unique_antibiotics = prescription_df['antibiotic_class'].unique()
        n_actions = len(unique_antibiotics)
        antibiotic_to_idx = {ab: i for i, ab in enumerate(unique_antibiotics)}
        idx_to_antibiotic = {i: ab for ab, i in antibiotic_to_idx.items()}
        
        print(f"Number of unique antibiotics: {n_actions}")
        print("Antibiotic classes:", unique_antibiotics)
        
        # Convert targets to indices
        y_indices = np.array([antibiotic_to_idx[ab] for ab in prescription_df['antibiotic_class']])
        X_train, X_test, y_train, y_test = train_test_split(X_processed, y_indices, test_size=0.2, random_state=42)
        
        print(f"Training with {X_train.shape[0]} samples, testing with {X_test.shape[0]} samples")
        
        # Initialize and train agent
        state_size = X_processed.shape[1]


        # Save data
        print("Saving data...")
        os.makedirs('dataset/train/', exist_ok=True)
        os.makedirs('dataset/test/', exist_ok=True)
        
        np.save('dataset/train/X_train.npy', X_train)
        np.save('dataset/test/X_test.npy', X_test)
        np.save('dataset/train/y_train.npy', y_train)
        np.save('dataset/test/y_test.npy', y_test)
        
        pickle.dump(prescription_contexts, open('dataset/train/prescription_contexts.pkl', 'wb'))
        pickle.dump(idx_to_antibiotic, open('dataset/train/idx_to_antibiotic.pkl', 'wb'))
        pickle.dump(num_cols, open('dataset/train/num_cols.pkl', 'wb'))
        pickle.dump(cat_cols, open('dataset/train/cat_cols.pkl', 'wb'))
        pickle.dump(X, open('dataset/train/X_original.pkl', 'wb'))
        
        with open('dataset/train/state_size.txt', 'w') as f:
            f.write(str(state_size))

        with open('dataset/train/n_actions.txt', 'w') as f:
            f.write(str(n_actions))

        print("Data saved successfully.\n")

    else:
        print("Data found. Loading data...\n")

        X_train = np.load('dataset/train/X_train.npy')
        X_test = np.load('dataset/test/X_test.npy')
        y_train = np.load('dataset/train/y_train.npy')
        y_test = np.load('dataset/test/y_test.npy')
        
        prescription_contexts = pickle.load(open('dataset/train/prescription_contexts.pkl', 'rb'))
        idx_to_antibiotic = pickle.load(open('dataset/train/idx_to_antibiotic.pkl', 'rb'))
        num_cols = pickle.load(open('dataset/train/num_cols.pkl', 'rb'))
        cat_cols = pickle.load(open('dataset/train/cat_cols.pkl', 'rb'))
        X = pickle.load(open('dataset/train/X_original.pkl', 'rb'))
        
        # Load state_size and n_actions from text files
        with open('dataset/train/state_size.txt', 'r') as f:
            state_size = int(f.read().strip())

        with open('dataset/train/n_actions.txt', 'r') as f:
            n_actions = int(f.read().strip())

    print("Data loaded successfully.\n")
    return state_size, n_actions, X_train, X_test, y_train, y_test, prescription_contexts, idx_to_antibiotic, num_cols, cat_cols, X