import pandas as pd
from datetime import datetime, timedelta
import numpy as np

# Load datasets
microbiologyevents = pd.read_csv("./Kaggle/MICROBIOLOGYEVENTS.csv")
prescriptions = pd.read_csv("./Kaggle/PRESCRIPTIONS.csv")
patients = pd.read_csv("./Kaggle/PATIENTS.csv")
labevents = pd.read_csv("./Kaggle/LABEVENTS.csv")
diagnoses = pd.read_csv("./Kaggle/DIAGNOSES_ICD.csv")
icustays = pd.read_csv("./Kaggle/ICUSTAYS.csv", parse_dates=['intime', 'outtime'])
chartevents = pd.read_csv("./Kaggle/CHARTEVENTS.csv", low_memory=False)

print("Data loaded successfully")

# Preprocessing steps
# 1. Filter antibiotic prescriptions
antibiotics_list = [
    'AMOXICILLIN', 'AZITHROMYCIN', 'CEFAZOLIN', 'CEFEPIME', 'CEFTRIAXONE',
    'CIPROFLOXACIN', 'CLINDAMYCIN', 'DOXYCYCLINE', 'ERYTHROMYCIN', 'GENTAMICIN',
    'LEVOFLOXACIN', 'MEROPENEM', 'METRONIDAZOLE', 'PIPERACILLIN', 'VANCOMYCIN'
]

# Filter prescriptions to only antibiotics
antibiotic_prescriptions = prescriptions[
    prescriptions['drug_name_generic'].str.upper().str.contains('|'.join(antibiotics_list), na=False) |
    prescriptions['drug'].str.upper().str.contains('|'.join(antibiotics_list), na=False)
].copy()

print(f"Found {len(antibiotic_prescriptions)} antibiotic prescriptions")

# 2. Merge ICU stays with patient data
# Fix the date handling in MIMIC-III
patients['dob'] = pd.to_datetime(patients['dob'])
patients['dod'] = pd.to_datetime(patients['dod'])

# Calculate approximate age at admission using year diff instead of exact days
# This avoids the overflow error
patient_icu = pd.merge(
    patients[['subject_id', 'gender', 'dob', 'expire_flag']],
    icustays[['subject_id', 'hadm_id', 'icustay_id', 'intime', 'outtime', 'los']],
    on='subject_id'
)


# Safe age calculation
def calculate_age(row):
    try:
        # Calculate year difference
        years_diff = row['intime'].year - row['dob'].year

        # Adjust for month and day
        if (row['intime'].month, row['intime'].day) < (row['dob'].month, row['dob'].day):
            years_diff -= 1

        # MIMIC-III sometimes has shifted dates for deidentification
        # Cap reasonable age range
        if years_diff > 100 or years_diff < 0:
            return 65  # Use median adult age as fallback
        return years_diff
    except:
        return 65  # Use median adult age as fallback

# Calculate age using the safer method
patient_icu['age'] = patient_icu.apply(calculate_age, axis=1)
patient_icu = patient_icu[patient_icu['age'] >= 18]  # Only adult patients

print(f"Working with {len(patient_icu)} ICU stays for adult patients")


# 3. Get microbiology and lab data for each ICU stay
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
            # If we can't determine the time, use a default approach
            return [], {}, {}

        # Convert chartdate safely, handling errors
        patient_micro['chartdate'] = pd.to_datetime(patient_micro['chartdate'], errors='coerce')

        # Remove rows with invalid dates
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

        # Convert charttime safely, handling errors
        patient_labs['charttime'] = pd.to_datetime(patient_labs['charttime'], errors='coerce')

        # Remove rows with invalid dates
        patient_labs = patient_labs.dropna(subset=['charttime'])

        # Get lab values in the 24 hours before prescription
        lab_start_time = rx_time - timedelta(hours=24)
        patient_labs = patient_labs[(patient_labs['charttime'] <= rx_time) &
                                   (patient_labs['charttime'] >= lab_start_time)]

        # Important lab values for infection - WBC, lactate, creatinine
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

# 4. Get diagnosis information
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

# 5. Prepare features for model training
# Let's create a dataset of prescriptions with their context
print("Preparing prescription contexts...")
prescription_contexts = []

# Sample only a portion for demonstration (adjust based on computational resources)
# Using a smaller sample size to ensure processing completes
sample_size = min(500, len(antibiotic_prescriptions))
sample_prescriptions = antibiotic_prescriptions.sample(sample_size, random_state=42)

for idx, row in sample_prescriptions.iterrows():
    try:
        organisms, resistance, labs = get_events_before_prescription(row, microbiologyevents, labevents)
        diagnoses_info = get_patient_diagnoses(row, diagnoses)

        # Get patient info
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

print(f"Created context for {len(prescription_contexts)} prescriptions")

# Convert to DataFrame
prescription_df = pd.DataFrame(prescription_contexts)

# Convert organism lists to binary features
def has_organism(row, organism_type):
    if isinstance(row.get('organisms'), list):
        return any(organism_type.lower() in str(org).lower() for org in row['organisms'])
    return False

common_organisms = ['staph', 'strep', 'e.coli', 'pseudomonas', 'klebsiella']
for org in common_organisms:
    prescription_df[f'has_{org}'] = prescription_df.apply(lambda row: has_organism(row, org), axis=1)


import matplotlib.pyplot as plt
import seaborn as sns

# Convert to appropriate types
bool_columns = ['pneumonia', 'uti', 'sepsis', 'skin_soft_tissue', 'intra_abdominal', 'meningitis']
for col in bool_columns:
    prescription_df[col] = prescription_df[col].astype(bool)

# Plotting function
import os

def plot_distribution(column, title, xlabel, rotation=0, bins=20, save_dir="plots"):
    os.makedirs(save_dir, exist_ok=True)  # Create the directory if it doesn't exist
    plt.figure(figsize=(10, 5))
    if prescription_df[column].dtype == 'bool' or prescription_df[column].nunique() < 10:
        sns.countplot(x=column, data=prescription_df)
    else:
        sns.histplot(prescription_df[column], bins=bins, kde=True)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.xticks(rotation=rotation)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save the figure
    filename = f"{column}_distribution.png".replace(" ", "_").lower()
    plt.savefig(os.path.join(save_dir, filename))
    plt.close()


# Continuous / categorical feature plots
plot_distribution('age', 'Age Distribution of Patients', 'Age')
plot_distribution('wbc', 'WBC Count Distribution', 'White Blood Cells')
plot_distribution('lactate', 'Lactate Level Distribution', 'Lactate')
plot_distribution('creatinine', 'Creatinine Level Distribution', 'Creatinine')
plot_distribution('gender', 'Gender Distribution', 'Gender')
#plot_distribution('antibiotic_class', 'Prescribed Antibiotic Class', 'Antibiotic Class', rotation=45)

# Infection types
for col in bool_columns:
    plot_distribution(col, f'{col.replace("_", " ").title()} Occurrence', col)

# Organism binary flags
organism_flags = ['has_staph', 'has_strep', 'has_e.coli', 'has_pseudomonas', 'has_klebsiella']
for col in organism_flags:
    plot_distribution(col, f'{col.replace("_", " ").title()} Presence', col)
