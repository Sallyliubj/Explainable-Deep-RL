import pandas as pd
from typing import Dict, List, Optional
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def preprocess_data(datasets: Dict[str, pd.DataFrame],
                    infection_codes: Optional[List[str]] = None, 
                    lab_itemids: Optional[List[int]] = None, 
                    vital_itemids: Optional[List[int]] = None) -> pd.DataFrame:
     """
     Preprocess and merge data with comprehensive debugging and error handling
    
     Args:
        datasets (Dict[str, pd.DataFrame]): Dictionary of loaded datasets
        infection_codes (Optional[List[str]]): Predefined infection codes
        lab_itemids (Optional[List[int]]): Predefined lab item IDs
        vital_itemids (Optional[List[int]]): Predefined vital sign item IDs
    
     Returns:
        pd.DataFrame: Preprocessed patient data
     """
     
     logger.info("Starting comprehensive data preprocessing...")
     try:
        # Validate required datasets
        required_datasets = ['diagnoses_icd', 'prescriptions', 'patients', 'icustays']
        for dataset in required_datasets:
            if dataset not in datasets or datasets[dataset].empty:
                logger.error(f"Missing or empty dataset: {dataset}")
                raise ValueError(f"Dataset {dataset} is missing or empty")
        
        # Convert all column names to lowercase for each dataset
        logger.info("Converting all column names to lowercase...")
        for name, df in datasets.items():
            datasets[name].columns = df.columns.str.lower()
            logger.info(f"Converted columns for {name}: {datasets[name].columns.tolist()}")
        
        # Extract core datasets
        diagnoses = datasets['diagnoses_icd']
        prescriptions = datasets['prescriptions']
        patients = datasets['patients']
        icustays = datasets['icustays']
        
        # Debug dataset information
        logger.info("Dataset Sizes:")
        for name, df in datasets.items():
            logger.info(f"{name}: {df.shape}")
        
        # Debug column information for diagnoses
        logger.info("\nDiagnoses Columns:")
        logger.info(diagnoses.columns.tolist())

        # Dynamically select infection codes if not provided
        if infection_codes is None or len(infection_codes) == 0:
            # Clean and count ICD9 codes
            diagnoses.columns = diagnoses.columns.str.lower()
            diagnoses['clean_icd9'] = diagnoses['icd9_code'].str.replace('.', '').str.strip()
            code_counts = diagnoses['clean_icd9'].value_counts()
            
            # Select top infection codes (adjust threshold as needed)
            infection_codes = list(code_counts[code_counts > 10].head(20).index.tolist())
            logger.info(f"Dynamically selected infection codes: {infection_codes}")
        
        # Clean ICD9 codes
        diagnoses['clean_icd9'] = diagnoses['icd9_code'].str.replace('.', '').str.strip()
        
        # Filter infection diagnoses with more robust method
        infection_mask = diagnoses['clean_icd9'].str.contains('|'.join(infection_codes), case=False, na=False)
        infected_admissions = diagnoses[infection_mask][['subject_id', 'hadm_id']].drop_duplicates()
        
        logger.info(f"Found {len(infected_admissions)} unique infected admissions")
        
        if infected_admissions.empty:
            logger.error("No infection admissions found. Check infection codes and data.")
            raise ValueError("No infection admissions found with given codes")
        
        # Merge patient information
        data = infected_admissions.merge(
            patients[['subject_id', 'gender', 'dob', 'dod']],
            on='subject_id',
            how='left'
        )
        
        # Add antibiotic prescriptions with more flexible matching
        abx_keywords = ['penicillin', 'cephalosporin', 'quinolone', 'vancomycin', 
                        'antibiotic', 'antimicrobial']
        abx = prescriptions[
            prescriptions['drug_name_generic'].str.contains('|'.join(abx_keywords), case=False, na=False)
        ]


        # Aggregate antibiotics per admission
        abx_grouped = abx.groupby('hadm_id')['drug_name_generic'].agg(lambda x: ','.join(set(x))).reset_index()
        
        # Merge antibiotics data
        data = data.merge(
            abx_grouped,
            on='hadm_id',
            how='left'
        )
        
        # Calculate mortality and length of stay
        data['mortality'] = (data['dod'].notna()).astype(int)
        
        # Calculate median length of stay per admission
        los_data = icustays.groupby('hadm_id')['los'].median().reset_index()
        data = data.merge(los_data, on='hadm_id', how='left')
        
        # Final data cleaning
        data = data.drop(columns=['dob', 'dod'], errors='ignore')
        data = data.fillna({
            'drug_name_generic': 'none', 
            'los': data['los'].median(),
            'mortality': 0
        })
        
        # Log final dataset information
        logger.info(f"Preprocessed data shape: {data.shape}")
        logger.info(f"Columns: {data.columns.tolist()}")
        
        return data
        
     except Exception as e:
        logger.error(f"Preprocessing failed: {str(e)}")
        # Log additional context
        import traceback
        logger.error(traceback.format_exc())
        return pd.DataFrame()