import os
import pandas as pd
from typing import Dict, List, Optional
import logging
import numpy as np
from sklearn.model_selection import train_test_split

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataLoader:
    """Class to handle data loading and validation with proper dtype handling"""
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.datasets: Dict[str, pd.DataFrame] = {}
        
    def load_file(self, filename: str) -> pd.DataFrame:
        """Load CSV with dtype specifications and error handling"""
        try:
            file_path = os.path.join(self.data_path, filename)
            if not os.path.exists(file_path):
                logger.error(f"File not found: {file_path}")
                return pd.DataFrame()
            
            dtype_map = {
                'DIAGNOSES_ICD.csv': {'icd9_code': str},
                'MICROBIOLOGYEVENTS.csv': {'org_itemid': str, 'ab_itemid': str},
                'PRESCRIPTIONS.csv': {'drug_name_generic': str}
            }
            data = pd.read_csv(
                file_path,
                dtype=dtype_map.get(filename, None),
                low_memory=False
            )
            logger.info(f"Loaded {filename}: {data.shape}")
            return data
        except Exception as e:
            logger.error(f"Error loading {filename}: {str(e)}")
            return pd.DataFrame()
            
    def load_mimic_data(self, split=None, test_size=0.2, random_state=42) -> Dict[str, pd.DataFrame]:
        """
        Load all required datasets with validation and optional train/test split
        
        Args:
            split: If 'train', returns training data; if 'test', returns test data; 
                   if None, returns all data
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
        
        Returns:
            Dictionary of DataFrames containing the requested data split
        """
        required_files = [
            "MICROBIOLOGYEVENTS.csv", "PRESCRIPTIONS.csv", "PATIENTS.csv",
            "LABEVENTS.csv", "DIAGNOSES_ICD.csv", "ICUSTAYS.csv", "CHARTEVENTS.csv"
        ]
        
        for file in required_files:
            key = file.split('.')[0].lower()
            self.datasets[key] = self.load_file(file)
        
        self._validate_datasets()
        
        if split is None:
            return self.datasets
        
        # Split each dataset using the same random state
        split_datasets = {}
        for key, df in self.datasets.items():
            train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
            split_datasets[key] = train_df if split == 'train' else test_df
        
        return split_datasets

    def _validate_datasets(self) -> None:
        """Validate loaded datasets"""
        required = ['diagnoses_icd', 'prescriptions', 'patients']
        for name in required:
            if name not in self.datasets or self.datasets[name].empty:
                raise ValueError(f"Missing or empty required dataset: {name}")
        