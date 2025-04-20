import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from typing import Tuple, List

class AntibioticEnvironment:
    """RL Environment for antibiotic prescription decisions"""
    
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.features, self.feature_names = self._preprocess_features()
        # Put 'none' first so it's not the default high-index action
        self.action_space = [
            'none',
            'piperacillin/tazobactam', 'vancomycin', 'cefepime', 
            'meropenem', 'levofloxacin'
        ]
        self.observation_shape = self.features.shape[1]
        self.current_patient_data = None

    # def _preprocess_features(self) -> np.ndarray:
    #     """Prepare feature matrix for RL"""
    #     # Convert categorical variables with fixed categories
    #     encoder = OneHotEncoder(sparse_output=False, categories=[['M', 'F', 'OTHER']])
    #     # Ensure gender is uppercase and handle unknown values
    #     gender_data = self.data['gender'].str.upper()
    #     gender_data = gender_data.map(lambda x: x if x in ['M', 'F'] else 'OTHER')
    #     categorical = encoder.fit_transform(gender_data.values.reshape(-1, 1))
    #
    #     # Scale numerical features individually
    #     scaler = StandardScaler()
    #     los_scaled = scaler.fit_transform(self.data[['los']])
    #     mortality_scaled = self.data['mortality'].values.reshape(-1, 1)  # Keep binary
    #
    #     # Combine all features
    #     features = np.concatenate([categorical, los_scaled, mortality_scaled], axis=1)
    #
    #     # Add feature to indicate if patient has any prescriptions
    #     has_prescription = (self.data['drug_name_generic'] != 'none').astype(float).values.reshape(-1, 1)
    #     features = np.concatenate([features, has_prescription], axis=1)
    #
    #     return features

    def _preprocess_features(self) -> Tuple[np.ndarray, List[str]]:
        """Prepare feature matrix and names for RL"""
        # 1. Gender (one-hot)
        encoder = OneHotEncoder(sparse_output=False, categories=[['M', 'F', 'OTHER']])
        gender_data = self.data['gender'].str.upper().map(lambda x: x if x in ['M', 'F'] else 'OTHER')
        gender_encoded = encoder.fit_transform(gender_data.values.reshape(-1, 1))
        gender_feature_names = encoder.get_feature_names_out(['gender']).tolist()

        # 2. Length of stay (LOS)
        scaler = StandardScaler()
        los_scaled = scaler.fit_transform(self.data[['los']])
        los_feature_name = ['length_of_stay']

        # 3. Mortality
        mortality = self.data['mortality'].values.reshape(-1, 1)
        mortality_feature_name = ['mortality']

        # 4. Has prescription
        has_prescription = (self.data['drug_name_generic'] != 'none').astype(float).values.reshape(-1, 1)
        has_prescription_name = ['has_prescription']

        # Combine all
        features = np.concatenate([gender_encoded, los_scaled, mortality, has_prescription], axis=1)
        feature_names = gender_feature_names + los_feature_name + mortality_feature_name + has_prescription_name

        return features, feature_names
    
    def reset(self) -> np.ndarray:
        """Start new episode"""
        self.current_idx = np.random.randint(0, len(self.data))
        self.current_patient_data = self.data.iloc[self.current_idx].to_dict()
        return self.features[self.current_idx]
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """Execute one timestep"""
        reward = self._calculate_reward(action)
        done = True  # Single-step episode
        return self.features[self.current_idx], reward, done, {}
    
    def _calculate_reward(self, action: int) -> float:
        """Calculate reward based on clinical outcome"""
        row = self.data.iloc[self.current_idx]
        prescribed = self.action_space[action]
        actual = row.get('drug_name_generic', 'none')
        
        # Normalize LOS penalty to be between -5 and 0
        los = float(row['los'])
        los_penalty = -5 * min(abs(los - 5) / 10, 1.0)  # Cap at -5 for LOS differences > 10
        
        # Stronger reward/penalty for prescription
        if prescribed == 'none':
            # Penalize choosing 'none' when there was an actual prescription
            appropriate = 15 if actual == 'none' else -15
        else:
            # Reward for matching any part of the actual prescription
            if actual != 'none' and (prescribed.lower() in actual.lower() or 
                any(drug.lower() in prescribed.lower() for drug in actual.split(','))):
                appropriate = 15
            else:
                appropriate = -15
        
        # Survival is still important but not overwhelmingly so
        survival = 10 if row['mortality'] == 0 else -10
        
        return survival + los_penalty + appropriate

    def get_current_patient(self):
        return self.current_patient_data

