import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from typing import Dict, List, Tuple
import tensorflow as tf

class ExplainableAgent:
    """Wrapper class to add explainability to the DQN agent"""
    
    def __init__(self, agent, feature_names: List[str] = None):
        self.agent = agent
        self.feature_names = feature_names or [
            'gender_M', 'gender_F', 'gender_OTHER',
            'length_of_stay', 'mortality',
            'has_prescription'
        ]
        
        # Set color palette
        self.colors = sns.color_palette("husl", len(self.feature_names))
        
        # Create SHAP explainer
        self.background_data = None
        self.explainer = None
        
    def _model_predict_wrapper(self, x):
        """Wrapper for model prediction to work with SHAP"""
        q_values, _ = self.agent.model.predict(x)
        return q_values
    
    def initialize_explainer(self, background_data: np.ndarray, n_samples: int = 100):
        """Initialize SHAP explainer with background data"""
        if background_data.shape[0] > n_samples:
            indices = np.random.choice(background_data.shape[0], n_samples, replace=False)
            self.background_data = background_data[indices]
        else:
            self.background_data = background_data
            
        self.explainer = shap.KernelExplainer(
            self._model_predict_wrapper,
            self.background_data
        )
        
    def explain_prediction(self, state: np.ndarray) -> Tuple[Dict, Dict]:
        """
        Explain a single prediction using both attention and SHAP values
        
        Returns:
            Tuple containing:
            - Dictionary of attention-based feature importance
            - Dictionary of SHAP-based feature importance
        """
        # Get attention-based importance
        attention_importance = self.agent.get_feature_importance(state)
        
        # Get SHAP values
        if self.explainer is None:
            raise ValueError("SHAP explainer not initialized. Call initialize_explainer first.")
            
        shap_values = self.explainer.shap_values(
            state.reshape(1, -1),
            nsamples=100  # Number of samples for SHAP approximation
        )
        
        # Average SHAP values across all actions
        avg_shap_values = np.mean([np.abs(sv) for sv in shap_values], axis=0)[0]
        shap_importance = dict(zip(self.feature_names, avg_shap_values))
        
        return attention_importance, shap_importance
    
    def plot_feature_importance(self, state: np.ndarray, save_path: str = None):
        """Plot feature importance using both attention and SHAP values"""
        attention_importance, shap_importance = self.explain_prediction(state)
        
        # Set style
        sns.set_style("whitegrid")
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 14))
        
        # Plot attention-based importance
        features = list(attention_importance.keys())
        values = list(attention_importance.values())
        
        # Sort by importance in descending order
        sorted_indices = np.argsort(values)[::-1]  # Reverse to get descending order
        features = [features[i] for i in sorted_indices]
        values = [values[i] for i in sorted_indices]
        
        # Create a color mapping to ensure consistency
        feature_to_color = dict(zip(self.feature_names, self.colors))
        colors = [feature_to_color[feature] for feature in features]
        
        bars = ax1.barh(features, values, color=colors)
        ax1.set_title('Attention-based Feature Importance', fontsize=12, pad=20)
        ax1.set_xlabel('Importance Score', fontsize=10)
        ax1.tick_params(axis='both', which='major', labelsize=10)
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            ax1.text(width, bar.get_y() + bar.get_height()/2,
                    f'{width:.3f}', 
                    ha='left', va='center', fontsize=9)
        
        # Plot SHAP-based importance
        features = list(shap_importance.keys())
        values = list(shap_importance.values())
        
        # Sort by importance in descending order
        sorted_indices = np.argsort(values)[::-1]  # Reverse to get descending order
        features = [features[i] for i in sorted_indices]
        values = [values[i] for i in sorted_indices]
        colors = [feature_to_color[feature] for feature in features]
        
        bars = ax2.barh(features, values, color=colors)
        ax2.set_title('SHAP-based Feature Importance', fontsize=12, pad=20)
        ax2.set_xlabel('|SHAP value|', fontsize=10)
        ax2.tick_params(axis='both', which='major', labelsize=10)
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            ax2.text(width, bar.get_y() + bar.get_height()/2,
                    f'{width:.3f}', 
                    ha='left', va='center', fontsize=9)
        
        # Add legend for feature categories
        legend_elements = [plt.Rectangle((0,0),1,1, facecolor=feature_to_color[feature]) 
                         for feature in self.feature_names]
        fig.legend(legend_elements, self.feature_names,
                  loc='center right', bbox_to_anchor=(1.15, 0.5),
                  title='Features', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
        else:
            plt.show()
            
    def explain_batch(self, states: np.ndarray, n_samples: int = 10) -> Dict[str, float]:
        """
        Analyze feature importance across multiple states
        
        Args:
            states: Array of states to analyze
            n_samples: Number of random states to analyze if states array is large
            
        Returns:
            Dictionary of average feature importance scores
        """
        if states.shape[0] > n_samples:
            indices = np.random.choice(states.shape[0], n_samples, replace=False)
            states = states[indices]
            
        attention_scores = []
        shap_scores = []
        
        for state in states:
            att_imp, shap_imp = self.explain_prediction(state.reshape(1, -1))
            attention_scores.append(list(att_imp.values()))
            shap_scores.append(list(shap_imp.values()))
            
        # Average importance scores
        avg_attention = np.mean(attention_scores, axis=0)
        avg_shap = np.mean(shap_scores, axis=0)
        
        # Combine both metrics (normalized)
        combined_importance = (avg_attention / np.sum(avg_attention) + 
                             avg_shap / np.sum(avg_shap)) / 2
                             
        return dict(zip(self.feature_names, combined_importance))

def explain_recommendation(recommendation, patient_data):
    antibiotic, q_value = recommendation
    explanation = f"Recommended {antibiotic} (Q-value: {q_value:.4f}).\n"

    organisms = []
    for org_key, org_name in [
        ('has_staph', 'Staphylococcus'),
        ('has_strep', 'Streptococcus'),
        ('has_e.coli', 'E. coli'),
        ('has_pseudomonas', 'Pseudomonas'),
        ('has_klebsiella', 'Klebsiella')
    ]:
        if patient_data.get(org_key, 0) == 1:
            organisms.append(org_name)

    if organisms:
        explanation += f"- Patient has {', '.join(organisms)} organism(s).\n"
        coverage = {
            'Staphylococcus': ['VANCOMYCIN', 'CEFAZOLIN'],
            'Streptococcus': ['AMOXICILLIN', 'CEFTRIAXONE'],
            'E. coli': ['CIPROFLOXACIN', 'CEFTRIAXONE'],
            'Pseudomonas': ['PIPERACILLIN', 'MEROPENEM', 'CIPROFLOXACIN'],
            'Klebsiella': ['CEFTRIAXONE', 'MEROPENEM']
        }
        for org in organisms:
            if antibiotic in coverage.get(org, []):
                explanation += f"- {antibiotic} provides good coverage for {org}.\n"

    if patient_data.get('pneumonia', 0):
        explanation += "- Patient has pneumonia.\n"
        if antibiotic in ['CEFTRIAXONE', 'LEVOFLOXACIN', 'AZITHROMYCIN']:
            explanation += f"- {antibiotic} is recommended for pneumonia treatment.\n"

    if patient_data.get('uti', 0):
        explanation += "- Patient has UTI.\n"
        if antibiotic in ['CIPROFLOXACIN', 'CEFTRIAXONE']:
            explanation += f"- {antibiotic} is recommended for UTI treatment.\n"

    if patient_data.get('sepsis', 0):
        explanation += "- Patient has sepsis.\n"
        if antibiotic in ['VANCOMYCIN', 'PIPERACILLIN', 'MEROPENEM']:
            explanation += f"- {antibiotic} is recommended for sepsis treatment.\n"

    if patient_data.get('wbc', 10) > 12:
        explanation += f"- Elevated WBC ({patient_data['wbc']:.1f}) suggests infection.\n"
    if patient_data.get('lactate', 1.5) > 2.0:
        explanation += f"- Elevated lactate ({patient_data['lactate']:.1f}) suggests hypoperfusion.\n"
    if patient_data.get('creatinine', 1.0) > 1.5:
        explanation += f"- Elevated creatinine ({patient_data['creatinine']:.1f}) may affect dosing.\n"

    if antibiotic in ['MEROPENEM', 'PIPERACILLIN', 'VANCOMYCIN']:
        explanation += "- This is a broad-spectrum antibiotic.\n"
    else:
        explanation += "- This is a narrower spectrum antibiotic.\n"

    return explanation

def explain_recommendation_v2(antibiotic: str, q_value: float, patient_state: np.ndarray, feature_names: List[str]) -> str:
    """
    Generate a clinical explanation for the antibiotic recommendation based on the patient state and known feature names.

    Args:
        antibiotic (str): Recommended antibiotic.
        q_value (float): Q-value for the recommendation.
        patient_state (np.ndarray): The input feature vector.
        feature_names (List[str]): Corresponding names of each feature in the state.

    Returns:
        str: Explanation string.
    """
    explanation = f"Recommended {antibiotic} (Q-value: {q_value:.4f}).\n"

    # Create feature dictionary from state vector
    patient_features = {name: val for name, val in zip(feature_names, patient_state)}

    # Gender
    if 'gender_F' in patient_features and patient_features['gender_F'] == 1:
        explanation += "- Patient is female.\n"
    elif 'gender_M' in patient_features and patient_features['gender_M'] == 1:
        explanation += "- Patient is male.\n"

    # LOS (length of stay)
    los = patient_features.get('length_of_stay', None)
    if los is not None:
        if los > 7:
            explanation += f"- Prolonged hospital stay (LOS = {los:.1f} days) may indicate a complicated case.\n"
        else:
            explanation += f"- Shorter hospital stay (LOS = {los:.1f} days) suggests stable condition.\n"
    else:
        explanation += "- Length of stay data unavailable.\n"

    # Mortality
    if patient_features.get('mortality', 0) == 1:
        explanation += "- Patient has died in hospital; consider aggressive treatment retrospectively.\n"
    else:
        explanation += "- Patient survived; previous treatment may have been effective.\n"

    # Custom logic: associate specific antibiotics with clinical patterns
    broad_spectrum = ['meropenem', 'piperacillin/tazobactam', 'vancomycin']
    if antibiotic.lower() in broad_spectrum:
        explanation += "- This is a broad-spectrum antibiotic.\n"
        if los is not None and los < 3 and patient_features.get('mortality', 0) == 0:
            explanation += "  Consider de-escalation if patient is improving.\n"
    else:
        explanation += "- This is a narrower-spectrum antibiotic.\n"
        if los is not None and los > 7 and patient_features.get('mortality', 0) == 1:
            explanation += "  Consider escalation if broad coverage was not initially used.\n"

    return explanation
