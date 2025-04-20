import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import tensorflow as tf
from typing import List, Dict, Any
import os
import warnings
warnings.filterwarnings("ignore")

def plot_feature_importance(importance_values: np.ndarray,
                          feature_names: List[str],
                          title: str,
                          output_path: str):
    """
    Plot feature importance values as a bar chart using seaborn.
    
    Args:
        importance_values: Array of importance values
        feature_names: List of feature names
        title: Plot title
        output_path: Path to save the plot
    """
    # Ensure importance_values is 1D
    importance_values = np.asarray(importance_values).flatten()
    
    # Safety check for matching dimensions
    if len(importance_values) != len(feature_names):
        print(f"Warning: Mismatch between importance_values ({len(importance_values)}) and feature_names ({len(feature_names)})")
        # Adjust feature_names or importance_values to match
        if len(importance_values) > len(feature_names):
            importance_values = importance_values[:len(feature_names)]
        else:
            feature_names = feature_names[:len(importance_values)]
    
    # Create DataFrame for seaborn
    df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance_values
    })
    
    # Sort by importance value
    df = df.sort_values('Importance', ascending=True)
    
    # Set up the matplotlib figure with extra width for legend
    plt.figure(figsize=(14, 8))
    
    # Create color palette
    n_features = len(feature_names)
    colors = sns.color_palette("husl", n_features)
    
    # Create horizontal bar plot
    ax = sns.barplot(data=df,
                    y='Feature',
                    x='Importance',
                    palette=colors,
                    orient='h')
    
    # Customize the plot
    plt.title(title, fontsize=14, pad=20)
    plt.xlabel('Importance Score', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    
    # Rotate x-axis labels if needed
    plt.xticks(rotation=0)
    
    # Add grid for better readability
    ax.grid(True, axis='x', linestyle='--', alpha=0.7)
    
    # Create legend with feature names
    legend_handles = [plt.Rectangle((0,0),1,1, color=colors[i]) for i in range(len(df))]
    plt.legend(legend_handles, df['Feature'], 
              title="Features",
              bbox_to_anchor=(1.05, 1),
              loc='upper left',
              borderaxespad=0.)
    
    # Adjust layout to prevent label cutoff and accommodate legend
    plt.tight_layout()
    
    # Save the plot with extra space for legend
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def analyze_attention_weights(agent,
                            states: np.ndarray,
                            feature_names: List[str],
                            output_dir: str):
    """
    Analyze and visualize attention weights from the network.
    
    Args:
        agent: PPO agent instance
        states: Input states to analyze
        feature_names: List of feature names
        output_dir: Directory to save outputs
    """
    # Get attention weights for all states
    attention_weights = []
    for state in states:
        state_tensor = tf.convert_to_tensor([state], dtype=tf.float32)
        weights = agent.network.get_attention_weights(state_tensor)
        attention_weights.append(weights.numpy().squeeze())
    
    # Average attention weights across all states and ensure it's 1D
    mean_attention = np.mean(attention_weights, axis=0).flatten()

    os.makedirs(output_dir, exist_ok=True)
    
    # Plot attention-based feature importance
    plot_feature_importance(
        mean_attention,
        feature_names,
        'Feature Importance Based on Attention Weights',
        f'{output_dir}/attention_feature_importance.png'
    )
    
    # Save attention weights to CSV
    attention_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': mean_attention
    })
    attention_df.to_csv(f'{output_dir}/attention_weights.csv', index=False)

def analyze_shap_values(agent,
                       states: np.ndarray,
                       feature_names: List[str],
                       output_dir: str,
                       n_samples: int = 100) -> np.ndarray:
    """
    Analyze and visualize SHAP values for feature importance.
    
    Args:
        agent: PPO agent instance
        states: Input states to analyze
        feature_names: List of feature names
        output_dir: Directory to save outputs
        n_samples: Number of background samples for SHAP
        
    Returns:
        np.ndarray: Processed SHAP values with shape (n_features,)
    """
    print(f"Starting SHAP analysis with {len(states)} states and {len(feature_names)} features")
    print(f"State shape: {states.shape}")
    
    # Create a wrapper function for the model that returns policy logits
    def model_predict(x):
        if isinstance(x, pd.DataFrame):
            x = x.values
        x_tensor = tf.convert_to_tensor(x, dtype=tf.float32)
        return agent.network.get_policy(x_tensor).numpy()
    
    # Sample background data
    background_data = states[np.random.choice(len(states), min(n_samples, len(states)), replace=False)]
    print(f"Using {len(background_data)} background samples")
    
    try:
        # Initialize SHAP explainer
        explainer = shap.KernelExplainer(model_predict, background_data)
        
        # Calculate SHAP values
        print("Calculating SHAP values...")
        shap_values = explainer.shap_values(states[:n_samples])
        
        print(f"Initial SHAP values shape/type: {type(shap_values)}")
        if isinstance(shap_values, list):
            print(f"Multi-class output detected with {len(shap_values)} classes")
            # Convert list to array and take absolute mean across classes and samples
            shap_array = np.array(shap_values)
            shap_values = np.mean(np.abs(shap_array), axis=0)  # Mean across classes
            shap_values = np.mean(shap_values, axis=0)         # Mean across samples
        else:
            print(f"Single output with shape: {shap_values.shape}")
            if len(shap_values.shape) == 3:  # Shape: (samples, features, actions)
                print("Taking mean across samples and actions...")
                shap_values = np.mean(np.abs(shap_values), axis=0)  # Mean across samples
                shap_values = np.mean(shap_values, axis=1)          # Mean across actions
            elif len(shap_values.shape) > 1:
                shap_values = np.mean(np.abs(shap_values), axis=0)  # Mean across samples
        
        print(f"Shape after aggregation: {shap_values.shape}")
        
        # Ensure the number of SHAP values matches the number of features
        if len(shap_values) != len(feature_names):
            print(f"Warning: SHAP values dimension ({len(shap_values)}) doesn't match feature names ({len(feature_names)})")
            if len(shap_values) > len(feature_names):
                print("Truncating SHAP values to match feature count")
                shap_values = shap_values[:len(feature_names)]
            else:
                # If we have fewer SHAP values, pad with zeros
                shap_values = np.pad(shap_values,
                                   (0, len(feature_names) - len(shap_values)),
                                   'constant')
        
        print(f"Final SHAP values shape: {shap_values.shape}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Plot SHAP-based feature importance
        plot_feature_importance(
            shap_values,
            feature_names,
            'Feature Importance Based on SHAP Values',
            f'{output_dir}/shap_feature_importance.png'
        )
        
        # Save SHAP values to CSV
        shap_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': shap_values
        })
        shap_df.to_csv(f'{output_dir}/shap_values.csv', index=False)
        
        return shap_values
        
    except Exception as e:
        print(f"Error during SHAP analysis: {str(e)}")
        print("Skipping SHAP analysis...")
        return None

def analyze_feature_importance(agent,
                             states: np.ndarray,
                             feature_names: List[str],
                             output_dir: str):
    """
    Analyze feature importance using attention weights, SHAP values, and a hybrid metric.
    
    Args:
        agent: PPO agent instance
        states: Input states to analyze
        feature_names: List of feature names
        output_dir: Directory to save outputs
    """
    print("\nAnalyzing feature importance...")
    print(f"Input shapes - States: {states.shape}, Number of features: {len(feature_names)}")
    
    # Analyze attention weights
    print("\nCalculating attention-based feature importance...")
    attention_weights = []
    for state in states:
        state_tensor = tf.convert_to_tensor([state], dtype=tf.float32)
        weights = agent.network.get_attention_weights(state_tensor)
        # Ensure weights are flattened properly
        weights_flat = weights.numpy().flatten()
        if len(weights_flat) > len(feature_names):
            weights_flat = weights_flat[:len(feature_names)]
        attention_weights.append(weights_flat)
    
    # Average attention weights across all states
    mean_attention = np.mean(attention_weights, axis=0)
    
    # Ensure mean_attention matches feature_names length
    if len(mean_attention) > len(feature_names):
        mean_attention = mean_attention[:len(feature_names)]
    elif len(mean_attention) < len(feature_names):
        # Pad with zeros if needed
        mean_attention = np.pad(mean_attention, 
                              (0, len(feature_names) - len(mean_attention)),
                              'constant')
    
    # Save and plot attention weights
    analyze_attention_weights(agent, states, feature_names, output_dir)
    
    # Calculate SHAP values
    print("\nCalculating SHAP values...")
    shap_values = analyze_shap_values(agent, states, feature_names, output_dir)
    
    if shap_values is not None:
        # Calculate hybrid importance
        print("\nCalculating hybrid feature importance...")
        
        # Verify shapes before normalization
        print(f"Shape check - Attention: {mean_attention.shape}, SHAP: {shap_values.shape}")
        
        # Normalize both metrics to [0, 1] scale
        norm_attention = (mean_attention - mean_attention.min()) / (mean_attention.max() - mean_attention.min() + 1e-10)
        norm_shap = (shap_values - shap_values.min()) / (shap_values.max() - shap_values.min() + 1e-10)
        
        # Calculate hybrid importance as the average of normalized scores
        hybrid_importance = (norm_attention + norm_shap) / 2
        
        # Plot hybrid importance
        plot_feature_importance(
            hybrid_importance,
            feature_names,
            'Hybrid Feature Importance (Attention + SHAP)',
            f'{output_dir}/hybrid_feature_importance.png'
        )
        
        # Save hybrid importance to CSV
        hybrid_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': hybrid_importance,
            'Normalized_Attention': norm_attention,
            'Normalized_SHAP': norm_shap
        })
        hybrid_df.to_csv(f'{output_dir}/hybrid_importance.csv', index=False)
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame({
            'Feature': feature_names,
            'Attention_Importance': mean_attention,
            'SHAP_Importance': shap_values,
            'Hybrid_Importance': hybrid_importance
        })
        comparison_df.to_csv(f'{output_dir}/importance_comparison.csv', index=False)
    
    print("\nFeature importance analysis completed. Results saved in:", output_dir)

def explain_recommendation(recommendation, patient_data):
    """
    Generate a clinical explanation for the antibiotic recommendation.

    Args:
        recommendation (tuple): (antibiotic_name, q_value or score placeholder).
        patient_data (dict): Dictionary containing patient context features.

    Returns:
        str: Explanation string.
    """
    antibiotic, q_value = recommendation
    explanation = f"Recommended {antibiotic} (Score: {q_value:.4f}).\n"

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
            if antibiotic.upper() in coverage.get(org, []):
                explanation += f"- {antibiotic} provides good coverage for {org}.\n"

    if patient_data.get('pneumonia', 0):
        explanation += "- Patient has pneumonia.\n"
        if antibiotic.upper() in ['CEFTRIAXONE', 'LEVOFLOXACIN', 'AZITHROMYCIN']:
            explanation += f"- {antibiotic} is recommended for pneumonia treatment.\n"

    if patient_data.get('uti', 0):
        explanation += "- Patient has UTI.\n"
        if antibiotic.upper() in ['CIPROFLOXACIN', 'CEFTRIAXONE']:
            explanation += f"- {antibiotic} is recommended for UTI treatment.\n"

    if patient_data.get('sepsis', 0):
        explanation += "- Patient has sepsis.\n"
        if antibiotic.upper() in ['VANCOMYCIN', 'PIPERACILLIN', 'MEROPENEM']:
            explanation += f"- {antibiotic} is recommended for sepsis treatment.\n"

    if patient_data.get('wbc', 10) > 12:
        explanation += f"- Elevated WBC ({patient_data['wbc']:.1f}) suggests infection.\n"
    if patient_data.get('lactate', 1.5) > 2.0:
        explanation += f"- Elevated lactate ({patient_data['lactate']:.1f}) suggests hypoperfusion.\n"
    if patient_data.get('creatinine', 1.0) > 1.5:
        explanation += f"- Elevated creatinine ({patient_data['creatinine']:.1f}) may affect antibiotic dosing.\n"

    if antibiotic.upper() in ['MEROPENEM', 'PIPERACILLIN', 'VANCOMYCIN']:
        explanation += "- This is a broad-spectrum antibiotic.\n"
    else:
        explanation += "- This is a narrower-spectrum antibiotic.\n"

    return explanation
