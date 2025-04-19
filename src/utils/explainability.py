import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import tensorflow as tf
from typing import List, Dict, Any
import os

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
                       n_samples: int = 100):
    """
    Analyze and visualize SHAP values for feature importance.
    
    Args:
        agent: PPO agent instance
        states: Input states to analyze
        feature_names: List of feature names
        output_dir: Directory to save outputs
        n_samples: Number of background samples for SHAP
    """
    print(f"Starting SHAP analysis with {len(states)} states and {len(feature_names)} features")
    print(f"State shape: {states.shape}")
    
    # Create a wrapper function for the model that returns policy logits
    def model_predict(x):
        if isinstance(x, pd.DataFrame):
            x = x.values
        return agent.network.get_policy(tf.convert_to_tensor(x, dtype=tf.float32)).numpy()
    
    # Sample background data
    background_data = states[np.random.choice(len(states), min(n_samples, len(states)), replace=False)]
    print(f"Using {len(background_data)} background samples")
    
    try:
        # Initialize SHAP explainer
        explainer = shap.KernelExplainer(model_predict, background_data)
        
        # Calculate SHAP values
        print("Calculating SHAP values...")
        shap_values = explainer.shap_values(states[:n_samples])
        
        # For multi-class output, aggregate SHAP values across all classes and ensure it's 1D
        if isinstance(shap_values, list):
            print(f"Multi-class output detected with {len(shap_values)} classes")
            # Take the mean across classes first
            shap_values = np.abs(np.array(shap_values)).mean(axis=0)
            print(f"Shape after class aggregation: {shap_values.shape}")
            # Then take mean across samples
            shap_values = shap_values.mean(axis=0)
            print(f"Shape after sample aggregation: {shap_values.shape}")
        else:
            print("Single-class output detected")
            shap_values = np.abs(shap_values).mean(axis=0)
            print(f"Shape after aggregation: {shap_values.shape}")
        
        # Ensure the number of SHAP values matches the number of features
        if len(shap_values) != len(feature_names):
            print(f"Warning: SHAP values dimension ({len(shap_values)}) doesn't match feature names ({len(feature_names)})")
            # If we have more SHAP values than features, take the first len(feature_names) values
            if len(shap_values) > len(feature_names):
                print("Truncating SHAP values to match feature count")
                shap_values = shap_values[:len(feature_names)]
            else:
                # If we have fewer SHAP values, this indicates a problem
                raise ValueError(f"Number of SHAP values ({len(shap_values)}) is less than number of features ({len(feature_names)})")
        
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
        
    except Exception as e:
        print(f"Error during SHAP analysis: {str(e)}")
        print("Skipping SHAP analysis...")

def analyze_feature_importance(agent,
                             states: np.ndarray,
                             feature_names: List[str],
                             output_dir: str):
    """
    Analyze feature importance using both attention weights and SHAP values.
    
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
    analyze_attention_weights(agent, states, feature_names, output_dir)
    
    # Analyze SHAP values
    print("\nCalculating SHAP values...")
    analyze_shap_values(agent, states, feature_names, output_dir)
    
    print("\nFeature importance analysis completed. Results saved in:", output_dir) 