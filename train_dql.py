from DQL_Agent import DQNAgent
from antibioticEnv import AntibioticEnvironment
import pandas as pd
import logging
from sklearn.model_selection import train_test_split
from data_loader import DataLoader
from data_preprocessor import preprocess_data
import os
import numpy as np
import matplotlib.pyplot as plt
from explainability import ExplainableAgent
import seaborn as sns

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def plot_training_feature_importance(feature_importance_history, save_path):
    """Plot feature importance evolution during training"""
    # Set style
    sns.set_style("whitegrid")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Set color palette
    colors = sns.color_palette("husl", len(feature_importance_history[0].keys()))
    
    # Plot importance for each feature
    for i, feature in enumerate(feature_importance_history[0].keys()):
        values = [d[feature] for d in feature_importance_history]
        ax.plot(values, label=feature, color=colors[i], linewidth=2, marker='o', markersize=4)
    
    # Customize plot
    ax.set_title('Feature Importance Evolution During Training', fontsize=14, pad=20)
    ax.set_xlabel('Training Steps (x10)', fontsize=12)
    ax.set_ylabel('Importance Score', fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.grid(True, alpha=0.3)
    
    # Add legend with better placement
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', 
             title='Features', fontsize=10, title_fontsize=12)
    
    # Add horizontal line at mean importance
    mean_importance = 1.0 / len(feature_importance_history[0])
    ax.axhline(y=mean_importance, color='gray', linestyle='--', alpha=0.5,
               label='Mean Importance')
    
    # Add annotations for final values
    last_values = feature_importance_history[-1]
    for i, (feature, value) in enumerate(last_values.items()):
        ax.annotate(f'{value:.3f}', 
                   xy=(len(feature_importance_history)-1, value),
                   xytext=(5, 0), textcoords='offset points',
                   ha='left', va='center', fontsize=9,
                   color=colors[i])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

def train_clinical_agent(
    data: pd.DataFrame,
    episodes: int = 100,
    batch_size: int = 64,
    validation_split: float = 0.2
) -> DQNAgent:
    """Train RL agent with clinical validation and explainability tracking"""
    # Validate input data
    if data.empty:
        raise ValueError("Input data is empty. Cannot train agent.")
    
    # Log data distribution
    print("\nTraining Data Distribution:")
    prescription_counts = data['drug_name_generic'].value_counts()
    total_cases = len(data)
    for drug, count in prescription_counts.items():
        percentage = (count / total_cases) * 100
        print(f"{drug}: {count} cases ({percentage:.1f}%)")
    
    # Split data
    train_data, val_data = train_test_split(data, test_size=validation_split)
    
    # Create environments
    train_env = AntibioticEnvironment(train_data)
    val_env = AntibioticEnvironment(val_data)
    
    # Validate environment properties
    if not hasattr(train_env, 'observation_shape'):
        raise AttributeError("Environment must have 'observation_shape' attribute")
    
    if not hasattr(train_env, 'action_space'):
        raise AttributeError("Environment must have 'action_space' attribute")
    
    # Initialize agent with slower exploration decay
    agent = DQNAgent(
        state_size=train_env.observation_shape, 
        action_size=len(train_env.action_space)
    )
    
    # Create explainable agent wrapper
    explainable_agent = ExplainableAgent(agent)
    explainable_agent.initialize_explainer(train_env.features)
    
    # Training loop
    best_val_reward = float('-inf')
    patience = 50  # Increased patience
    no_improve_count = 0
    
    # Track feature importance history
    feature_importance_history = []
    
    # Create checkpoints directory
    checkpoint_dir = './checkpoints/simulated_explainable'
    os.makedirs(checkpoint_dir, exist_ok=True)

    
    for ep in range(episodes):
        state = train_env.reset()
        total_reward = 0
        episode_actions = []
        episode_states = []
        
        # Multiple steps per episode
        for _ in range(5):
            # Store state for analysis
            episode_states.append(state.copy())
            
            # Get action with attention weights
            action, attention_weights = agent.act(state)
            next_state, reward, done, _ = train_env.step(action)
            agent.remember(state, action, reward, next_state, done)
            total_reward += reward
            episode_actions.append(action)
            
            if done:
                state = train_env.reset()
            else:
                state = next_state
        
        # Experience replay with multiple updates
        if len(agent.memory) >= batch_size:
            for _ in range(4):
                agent.replay(batch_size)
        
        # Validation and feature importance analysis
        if ep % 10 == 0:
            val_reward = evaluate_agent(agent, val_env)
            
            # Analyze feature importance
            episode_states = np.array(episode_states)
            feature_importance = explainable_agent.explain_batch(episode_states)
            feature_importance_history.append(feature_importance)
            
            # Log metrics
            action_counts = {i: episode_actions.count(i) for i in range(len(train_env.action_space))}
            action_dist = {train_env.action_space[i]: count for i, count in action_counts.items()}
            logger.info(f"Episode {ep}: Train Reward: {total_reward:.1f} | Val Reward: {val_reward:.1f}")
            logger.info(f"Action distribution: {action_dist}")
            logger.info("Feature Importance:")
            for feature, importance in feature_importance.items():
                logger.info(f"  {feature}: {importance:.3f}")
            
            # Save feature importance plot
            plot_training_feature_importance(
                feature_importance_history,
                save_path=f"{checkpoint_dir}/feature_importance_history.png"
            )
            
            if val_reward > best_val_reward:
                best_val_reward = val_reward
                no_improve_count = 0
                # Save best model
                agent.model.save(f"{checkpoint_dir}/best_antibiotic_agent.h5")
                
                # Save feature importance data
                np.save(
                    f"{checkpoint_dir}/feature_importance_history.npy",
                    feature_importance_history
                )
            else:
                no_improve_count += 1
            
            if no_improve_count >= patience:
                logger.info(f"Early stopping at episode {ep}")
                # Load best model before returning
                agent.model.load_weights(f"{checkpoint_dir}/best_antibiotic_agent.h5")
                break
    
    return agent


def evaluate_agent(agent: DQNAgent, env: AntibioticEnvironment, trials: int = 100) -> float:
    """Evaluate agent performance"""
    total_reward = 0
    for _ in range(trials):
        state = env.reset()
        action, _ = agent.act(state)  # Now returns action and attention weights
        _, reward, _, _ = env.step(action)
        total_reward += reward
    return total_reward / trials


def main():
    """End-to-end pipeline"""
    try:
        # Data handling
        # print("Loading training data...")
        # data_loader = DataLoader(data_path="dataset/mimic-iii-clinical-database-demo-1.4/")
        # train_data = data_loader.load_mimic_data(split='train')
        # train_data = preprocess_data(train_data)
        
        # use simulated data
        print("Loading simulated training data...")
        train_data = pd.read_csv("dataset/simulated/train_processed.csv")
        
        if train_data.empty:
            raise ValueError("Data preprocessing failed - check input data and codes")
        
        # Inspect and save dataset structure
        print("\nProcessed Dataset Structure:")
        print("Shape:", train_data.shape)
        print("\nColumns and their types:")
        for col in train_data.columns:
            print(f"{col}: {train_data[col].dtype}")
            
        print("\nSample of non-null values for each column:")
        for col in train_data.columns:
            sample_val = train_data[col].dropna().iloc[0] if not train_data[col].empty else "No non-null values"
            print(f"{col} example: {sample_val}")
            
        # Save dataset structure to file
        os.makedirs('checkpoints', exist_ok=True)
        with open('checkpoints/simulated_explainable/dataset_structure.txt', 'w') as f:
            f.write("Processed Dataset Structure\n")
            f.write("========================\n\n")
            f.write(f"Total rows: {train_data.shape[0]}\n")
            f.write(f"Total columns: {train_data.shape[1]}\n\n")
            f.write("Columns and their types:\n")
            f.write("------------------------\n")
            for col in train_data.columns:
                f.write(f"{col}: {train_data[col].dtype}\n")
            
            f.write("\nColumn Statistics:\n")
            f.write("------------------\n")
            for col in train_data.columns:
                f.write(f"\n{col}:\n")
                f.write(f"  Null values: {train_data[col].isnull().sum()}\n")
                if train_data[col].dtype in ['int64', 'float64']:
                    f.write(f"  Mean: {train_data[col].mean():.2f}\n")
                    f.write(f"  Std: {train_data[col].std():.2f}\n")
                    f.write(f"  Min: {train_data[col].min()}\n")
                    f.write(f"  Max: {train_data[col].max()}\n")
                elif train_data[col].dtype == 'object':
                    f.write(f"  Unique values: {train_data[col].nunique()}\n")
                    f.write("  Top 5 values:\n")
                    for val, count in train_data[col].value_counts().head().items():
                        f.write(f"    {val}: {count}\n")
        
        # RL training
        agent = train_clinical_agent(train_data)
        
        # Save final model
        agent.model.save("checkpoints/simulated_explainable/antibiotic_agent.h5")
        logger.info("Training completed and model saved")
        
        return agent
    
    except Exception as e:
        logger.error(f"Main execution failed: {str(e)}")
        raise

if __name__ == "__main__":
    trained_agent = main()