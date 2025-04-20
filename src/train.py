import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import tensorflow as tf
import os

from data.data_loader import load_mimic_data, ANTIBIOTICS_LIST
from utils.preprocessing import calculate_age, get_events_before_prescription, get_patient_diagnoses, preprocess_and_save_data
from utils.reward import calculate_reward
from utils.custom_logging import log_dataset_distribution, log_evaluation_step, plot_confusion_matrix, plot_rewards, save_classification_report, plot_training_progress
from models.ppo_agent import PPOAgent
from utils.explainability import analyze_feature_importance

def train_ppo(agent, X_train, y_train, prescription_contexts, idx_to_antibiotic, num_cols, cat_cols, X_original, batch_size=32, episodes=200):
    """Train the PPO agent.
    
    Args:
        agent: PPO agent instance
        X_train: Training features
        y_train: Training labels
        prescription_contexts: List of prescription contexts
        idx_to_antibiotic: Dictionary mapping indices to antibiotic names
        num_cols: List of numerical column names
        cat_cols: List of categorical column names
        X_original: Original DataFrame before preprocessing
        batch_size: Size of training batches
        episodes: Number of training episodes
    """
    print("\nStarting PPO training...")
    
    # Create checkpoints directory if it doesn't exist
    os.makedirs('checkpoints', exist_ok=True)
    
    # Log training data distribution
    class_names = [idx_to_antibiotic[i] for i in range(len(idx_to_antibiotic))]
    log_dataset_distribution(y_train, class_names, 'train_logs')
    
    best_reward = float('-inf')
    best_accuracy = float('-inf')
    episode_rewards = []
    episode_accuracies = []
    
    for episode in range(episodes):
        total_reward = 0
        correct_actions = 0
        
        # Shuffle training data
        indices = np.random.permutation(len(X_train))
        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]
        
        # Collect episode data
        states, actions, rewards, values, probs = [], [], [], [], []
        
        for i in range(0, len(X_shuffled), batch_size):
            batch_states = X_shuffled[i:i+batch_size]
            batch_y = y_shuffled[i:i+batch_size]
            
            for j, state in enumerate(batch_states):
                # Get action from policy
                action, action_probs = agent.get_action(state)
                actual_action = batch_y[j]
                
                # Get patient context and calculate reward
                patient_data = {}
                patient_outcome = {'expire_flag': 0, 'los': 5}
                
                if i + j < len(prescription_contexts):
                    patient_idx = indices[i + j]
                    if patient_idx < len(prescription_contexts):
                        context = prescription_contexts[patient_idx]
                        patient_data = {
                            'has_staph': 1 if 'staph' in str(context.get('organisms', [])).lower() else 0,
                            'has_strep': 1 if 'strep' in str(context.get('organisms', [])).lower() else 0,
                            'has_e.coli': 1 if 'e.coli' in str(context.get('organisms', [])).lower() else 0,
                            'has_pseudomonas': 1 if 'pseudomonas' in str(context.get('organisms', [])).lower() else 0,
                            'has_klebsiella': 1 if 'klebsiella' in str(context.get('organisms', [])).lower() else 0,
                            'pneumonia': context.get('pneumonia', False),
                            'uti': context.get('uti', False),
                            'sepsis': context.get('sepsis', False),
                            'wbc': context.get('wbc', 10),
                            'lactate': context.get('lactate', 1.5),
                            'creatinine': context.get('creatinine', 1.0)
                        }
                        patient_outcome = {
                            'expire_flag': context.get('expire_flag', 0),
                            'los': context.get('los', 5)
                        }
                
                # Calculate reward
                prescribed_antibiotic = idx_to_antibiotic[action]
                actual_antibiotic = idx_to_antibiotic[actual_action]
                reward = calculate_reward(prescribed_antibiotic, actual_antibiotic, patient_outcome, patient_data)
                
                # Store transition
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                values.append(agent.network.get_value(np.array([state], dtype=np.float32))[0][0].numpy())
                probs.append(action_probs)
                
                total_reward += reward
                if action == actual_action:
                    correct_actions += 1
        
        # Convert to numpy arrays and ensure float32 type
        states = np.array(states, dtype=np.float32)
        actions = np.array(actions, dtype=np.int32)
        rewards = np.array(rewards, dtype=np.float32)
        values = np.array(values, dtype=np.float32)
        probs = np.array(probs, dtype=np.float32)
        
        # Compute advantages and returns
        advantages = np.zeros_like(rewards, dtype=np.float32)
        returns = np.zeros_like(rewards, dtype=np.float32)
        gae = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + agent.gamma * next_value - values[t]
            gae = delta + agent.gamma * agent.gae_lambda * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Convert to TensorFlow tensors with correct types
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.int32)
        advantages = tf.convert_to_tensor(advantages, dtype=tf.float32)
        returns = tf.convert_to_tensor(returns, dtype=tf.float32)
        probs = tf.convert_to_tensor(probs, dtype=tf.float32)
        
        # PPO update
        for _ in range(agent.epochs):
            agent.train_step(states, actions, advantages, returns, probs)
        
        # Calculate episode metrics
        accuracy = correct_actions / len(X_shuffled) if len(X_shuffled) > 0 else 0
        episode_rewards.append(total_reward)
        episode_accuracies.append(accuracy)
        
        # Save best model based on reward and accuracy
        if total_reward > best_reward:
            best_reward = total_reward
            agent.network.save_weights('checkpoints/best_reward_ppo_model.weights.h5')
            print(f"Saved best model (reward) at episode {episode+1}")
            
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            agent.network.save_weights('checkpoints/best_accuracy_ppo_model.weights.h5')
            print(f"Saved best model (accuracy) at episode {episode+1}")
        
        # Print episode stats
        print(f"Episode: {episode+1}/{episodes}, Reward: {total_reward:.2f}, Accuracy: {accuracy:.4f}")
        
        # Plot training progress every 10 episodes
        if (episode + 1) % 10 == 0:
            plot_training_progress(episode_rewards, episode_accuracies, 'train_logs')
    
    # Save final model after training
    agent.network.save_weights('checkpoints/final_ppo_model.weights.h5')
    print("\nSaved final model after training completion")
    
    # Final training progress plot
    plot_training_progress(episode_rewards, episode_accuracies, 'train_logs')
    
    # After training loop
    print("\nTraining completed. Analyzing feature importance...")
    
    # Get the actual feature names from the preprocessed data
    feature_names = []
    # Add numerical feature names
    feature_names.extend(num_cols)
    # Add encoded categorical feature names
    for col in cat_cols:
        unique_values = X_original[col].unique()
        feature_names.extend([f"{col}_{val}" for val in unique_values])
    
    print(f"Feature names: {feature_names}")
    
    # Run explainability analysis
    analyze_feature_importance(
        agent=agent,
        states=X_train,
        feature_names=feature_names,
        output_dir='train_logs/explainability'
    )
    
    return agent

# def evaluate_ppo(agent, X_test, y_test, prescription_contexts, idx_to_antibiotic):
#     """Evaluate the trained PPO agent with detailed logging."""
#     print("\nEvaluating PPO model...")
    
#     # Log test data distribution
#     class_names = [idx_to_antibiotic[i] for i in range(len(idx_to_antibiotic))]
#     log_dataset_distribution(y_test, class_names, 'test_logs', phase='test')
    
#     predictions = []
#     true_labels = []
#     all_rewards = []
    
#     for i, state in enumerate(X_test):
#         action, _ = agent.get_action(state, training=False)
#         predictions.append(action)
#         true_labels.append(y_test[i])
        
#         predicted_antibiotic = idx_to_antibiotic[action]
#         actual_antibiotic = idx_to_antibiotic[y_test[i]]
        
#         # Default outcome and patient data
#         patient_outcome = {'expire_flag': 0, 'los': 5}
#         patient_data = {}
        
#         if i < len(prescription_contexts):
#             context = prescription_contexts[i]
#             patient_data = {
#                 'has_staph': 1 if 'staph' in str(context.get('organisms', [])).lower() else 0,
#                 'has_strep': 1 if 'strep' in str(context.get('organisms', [])).lower() else 0,
#                 'has_e.coli': 1 if 'e.coli' in str(context.get('organisms', [])).lower() else 0,
#                 'has_pseudomonas': 1 if 'pseudomonas' in str(context.get('organisms', [])).lower() else 0,
#                 'has_klebsiella': 1 if 'klebsiella' in str(context.get('organisms', [])).lower() else 0,
#                 'pneumonia': context.get('pneumonia', False),
#                 'uti': context.get('uti', False),
#                 'sepsis': context.get('sepsis', False),
#                 'wbc': context.get('wbc', 10),
#                 'lactate': context.get('lactate', 1.5),
#                 'creatinine': context.get('creatinine', 1.0)
#             }
        
#         reward = calculate_reward(predicted_antibiotic, actual_antibiotic, patient_outcome, patient_data)
#         all_rewards.append(reward)
        
#         # Log evaluation step
#         log_evaluation_step(i, action, reward, predicted_antibiotic, actual_antibiotic, 'test_logs')
    
#     # Generate evaluation plots and reports
#     plot_confusion_matrix(true_labels, predictions, class_names, 'test_logs')
#     plot_rewards(all_rewards, 'test_logs')
#     save_classification_report(true_labels, predictions, class_names, 'test_logs')
    
#     # Calculate metrics
#     accuracy = np.mean(np.array(predictions) == y_test)
#     avg_reward = np.mean(all_rewards)
    
#     print(f"Test Accuracy: {accuracy:.4f}")
#     print(f"Average Reward: {avg_reward:.4f}")
    
#     return {
#         'accuracy': accuracy,
#         'avg_reward': avg_reward,
#         'predictions': predictions,
#         'rewards': all_rewards
#     }

def main():

    state_size, n_actions, X_train, X_test, y_train, y_test, prescription_contexts, idx_to_antibiotic, num_cols, cat_cols, X_original = preprocess_and_save_data("dataset/mimic-iii-clinical-database-demo-1.4")
    
    ppo_agent = PPOAgent(state_size, n_actions)
    train_ppo(
        agent=ppo_agent,
        X_train=X_train,
        y_train=y_train,
        prescription_contexts=prescription_contexts,
        idx_to_antibiotic=idx_to_antibiotic,
        num_cols=num_cols,
        cat_cols=cat_cols,
        X_original=X_original,
        batch_size=32,
        episodes=200
    )
    
if __name__ == "__main__":
    main() 