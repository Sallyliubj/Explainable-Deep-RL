import numpy as np
import shap
import tensorflow as tf
import os

from data.data_loader import load_mimic_data, ANTIBIOTICS_LIST
from utils.preprocessing import calculate_age, get_events_before_prescription, get_patient_diagnoses, preprocess_and_save_data
from utils.reward import calculate_reward
from utils.custom_logging import log_dataset_distribution, log_evaluation_step, plot_confusion_matrix, plot_rewards, save_classification_report, plot_training_progress
from models.ppo_agent import PPOAgent
from utils.explainability import analyze_feature_importance, explain_recommendation

def evaluate_ppo(agent, X_test, y_test, prescription_contexts, idx_to_antibiotic, explanation_file_path, feature_importance_path, feature_names):
    """Evaluate the trained PPO agent with detailed logging."""
    print("\nEvaluating PPO model...")
    
    # Log test data distribution
    class_names = [idx_to_antibiotic[i] for i in range(len(idx_to_antibiotic))]
    log_dataset_distribution(y_test, class_names, 'test_logs', phase='test')
    
    predictions = []
    true_labels = []
    all_rewards = []
    
    # Define a model wrapper for SHAP
    def model_wrapper(x):
        x_tensor = tf.convert_to_tensor(x, dtype=tf.float32)
        return agent.network.get_policy(x_tensor).numpy()
    
    # Initialize SHAP explainer with background data
    background_data = X_test[:100]  # Use first 100 samples as background
    explainer = shap.KernelExplainer(model_wrapper, background_data)
    print("SHAP explainer initialized successfully")

    for i, state in enumerate(X_test):
        state_tensor = tf.convert_to_tensor([state], dtype=tf.float32)
        action, probs = agent.get_action(state, training=False)
        predictions.append(action)
        true_labels.append(y_test[i])
        
        predicted_antibiotic = idx_to_antibiotic[action]
        actual_antibiotic = idx_to_antibiotic[y_test[i]]
        
        # Default outcome and patient data
        patient_outcome = {'expire_flag': 0, 'los': 5}
        patient_data = {}
        
        if i < len(prescription_contexts):
            context = prescription_contexts[i]
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
        
        reward = calculate_reward(predicted_antibiotic, actual_antibiotic, patient_outcome, patient_data)
        all_rewards.append(reward)
        
        try:
            # Calculate SHAP values for the predicted class
            shap_values = explainer.shap_values(state.reshape(1, -1))
            
            # Print shapes for debugging
            print(f"SHAP values shape: {[np.array(sv).shape for sv in shap_values]}")
            
            # SHAP values have shape (1, 19, 15) - batch x features x classes
            # Remove batch dimension and transpose to get (15, 19)
            shap_array = np.squeeze(shap_values, axis=0).T  # Remove batch dim and transpose
            # Take absolute values and average across classes
            action_shap_values = np.mean(np.abs(shap_array), axis=0)  # Average across classes to get (19,)
            
            # Get attention weights from the model
            attention_weights = agent.network.get_attention_weights(state_tensor).numpy()[0]  # Shape: (19,)
            
            print(f"Action SHAP values shape: {action_shap_values.shape}")
            print(f"Attention weights shape: {attention_weights.shape}")
            
            # Normalize both metrics to [0,1] scale
            normalized_shap = (action_shap_values - action_shap_values.min()) / (action_shap_values.max() - action_shap_values.min() + 1e-10)
            normalized_attention = (attention_weights - attention_weights.min()) / (attention_weights.max() - attention_weights.min() + 1e-10)
            
            # Calculate hybrid importance as average of normalized metrics
            hybrid_importance = (normalized_shap + normalized_attention) / 2
            
            # Save feature importance metrics
            with open(feature_importance_path, 'a') as f:
                for feat_idx in range(len(hybrid_importance)):
                    f.write(f"{i},{feat_idx},{feature_names[feat_idx]},{normalized_shap[feat_idx]:.6f},{normalized_attention[feat_idx]:.6f},{hybrid_importance[feat_idx]:.6f}\n")
        
        except Exception as e:
            print(f"Warning: Could not calculate feature importance for step {i}: {str(e)}")
            print(f"SHAP values type: {type(shap_values)}")
            if isinstance(shap_values, np.ndarray):
                print(f"SHAP values array shape: {shap_values.shape}")
            continue
        
        # Log evaluation step
        log_evaluation_step(i, action, reward, predicted_antibiotic, actual_antibiotic, 'test_logs')

        # Generate explanation
        try:
            explanation = explain_recommendation(
                recommendation=(predicted_antibiotic, probs[action]),
                patient_data=patient_data
            )
        except Exception as e:
            explanation = f"(Error generating explanation: {e})"

        print(f"\nEvaluation Sample {i + 1}")
        print(f"Predicted: {predicted_antibiotic}, Actual: {actual_antibiotic}")
        print("Explanation:")
        print(explanation)

        # Save explanation to test_logs
        with open(explanation_file_path, 'a') as f:
            f.write(f"{i + 1},{predicted_antibiotic},{actual_antibiotic},{probs[action]:.4f},{explanation.replace(',', ';')}\n")
    
    # Generate evaluation plots and reports
    plot_confusion_matrix(true_labels, predictions, class_names, 'test_logs')
    plot_rewards(all_rewards, 'test_logs')
    save_classification_report(true_labels, predictions, class_names, 'test_logs')
    
    # Calculate metrics
    accuracy = np.mean(np.array(predictions) == y_test)
    avg_reward = np.mean(all_rewards)
    
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Average Reward: {avg_reward:.4f}")
    
    return {
        'accuracy': accuracy,
        'avg_reward': avg_reward,
        'predictions': predictions,
        'rewards': all_rewards
    }


if __name__ == "__main__":
    os.makedirs('test_logs', exist_ok=True)

    # Load data
    state_size, n_actions, X_train, X_test, y_train, y_test, prescription_contexts, idx_to_antibiotic, num_cols, cat_cols, X_original = preprocess_and_save_data()
    
    feature_names = []
    # Add numerical feature names
    feature_names.extend(num_cols)
    # Add encoded categorical feature names
    for col in cat_cols:
        unique_values = X_original[col].unique()
        feature_names.extend([f"{col}_{val}" for val in unique_values])
    
    print(f"Feature names: {feature_names}")
    
    # Initialize the agent
    ppo_agent = PPOAgent(state_size, n_actions)
    
    # Build the model by making a dummy prediction
    dummy_state = np.zeros((1, state_size), dtype=np.float32)
    _ = ppo_agent.network(dummy_state)  # This will build the model
    
    # Now we can load the weights
    print("Loading best accuracy model weights...")
    ppo_agent.network.load_weights('checkpoints/best_accuracy_ppo_model.weights.h5')
    # ppo_agent.network.load_weights('checkpoints/final_ppo_model.weights.h5')

    # Create feature importance CSV file
    feature_importance_path = 'test_logs/feature_importance.csv'
    with open(feature_importance_path, 'w') as f:
        f.write("step,feature_idx,feature_name,shap_value,attention_weight,hybrid_importance\n")

    explanation_file_path = 'test_logs/explanation.csv'
    with open(explanation_file_path, 'w') as explanation_file:
        explanation_file.write("step,predicted_antibiotic,actual_antibiotic,score,explanation\n")

    evaluation_results = evaluate_ppo(ppo_agent, X_test, y_test, prescription_contexts, idx_to_antibiotic, explanation_file_path, feature_importance_path, feature_names)
    
    print("Evaluation finished \n")
    print(f"Accuracy: {evaluation_results['accuracy']:.4f}")
    print(f"Average Reward: {evaluation_results['avg_reward']:.4f}")
