import numpy as np

from data.data_loader import load_mimic_data, ANTIBIOTICS_LIST
from utils.preprocessing import calculate_age, get_events_before_prescription, get_patient_diagnoses, preprocess_and_save_data
from utils.reward import calculate_reward
from utils.custom_logging import log_dataset_distribution, log_evaluation_step, plot_confusion_matrix, plot_rewards, save_classification_report, plot_training_progress
from models.ppo_agent import PPOAgent
from utils.explainability import analyze_feature_importance

def evaluate_ppo(agent, X_test, y_test, prescription_contexts, idx_to_antibiotic):
    """Evaluate the trained PPO agent with detailed logging."""
    print("\nEvaluating PPO model...")
    
    # Log test data distribution
    class_names = [idx_to_antibiotic[i] for i in range(len(idx_to_antibiotic))]
    log_dataset_distribution(y_test, class_names, 'test_logs', phase='test')
    
    predictions = []
    true_labels = []
    all_rewards = []
    
    for i, state in enumerate(X_test):
        action, _ = agent.get_action(state, training=False)
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
        
        # Log evaluation step
        log_evaluation_step(i, action, reward, predicted_antibiotic, actual_antibiotic, 'test_logs')
    
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
    # Load data
    state_size, n_actions, X_train, X_test, y_train, y_test, prescription_contexts, idx_to_antibiotic, num_cols, cat_cols, X_original = preprocess_and_save_data()
    
    # Initialize the agent
    ppo_agent = PPOAgent(state_size, n_actions)
    
    # Build the model by making a dummy prediction
    dummy_state = np.zeros((1, state_size), dtype=np.float32)
    _ = ppo_agent.network(dummy_state)  # This will build the model
    
    # Now we can load the weights
    print("Loading best accuracy model weights...")
    ppo_agent.network.load_weights('checkpoints/best_accuracy_ppo_model.weights.h5')
    # ppo_agent.network.load_weights('checkpoints/final_ppo_model.weights.h5')

    evaluation_results = evaluate_ppo(ppo_agent, X_test, y_test, prescription_contexts, idx_to_antibiotic)
    print("Evaluation finished \n")
    print(f"Accuracy: {evaluation_results['accuracy']:.4f}")
    print(f"Average Reward: {evaluation_results['avg_reward']:.4f}")
