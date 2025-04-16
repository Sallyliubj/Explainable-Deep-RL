import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from DQL_Agent import DQNAgent
from antibioticEnv import AntibioticEnvironment
import os
from data_loader import DataLoader
from data_preprocessor import preprocess_data
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

def test_agent(agent, env, episodes=100, render=False):
    """
    Test the trained DQN agent on the test environment
    
    Args:
        agent: Trained DQN agent
        env: Test environment
        episodes: Number of test episodes
        render: Whether to render the environment
        
    Returns:
        Dictionary of test metrics and episode data
    """
    rewards = []
    steps_per_episode = []
    correct_prescriptions = 0
    total_prescriptions = 0
    
    # Track predictions and actual values for confusion matrix
    y_true = []
    y_pred = []
    
    # Track action distribution
    action_counts = {action: 0 for action in env.action_space}
    
    # Print initial data distribution
    print("\nTest Data Distribution:")
    prescription_counts = env.data['drug_name_generic'].value_counts()
    total_cases = len(env.data)
    for drug, count in prescription_counts.items():
        percentage = (count / total_cases) * 100
        print(f"{drug}: {count} cases ({percentage:.1f}%)")
    
    # Record detailed episode data
    episode_data = []
    
    # Track Q-value statistics
    q_value_stats = {action: [] for action in env.action_space}
    
    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        step = 0
        
        episode_states = []
        episode_actions = []
        episode_rewards = []
        
        while not done:
            # Agent selects action (no exploration during testing)
            state_2d = np.atleast_2d(state.astype(np.float32))
            q_values = agent.model.predict(state_2d)[0]
            
            # Store Q-values for each action
            for i, action_name in enumerate(env.action_space):
                q_value_stats[action_name].append(q_values[i])
            
            # Print Q-values for every episode
            print(f"\nEpisode {episode+1}, Step {step+1} Q-values:")
            for i, (act, q) in enumerate(zip(env.action_space, q_values)):
                print(f"{act}: {q:.3f}")
            
            # Get prescription indicator
            has_prescription = state[-1]
            print(f"Has prescription: {bool(has_prescription)}")
            
            action = np.argmax(q_values)
            print(f"Selected action: {env.action_space[action]}")
            
            # Track action distribution
            action_counts[env.action_space[action]] += 1
            
            # Record actual and predicted prescriptions
            actual_prescription = env.data.iloc[env.current_idx]['drug_name_generic']
            predicted_prescription = env.action_space[action]
            y_true.append(actual_prescription)
            y_pred.append(predicted_prescription)
            
            # Take action in environment
            next_state, reward, done, info = env.step(action)
            print(f"Reward: {reward:.2f}")
            
            # Record state, action, reward
            episode_states.append(state.copy())
            episode_actions.append(action)
            episode_rewards.append(reward)
            
            # Update metrics
            episode_reward += reward
            step += 1
            
            # Track prescription accuracy if available in info
            if 'correct_prescription' in info:
                total_prescriptions += 1
                if info['correct_prescription']:
                    correct_prescriptions += 1
            
            # Render if requested
            if render:
                env.render()
                
            # Move to next state
            state = next_state
        
        # Store episode data
        episode_data.append({
            'episode': episode,
            'states': np.array(episode_states),
            'actions': np.array(episode_actions),
            'rewards': np.array(episode_rewards),
            'total_reward': episode_reward,
            'steps': step,
            'action_names': [env.action_space[a] for a in episode_actions]
        })
        
        rewards.append(episode_reward)
        steps_per_episode.append(step)
        
        print(f"Episode {episode+1}/{episodes}, Reward: {episode_reward:.2f}, Steps: {step}")
    
    # Calculate confusion matrix
    unique_labels = sorted(list(set(y_true) | set(y_pred)))
    conf_matrix = confusion_matrix(y_true, y_pred, labels=unique_labels)
    
    # Calculate metrics
    metrics = {
        'avg_reward': np.mean(rewards),
        'std_reward': np.std(rewards),
        'avg_steps': np.mean(steps_per_episode),
        'total_episodes': episodes,
        'action_distribution': action_counts,
        'q_value_stats': q_value_stats,
        'confusion_matrix': conf_matrix,
        'confusion_matrix_labels': unique_labels,
        'classification_report': classification_report(y_true, y_pred, labels=unique_labels, output_dict=True)
    }
    
    if total_prescriptions > 0:
        metrics['prescription_accuracy'] = correct_prescriptions / total_prescriptions
    
    return metrics, rewards, episode_data

def plot_rewards(rewards, save_path=None):
    """Plot and optionally save the rewards over episodes"""
    plt.figure(figsize=(12, 6))
    
    # Plot episode rewards
    episodes = range(1, len(rewards) + 1)
    plt.plot(episodes, rewards, 'b-', alpha=0.7, label='Episode Reward')
    
    # Plot moving average
    window_size = 10
    moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
    plt.plot(range(window_size, len(rewards) + 1), moving_avg, 'r--', 
             label=f'{window_size}-Episode Moving Average', linewidth=2)
    
    # Add mean line
    mean_reward = np.mean(rewards)
    plt.axhline(y=mean_reward, color='g', linestyle=':', 
                label=f'Mean Reward: {mean_reward:.2f}', alpha=0.8)
    
    plt.title('Episode Rewards During Testing')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    if save_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    
    # Close the figure to free memory
    plt.close()

def save_episode_data(episode_data, save_dir):
    """Save detailed episode data to files"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Save summary to CSV
    summary_data = []
    for ep in episode_data:
        for step in range(len(ep['rewards'])):
            summary_data.append({
                'episode': ep['episode'],
                'step': step,
                'action': ep['action_names'][step],
                'reward': ep['rewards'][step],
                'total_episode_reward': ep['total_reward']
            })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(f"{save_dir}/episode_summary.csv", index=False)
    
    # Save detailed state-action data
    np.savez(
        f"{save_dir}/detailed_episode_data.npz",
        episode_data=[{
            'episode': ep['episode'],
            'states': ep['states'],
            'actions': ep['actions'],
            'rewards': ep['rewards']
        } for ep in episode_data]
    )

def plot_confusion_matrix(conf_matrix, labels, save_path):
    """Plot and save confusion matrix"""
    plt.figure(figsize=(12, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    # Load test data
    # print("Loading test data...")
    # data_loader = DataLoader(data_path="dataset/mimic-iii-clinical-database-demo-1.4/")
    # test_data = data_loader.load_mimic_data(split='test')
    # test_data = preprocess_data(test_data)

    # use simulated data
    print("Loading simulated test data...")
    test_data = pd.read_csv("dataset/simulated/test_processed.csv")
    
    if test_data.empty:
        raise ValueError("Data loading failed - check simulated data files")
    
    # Create test environment
    print("Creating test environment...")
    test_env = AntibioticEnvironment(test_data)
    
    # Get state and action dimensions from environment
    state_size = test_env.observation_shape
    action_size = len(test_env.action_space)
    
    # Create agent with same architecture as training
    print("Creating agent...")
    agent = DQNAgent(state_size, action_size)
    
    # Load trained model weights
    model_path = "./best_antibiotic_agent.h5"
    if os.path.exists(model_path):
        print(f"Loading trained model from {model_path}")
        agent.model.load_weights(model_path)
    else:
        print(f"Error: Trained model not found at {model_path}")
        return
    
    # Test the agent
    print("Testing agent...")
    metrics, rewards, episode_data = test_agent(agent, test_env, episodes=100)
    
    # Print metrics
    print("\nTest Results:")
    print(f"Average Reward: {metrics['avg_reward']:.2f} ± {metrics['std_reward']:.2f}")
    print(f"Average Steps per Episode: {metrics['avg_steps']:.2f}")
    
    if 'prescription_accuracy' in metrics:
        print(f"Prescription Accuracy: {metrics['prescription_accuracy']*100:.2f}%")
    
    # Create results directory
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    
    # Save initial data distribution
    print("Saving initial data distribution...")
    prescription_counts = test_env.data['drug_name_generic'].value_counts()
    total_cases = len(test_env.data)
    with open(f"{results_dir}/initial_distribution.txt", 'w') as f:
        f.write("Initial Data Distribution:\n")
        for drug, count in prescription_counts.items():
            percentage = (count / total_cases) * 100
            f.write(f"{drug}: {count} cases ({percentage:.1f}%)\n")
    
    # Plot and save confusion matrix
    print("Saving confusion matrix...")
    plot_confusion_matrix(
        metrics['confusion_matrix'],
        metrics['confusion_matrix_labels'],
        f"{results_dir}/confusion_matrix.png"
    )
    
    # Save classification report
    print("Saving classification report...")
    with open(f"{results_dir}/classification_report.txt", 'w') as f:
        f.write("Classification Report\n")
        f.write("====================\n\n")
        for label in metrics['confusion_matrix_labels']:
            report = metrics['classification_report'][label]
            f.write(f"{label}:\n")
            f.write(f"  Precision: {report['precision']:.3f}\n")
            f.write(f"  Recall: {report['recall']:.3f}\n")
            f.write(f"  F1-score: {report['f1-score']:.3f}\n")
            f.write(f"  Support: {report['support']}\n\n")
        
        # Write overall metrics
        f.write("Overall:\n")
        # Handle accuracy separately as it's a float
        f.write(f"Accuracy: {metrics['classification_report']['accuracy']:.3f}\n\n")
        # Handle macro and weighted averages
        for metric in ['macro avg', 'weighted avg']:
            if metric in metrics['classification_report']:
                f.write(f"{metric}:\n")
                for k, v in metrics['classification_report'][metric].items():
                    f.write(f"  {k}: {v:.3f}\n")
                f.write("\n")
    
    # Plot rewards
    print("Plotting rewards...")
    plot_rewards(rewards, save_path=f'{results_dir}/test_rewards_distribution.png')
    
    # Save episode data
    print("Saving episode data...")
    save_episode_data(episode_data, results_dir)
    
    # Save metrics to file
    print("Saving metrics...")
    with open(f"{results_dir}/test_metrics.txt", 'w') as f:
        # Write general metrics
        f.write("General Metrics:\n")
        f.write(f"Average Reward: {metrics['avg_reward']:.2f} ± {metrics['std_reward']:.2f}\n")
        f.write(f"Average Steps per Episode: {metrics['avg_steps']:.2f}\n")
        if 'prescription_accuracy' in metrics:
            f.write(f"Prescription Accuracy: {metrics['prescription_accuracy']*100:.2f}%\n")
        
        # Write action distribution
        f.write("\nAction Distribution:\n")
        total_actions = sum(metrics['action_distribution'].values())
        for action, count in metrics['action_distribution'].items():
            percentage = (count / total_actions) * 100
            f.write(f"{action}: {count} times ({percentage:.1f}%)\n")
        
        # Write Q-value statistics
        f.write("\nQ-value Statistics:\n")
        for action in metrics['q_value_stats']:
            q_values = metrics['q_value_stats'][action]
            f.write(f"{action}:\n")
            f.write(f"  Mean: {np.mean(q_values):.3f}\n")
            f.write(f"  Std: {np.std(q_values):.3f}\n")
            f.write(f"  Min: {np.min(q_values):.3f}\n")
            f.write(f"  Max: {np.max(q_values):.3f}\n")
    
    print(f"Results saved to {results_dir}/")

if __name__ == "__main__":
    main() 