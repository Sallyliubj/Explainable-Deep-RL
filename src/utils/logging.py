import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

def ensure_dir(directory):
    """Ensure directory exists, create if it doesn't."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def log_dataset_distribution(y, class_names, output_dir, phase='train'):
    """Log dataset distribution to a file and create a distribution plot."""
    ensure_dir(output_dir)
    
    # Calculate distribution
    unique, counts = np.unique(y, return_counts=True)
    total = len(y)
    distribution = pd.DataFrame({
        'Class': [class_names[i] for i in unique],
        'Count': counts,
        'Percentage': (counts / total) * 100
    })
    
    # Save distribution to CSV
    distribution.to_csv(f'{output_dir}/{phase}_distribution.csv', index=False)
    
    # Create distribution plot
    plt.figure(figsize=(12, 6))
    sns.barplot(data=distribution, x='Class', y='Count')
    plt.xticks(rotation=45, ha='right')
    plt.title(f'Dataset Distribution ({phase.capitalize()} Set)')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{phase}_distribution.png')
    plt.close()

def log_evaluation_step(step, action, reward, predicted_class, actual_class, output_dir):
    """Log each evaluation step to a file."""
    ensure_dir(output_dir)

    step_data = pd.DataFrame({
        'Step': [step],
        'Action': [action],
        'Reward': [reward],
        'Predicted_Class': [predicted_class],
        'Actual_Class': [actual_class]
    })
    
    log_file = f'{output_dir}/evaluation_steps.csv'
    if not os.path.exists(log_file):
        step_data.to_csv(log_file, index=False)
    else:
        step_data.to_csv(log_file, mode='a', header=False, index=False)

def plot_confusion_matrix(y_true, y_pred, class_names, output_dir):
    """Create and save confusion matrix plot."""
    ensure_dir(output_dir)

    # Get unique classes from both true and predicted labels
    unique_classes = sorted(list(set(np.unique(y_true)) | set(np.unique(y_pred))))
    
    # Create mapping for class indices
    class_indices = {cls: i for i, cls in enumerate(unique_classes)}
    
    # Convert labels to new indices
    y_true_mapped = [class_indices[y] for y in y_true]
    y_pred_mapped = [class_indices[y] for y in y_pred]
    
    # Get class names for actual classes
    actual_class_names = [class_names[i] for i in unique_classes]
    
    cm = confusion_matrix(y_true_mapped, y_pred_mapped)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=actual_class_names,
                yticklabels=actual_class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/confusion_matrix.png')
    plt.close()

def plot_rewards(rewards, output_dir, filename='rewards_plot.png', title='Rewards over Steps'):
    """Create and save rewards plot."""
    ensure_dir(output_dir)
    
    plt.figure(figsize=(10, 6))
    plt.plot(rewards, label='Reward', alpha=0.6)
    plt.plot(pd.Series(rewards).rolling(window=10).mean(), 
             label='Moving Average (10 steps)', linewidth=2)
    plt.title(title)
    plt.xlabel('Step')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{filename}')
    plt.close()

def plot_training_progress(episode_rewards, episode_accuracies, output_dir):
    """Plot and save training progress metrics."""
    ensure_dir(output_dir)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot rewards
    raw_rewards = ax1.plot(episode_rewards, alpha=0.6, color='blue', label='Episode Reward')
    ma_rewards = ax1.plot(pd.Series(episode_rewards).rolling(window=5).mean(), 
                         linewidth=2, color='red', label='Moving Average (5 episodes)')
    ax1.set_title('Training Rewards over Episodes')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')
    
    # Plot accuracies
    raw_acc = ax2.plot(episode_accuracies, alpha=0.6, color='green', label='Episode Accuracy')
    ma_acc = ax2.plot(pd.Series(episode_accuracies).rolling(window=5).mean(), 
                      linewidth=2, color='orange', label='Moving Average (5 episodes)')
    ax2.set_title('Training Accuracy over Episodes')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Accuracy')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper left')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(f'{output_dir}/training_progress.png', dpi=300, bbox_inches='tight')
    plt.close()

def save_classification_report(y_true, y_pred, class_names, output_dir):
    """Save classification report to a file."""
    ensure_dir(output_dir)
    
    # Get unique classes from both true and predicted labels
    unique_classes = sorted(list(set(np.unique(y_true)) | set(np.unique(y_pred))))
    actual_class_names = [class_names[i] for i in unique_classes]
    
    # Log prediction distribution
    pred_unique, pred_counts = np.unique(y_pred, return_counts=True)
    true_unique, true_counts = np.unique(y_true, return_counts=True)
    
    distribution_df = pd.DataFrame({
        'Class': [class_names[i] for i in unique_classes],
        'True_Count': [true_counts[list(true_unique).index(i)] if i in true_unique else 0 for i in unique_classes],
        'Predicted_Count': [pred_counts[list(pred_unique).index(i)] if i in pred_unique else 0 for i in unique_classes]
    })
    distribution_df.to_csv(f'{output_dir}/class_distribution_comparison.csv', index=False)
    
    # Create distribution comparison plot
    plt.figure(figsize=(12, 6))
    x = np.arange(len(unique_classes))
    width = 0.35
    
    plt.bar(x - width/2, distribution_df['True_Count'], width, label='True')
    plt.bar(x + width/2, distribution_df['Predicted_Count'], width, label='Predicted')
    
    plt.xlabel('Classes')
    plt.ylabel('Count')
    plt.title('True vs Predicted Class Distribution')
    plt.xticks(x, [class_names[i] for i in unique_classes], rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{output_dir}/class_distribution_comparison.png')
    plt.close()
    
    # Generate classification report with zero_division parameter
    try:
        report = classification_report(
            y_true, y_pred,
            labels=unique_classes,
            target_names=actual_class_names,
            output_dict=True,
            zero_division=0
        )
        
        # Convert to DataFrame and add support information
        report_df = pd.DataFrame(report).transpose()
        
        # Add class distribution information
        report_df['true_samples'] = [distribution_df.loc[i, 'True_Count'] if i < len(distribution_df) else None 
                                   for i in range(len(report_df))]
        report_df['predicted_samples'] = [distribution_df.loc[i, 'Predicted_Count'] if i < len(distribution_df) else None 
                                        for i in range(len(report_df))]
        
        report_df.to_csv(f'{output_dir}/classification_report.csv')
        
        # Log classes with no predictions
        zero_pred_classes = distribution_df[distribution_df['Predicted_Count'] == 0]
        if not zero_pred_classes.empty:
            with open(f'{output_dir}/zero_prediction_classes.txt', 'w') as f:
                f.write("Classes with no predictions:\n")
                for _, row in zero_pred_classes.iterrows():
                    f.write(f"{row['Class']}: {row['True_Count']} true samples, 0 predictions\n")
    
    except Exception as e:
        # Log any errors that occur during report generation
        with open(f'{output_dir}/classification_report_error.txt', 'w') as f:
            f.write(f"Error generating classification report: {str(e)}\n")
            f.write("\nClass distribution:\n")
            f.write(distribution_df.to_string()) 