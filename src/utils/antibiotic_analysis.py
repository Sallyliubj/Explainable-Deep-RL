import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from typing import Dict, List, Tuple
import os
import logging

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Define antibiotic classes with all possible antibiotics
ANTIBIOTIC_CLASSES = {
    'penicillins': ['AMOXICILLIN', 'PIPERACILLIN', 'AMPICILLIN', 'NAFCILLIN', 'OXACILLIN', 'PENICILLIN'],
    'cephalosporins': ['CEFAZOLIN', 'CEFEPIME', 'CEFTRIAXONE', 'CEFTAZIDIME', 'CEFOTAXIME', 'CEFOXITIN'],
    'fluoroquinolones': ['CIPROFLOXACIN', 'LEVOFLOXACIN', 'MOXIFLOXACIN'],
    'carbapenems': ['MEROPENEM', 'IMIPENEM', 'ERTAPENEM'],
    'glycopeptides': ['VANCOMYCIN'],
    'tetracyclines': ['DOXYCYCLINE', 'TETRACYCLINE', 'MINOCYCLINE'],
    'lincosamides': ['CLINDAMYCIN'],
    'macrolides': ['AZITHROMYCIN', 'ERYTHROMYCIN', 'CLARITHROMYCIN'],
    'aminoglycosides': ['GENTAMICIN', 'TOBRAMYCIN', 'AMIKACIN']
}

def get_antibiotic_class(antibiotic: str) -> str:
    """
    Get the class of an antibiotic.
    
    Args:
        antibiotic: Name of the antibiotic
        
    Returns:
        str: Class name of the antibiotic
    """
    if pd.isna(antibiotic):
        return 'other'
        
    antibiotic = str(antibiotic).upper()
    
    # Check each class for the antibiotic
    for class_name, antibiotics in ANTIBIOTIC_CLASSES.items():
        if antibiotic in antibiotics:
            return class_name
            
        # Check if any antibiotic in the class is a substring of the input
        # This handles cases like "PIPERACILLIN-TAZOBACTAM" matching "PIPERACILLIN"
        for ab in antibiotics:
            if ab in antibiotic:
                return class_name
    
    return 'other'

def analyze_class_performance(evaluation_file: str, output_dir: str) -> Dict:
    """
    Analyze the performance of antibiotic predictions at the class level.
    
    Args:
        evaluation_file: Path to the evaluation CSV file
        output_dir: Directory to save analysis results
        
    Returns:
        Dictionary containing performance metrics
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Reading evaluation data from {evaluation_file}")
        
        # Read evaluation data
        df = pd.read_csv(evaluation_file)
        
        # Log unique antibiotics for debugging
        logger.info("Unique actual antibiotics: %s", df['Actual_Class'].unique())
        logger.info("Unique predicted antibiotics: %s", df['Predicted_Class'].unique())
        
        # Get class labels for actual and predicted antibiotics
        df['actual_class'] = df['Actual_Class'].apply(get_antibiotic_class)
        df['predicted_class'] = df['Predicted_Class'].apply(get_antibiotic_class)
        
        # Log unique classes for debugging
        logger.info("Unique actual classes: %s", df['actual_class'].unique())
        logger.info("Unique predicted classes: %s", df['predicted_class'].unique())
        
        # Calculate matches for each data point
        df['is_match'] = (df['actual_class'] == df['predicted_class']).astype(int)
        total_matches = df['is_match'].sum()
        total_samples = len(df)
        logger.info(f"Total matches: {total_matches} out of {total_samples} samples")
        
        # Create and save class match plot
        logger.info("Generating class match plot")
        plt.figure(figsize=(12, 6))
        plt.plot(df.index, df['is_match'], 'bo-', markersize=4)
        plt.xlabel('Test Point Index')
        plt.ylabel('Class Match (1=Match, 0=Mismatch)')
        plt.title('Antibiotic Class Matches Across Test Dataset')
        plt.grid(True)
        plt.ylim(-0.1, 1.1)  # Add some padding to y-axis
        plt.savefig(f'{output_dir}/class_match_rate.png')
        plt.close()
        
        # Calculate confusion matrix for all data
        logger.info(f"Generating confusion matrix for all {total_samples} samples")
        
        # Get all possible classes in a fixed order
        class_names = sorted(list(ANTIBIOTIC_CLASSES.keys()) + ['other'])
        
        # Create confusion matrix with fixed class order
        conf_matrix = confusion_matrix(
            df['actual_class'],
            df['predicted_class'],
            labels=class_names
        )
        
        # Verify matrix totals
        matrix_sum = conf_matrix.sum()
        if matrix_sum != total_samples:
            logger.warning(f"Confusion matrix sum ({matrix_sum}) doesn't match total samples ({total_samples})")
            # Log actual vs predicted class counts for debugging
            actual_counts = df['actual_class'].value_counts()
            predicted_counts = df['predicted_class'].value_counts()
            logger.info("Actual class counts:\n%s", actual_counts)
            logger.info("Predicted class counts:\n%s", predicted_counts)
        
        # Plot confusion matrix
        plot_class_confusion_matrix(
            conf_matrix,
            class_names,
            f'{output_dir}/confusion_matrix.png'
        )
        
        # Generate classification report
        class_report = classification_report(
            df['actual_class'],
            df['predicted_class'],
            labels=class_names,
            output_dict=True,
            zero_division=0
        )
        
        # Save classification report
        report_df = pd.DataFrame(class_report).transpose()
        report_df.to_csv(f'{output_dir}/classification_report.csv')
        
        # Calculate overall metrics
        metrics = {
            'total_samples': total_samples,
            'total_matches': total_matches,
            'match_rate': total_matches / total_samples,
            'confusion_matrix': conf_matrix.tolist(),
            'classification_report': class_report
        }
        
        # Save detailed results with class mapping
        results_df = df[['Actual_Class', 'Predicted_Class', 'actual_class', 'predicted_class', 'is_match']]
        results_df.to_csv(f'{output_dir}/detailed_class_analysis.csv', index=True)
        
        # Save class mapping for reference
        class_mapping_df = pd.DataFrame([
            {'Antibiotic': ab, 'Class': class_name}
            for class_name, antibiotics in ANTIBIOTIC_CLASSES.items()
            for ab in antibiotics
        ])
        class_mapping_df.to_csv(f'{output_dir}/class_mapping.csv', index=False)
        
        logger.info("Analysis completed successfully")
        return metrics
        
    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}")
        raise

def plot_class_confusion_matrix(conf_matrix: np.ndarray,
                              class_names: List[str],
                              output_path: str):
    """
    Plot a confusion matrix with customized styling.
    
    Args:
        conf_matrix: Confusion matrix array
        class_names: List of class names
        output_path: Path to save the plot
    """
    try:
        plt.figure(figsize=(15, 12))  # Increased figure size for better readability
        
        # Convert matrix to float for proper color scaling
        conf_matrix_float = conf_matrix.astype(float)
        
        # Create annotation text
        annotations = []
        for i in range(conf_matrix.shape[0]):
            row = []
            row_sum = conf_matrix[i].sum()
            for j in range(conf_matrix.shape[1]):
                count = conf_matrix[i, j]
                percentage = (count / row_sum * 100) if row_sum > 0 else 0
                if count > 0:
                    row.append(f'{count}\n({percentage:.1f}%)')
                else:
                    row.append('0')
            annotations.append(row)
        
        # Create heatmap
        ax = plt.gca()
        im = ax.imshow(conf_matrix_float, cmap='YlOrRd')
        
        # Add colorbar
        plt.colorbar(im, label='Number of Samples')
        
        # Add annotations
        for i in range(conf_matrix.shape[0]):
            for j in range(conf_matrix.shape[1]):
                text = plt.text(j, i, annotations[i][j],
                              ha='center', va='center')
        
        # Add labels and title
        plt.xticks(range(len(class_names)), class_names, rotation=45, ha='right')
        plt.yticks(range(len(class_names)), class_names)
        
        total_samples = int(conf_matrix.sum())
        correct_predictions = int(np.diag(conf_matrix).sum())
        accuracy = correct_predictions / total_samples
        
        # Calculate class distribution
        class_dist = []
        for i, name in enumerate(class_names):
            count = conf_matrix[i].sum()
            if count > 0:
                class_dist.append(f"{name}: {int(count)}")
        class_dist_str = "\nClass distribution: " + ", ".join(class_dist)
        
        plt.title(f'Antibiotic Class Confusion Matrix\n' + 
                 f'Total samples: {total_samples}, Accuracy: {accuracy:.2%}\n' +
                 'Values: count (percentage of actual class)' +
                 class_dist_str)
        
        plt.xlabel('Predicted Class')
        plt.ylabel('Actual Class')
        
        # Add gridlines
        ax.set_xticks(np.arange(len(class_names) + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(len(class_names) + 1) - 0.5, minor=True)
        ax.grid(which="minor", color="w", linestyle='-', linewidth=2)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Log detailed analysis
        logger.info(f"Confusion matrix shape: {conf_matrix.shape}")
        logger.info(f"Total samples in matrix: {total_samples}")
        logger.info("Row sums (actual class distribution):")
        for i, name in enumerate(class_names):
            count = conf_matrix[i].sum()
            if count > 0:
                logger.info(f"  {name}: {int(count)} samples")
        logger.info("Column sums (predicted class distribution):")
        for j, name in enumerate(class_names):
            count = conf_matrix[:, j].sum()
            if count > 0:
                logger.info(f"  {name}: {int(count)} predictions")
        
    except Exception as e:
        logger.error(f"Error plotting confusion matrix: {str(e)}")
        raise

def get_class_performance_summary(metrics: Dict) -> str:
    """
    Generate a human-readable summary of class performance.
    
    Args:
        metrics: Dictionary containing performance metrics
        
    Returns:
        String containing performance summary
    """
    try:
        report = metrics['classification_report']
        
        summary = "Performance Summary:\n\n"
        summary += f"Total Samples: {metrics['total_samples']}\n"
        summary += f"Total Matches: {metrics['total_matches']}\n"
        summary += f"Overall Match Rate: {metrics['match_rate']:.2%}\n\n"
        
        summary += "Performance by Class:\n"
        for class_name in ANTIBIOTIC_CLASSES.keys():
            if class_name in report:
                class_metrics = report[class_name]
                summary += f"\n{class_name.capitalize()}:\n"
                summary += f"- Precision: {class_metrics['precision']:.2%}\n"
                summary += f"- Recall: {class_metrics['recall']:.2%}\n"
                summary += f"- F1-Score: {class_metrics['f1-score']:.2%}\n"
                summary += f"- Support: {class_metrics['support']}\n"
        
        return summary
        
    except Exception as e:
        logger.error(f"Error generating performance summary: {str(e)}")
        raise 

if __name__ == "__main__":
    evaluation_file = "test_logs/evaluation_steps.csv"
    output_dir = "test_logs/class_analysis"
    metrics = analyze_class_performance(evaluation_file, output_dir)
    summary = get_class_performance_summary(metrics)
    print(summary)