from DQL_Agent import DQNAgent
from antibioticEnv import AntibioticEnvironment
import pandas as pd
import logging
from sklearn.model_selection import train_test_split
from data_loader import DataLoader
from data_preprocessor import preprocess_data

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_clinical_agent(
    data: pd.DataFrame,
    episodes: int = 100,
    batch_size: int = 64,
    validation_split: float = 0.2
) -> DQNAgent:
    """Train RL agent with clinical validation"""
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
    
    # Training loop
    best_val_reward = float('-inf')
    patience = 50  # Increased patience
    no_improve_count = 0
    
    for ep in range(episodes):
        state = train_env.reset()
        total_reward = 0
        episode_actions = []
        
        # Multiple steps per episode
        for _ in range(5):
            action = agent.act(state)
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
        
        # Validation and early stopping
        if ep % 10 == 0:
            val_reward = evaluate_agent(agent, val_env)
            # Log action distribution during training
            action_counts = {i: episode_actions.count(i) for i in range(len(train_env.action_space))}
            action_dist = {train_env.action_space[i]: count for i, count in action_counts.items()}
            logger.info(f"Episode {ep}: Train Reward: {total_reward:.1f} | Val Reward: {val_reward:.1f}")
            logger.info(f"Action distribution: {action_dist}")
            
            if val_reward > best_val_reward:
                best_val_reward = val_reward
                no_improve_count = 0
                # Save best model
                agent.model.save("best_antibiotic_agent.h5")
            else:
                no_improve_count += 1
            
            if no_improve_count >= patience:
                logger.info(f"Early stopping at episode {ep}")
                # Load best model before returning
                agent.model.load_weights("best_antibiotic_agent.h5")
                break
    
    return agent


def evaluate_agent(agent: DQNAgent, env: AntibioticEnvironment, trials: int = 100) -> float:
    """Evaluate agent performance"""
    total_reward = 0
    for _ in range(trials):
        state = env.reset()
        action = agent.act(state)
        _, reward, _, _ = env.step(action)
        total_reward += reward
    return total_reward / trials


def main():
    """End-to-end pipeline"""
    try:
        # Data handling
        print("Loading training data...")
        data_loader = DataLoader(data_path="dataset/mimic-iii-clinical-database-demo-1.4/")
        train_data = data_loader.load_mimic_data(split='train')
        train_data = preprocess_data(train_data)
        
        if train_data.empty:
            raise ValueError("Data preprocessing failed - check input data and codes")
        
        # RL training
        agent = train_clinical_agent(train_data)
        
        # Save final model
        agent.model.save("antibiotic_agent.h5")
        logger.info("Training completed and model saved")
        
        return agent
    
    except Exception as e:
        logger.error(f"Main execution failed: {str(e)}")
        raise

if __name__ == "__main__":
    trained_agent = main()