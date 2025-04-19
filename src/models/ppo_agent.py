import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import os
from .ppo_network import PPONetwork

class PPOAgent:
    """PPO Agent implementation."""
    
    def __init__(self, state_dim, action_dim, lr_actor=0.0003, lr_critic=0.001, checkpoint_dir='checkpoints'):
        """
        Initialize the PPO agent.
        
        Args:
            state_dim (int): Dimension of the state space
            action_dim (int): Dimension of the action space
            lr_actor (float): Learning rate for the actor network
            lr_critic (float): Learning rate for the critic network
            checkpoint_dir (str): Directory to save model checkpoints
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.clip_ratio = 0.2
        self.epochs = 10
        self.checkpoint_dir = checkpoint_dir
        
        # Create checkpoint directory if it doesn't exist
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        
        self.network = PPONetwork(state_dim, action_dim)
        self.optimizer_actor = tf.keras.optimizers.Adam(learning_rate=lr_actor)
        self.optimizer_critic = tf.keras.optimizers.Adam(learning_rate=lr_critic)
        
        # Initialize checkpoint manager
        self.checkpoint = tf.train.Checkpoint(
            network=self.network,
            optimizer_actor=self.optimizer_actor,
            optimizer_critic=self.optimizer_critic
        )
        self.checkpoint_manager = tf.train.CheckpointManager(
            self.checkpoint, self.checkpoint_dir, max_to_keep=5
        )
    
    def save_model(self, episode=None):
        """Save model checkpoint."""
        if episode is not None:
            path = self.checkpoint_manager.save(checkpoint_number=episode)
        else:
            path = self.checkpoint_manager.save()
        return path
    
    def load_model(self, checkpoint_path=None):
        """Load model checkpoint."""
        if checkpoint_path is None:
            latest_checkpoint = self.checkpoint_manager.latest_checkpoint
            if latest_checkpoint:
                status = self.checkpoint.restore(latest_checkpoint)
                print(f"Restored from checkpoint: {latest_checkpoint}")
                return True
        else:
            status = self.checkpoint.restore(checkpoint_path)
            print(f"Restored from checkpoint: {checkpoint_path}")
            return True
        return False
    
    def get_action(self, state, training=True):
        """
        Get action from the policy network.
        
        Args:
            state: Current state
            training (bool): Whether to sample from the policy or take the most probable action
            
        Returns:
            tuple: (action, action probabilities)
        """
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        probs = self.network.get_policy(state)[0]
        
        if training:
            dist = tfp.distributions.Categorical(probs=probs)
            action = dist.sample()
            return int(action.numpy()), probs.numpy()
        else:
            return np.argmax(probs.numpy()), probs.numpy()
    
    @tf.function
    def train_step(self, states, actions, advantages, returns, old_probs):
        """
        Perform a single training step.
        
        Args:
            states: Batch of states
            actions: Batch of actions
            advantages: Batch of advantages
            returns: Batch of returns
            old_probs: Batch of old action probabilities
            
        Returns:
            tuple: (policy loss, value loss)
        """
        with tf.GradientTape() as tape_actor, tf.GradientTape() as tape_critic:
            # Get current policy and values
            new_probs, values = self.network(states)
            values = tf.squeeze(values)
            
            # Get probabilities of actions taken
            actions_one_hot = tf.one_hot(actions, self.action_dim)
            new_probs_actions = tf.reduce_sum(new_probs * actions_one_hot, axis=1)
            old_probs_actions = tf.reduce_sum(old_probs * actions_one_hot, axis=1)
            
            # Compute ratio and clipped ratio
            ratio = new_probs_actions / old_probs_actions
            clipped_ratio = tf.clip_by_value(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
            
            # Compute losses
            policy_loss = -tf.reduce_mean(
                tf.minimum(ratio * advantages, clipped_ratio * advantages)
            )
            value_loss = tf.reduce_mean(tf.square(returns - values))
            
        # Compute and apply gradients
        actor_grads = tape_actor.gradient(policy_loss, self.network.policy_net.trainable_variables)
        critic_grads = tape_critic.gradient(value_loss, self.network.value_net.trainable_variables)
        
        self.optimizer_actor.apply_gradients(
            zip(actor_grads, self.network.policy_net.trainable_variables)
        )
        self.optimizer_critic.apply_gradients(
            zip(critic_grads, self.network.value_net.trainable_variables)
        )
        
        return policy_loss, value_loss 