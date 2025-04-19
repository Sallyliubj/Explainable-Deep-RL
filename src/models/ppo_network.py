import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class AttentionLayer(layers.Layer):
    """Custom attention layer for feature importance."""
    
    def __init__(self, units):
        super(AttentionLayer, self).__init__()
        self.W = layers.Dense(units)
        self.V = layers.Dense(1)
    
    def call(self, inputs):
        # Reshape input to (batch_size, feature_dim, 1)
        x = tf.expand_dims(inputs, axis=2)
        
        # Calculate attention scores
        score = self.V(tf.nn.tanh(self.W(x)))
        attention_weights = tf.nn.softmax(score, axis=1)
        
        # Apply attention weights
        context_vector = inputs * tf.squeeze(attention_weights, axis=-1)
        
        return context_vector, attention_weights

class PPONetwork(keras.Model):
    """PPO Network Architecture implementing both actor and critic networks with attention."""
    
    def __init__(self, state_dim, action_dim):
        """
        Initialize the PPO network.
        
        Args:
            state_dim (int): Dimension of the state space
            action_dim (int): Dimension of the action space
        """
        super(PPONetwork, self).__init__()
        
        # Attention layer
        self.attention = AttentionLayer(64)
        
        # Shared layers
        self.shared = keras.Sequential([
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2)
        ])
        
        # Policy network (actor)
        self.policy_net = keras.Sequential([
            layers.Dense(32, activation='relu'),
            layers.Dense(action_dim, activation='softmax')
        ])
        
        # Value network (critic)
        self.value_net = keras.Sequential([
            layers.Dense(32, activation='relu'),
            layers.Dense(1)
        ])
    
    def get_attention_weights(self, state):
        """Get attention weights for feature importance."""
        _, attention_weights = self.attention(state)
        return attention_weights
    
    def call(self, state):
        """Forward pass through the network."""
        # Apply attention
        attended_state, _ = self.attention(state)
        
        # Pass through shared layers
        shared_out = self.shared(attended_state)
        
        # Get policy and value outputs
        return self.policy_net(shared_out), self.value_net(shared_out)
    
    def get_policy(self, state):
        """Get policy (actor) output."""
        attended_state, _ = self.attention(state)
        shared_out = self.shared(attended_state)
        return self.policy_net(shared_out)
    
    def get_value(self, state):
        """Get value (critic) output."""
        attended_state, _ = self.attention(state)
        shared_out = self.shared(attended_state)
        return self.value_net(shared_out) 