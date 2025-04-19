import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class MultiHeadAttentionLayer(layers.Layer):
    """Multi-head attention layer with entropy regularization for improved interpretability."""
    
    def __init__(self, num_heads=4, units_per_head=16, entropy_reg_weight=0.1):
        super(MultiHeadAttentionLayer, self).__init__()
        self.num_heads = num_heads
        self.units_per_head = units_per_head
        self.entropy_reg_weight = entropy_reg_weight
        
        # Create attention heads
        self.attention_heads = []
        for _ in range(num_heads):
            head = {
                'W': layers.Dense(units_per_head),
                'V': layers.Dense(1)
            }
            self.attention_heads.append(head)
        
        # Output projection
        self.output_projection = layers.Dense(units_per_head * num_heads)
    
    def compute_entropy(self, attention_weights):
        """Compute entropy of attention weights to encourage diversity."""
        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        entropy = -tf.reduce_sum(attention_weights * tf.math.log(attention_weights + epsilon), axis=1)
        return tf.reduce_mean(entropy)
    
    def call(self, inputs, training=None):
        batch_size = tf.shape(inputs)[0]
        
        # Process each attention head
        head_outputs = []
        head_weights = []
        
        for head in self.attention_heads:
            # Project inputs through ReLU instead of tanh
            x = tf.expand_dims(inputs, axis=2)
            projection = head['W'](x)
            projection = tf.nn.relu(projection)  # Using ReLU activation
            
            # Calculate attention scores
            score = head['V'](projection)
            attention_weights = tf.nn.softmax(score, axis=1)
            
            # Apply attention weights
            context = inputs * tf.squeeze(attention_weights, axis=-1)
            head_outputs.append(context)
            head_weights.append(tf.squeeze(attention_weights, axis=-1))
        
        # Concatenate all head outputs
        combined_output = tf.concat(head_outputs, axis=-1)
        combined_output = self.output_projection(combined_output)
        
        # Average attention weights across heads
        avg_attention_weights = tf.reduce_mean(tf.stack(head_weights, axis=1), axis=1)
        
        if training:
            # Add entropy regularization loss
            entropy = self.compute_entropy(avg_attention_weights)
            entropy_loss = -self.entropy_reg_weight * entropy  # Negative because we want to maximize entropy
            self.add_loss(entropy_loss)
        
        return combined_output, avg_attention_weights

class PPONetwork(keras.Model):
    """PPO Network Architecture implementing both actor and critic networks with multi-head attention."""
    
    def __init__(self, state_dim, action_dim):
        """
        Initialize the PPO network.
        
        Args:
            state_dim (int): Dimension of the state space
            action_dim (int): Dimension of the action space
        """
        super(PPONetwork, self).__init__()
        
        # Multi-head attention layer
        self.attention = MultiHeadAttentionLayer()
        
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