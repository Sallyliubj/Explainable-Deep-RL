import numpy as np
import random
from collections import deque
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, Multiply, Activation, Layer

class AttentionLayer(Layer):
    """Custom attention layer to highlight important features"""
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        
    def build(self, input_shape):
        # Single weight vector for feature importance
        self.W = self.add_weight(name='attention_weight',
                               shape=(input_shape[-1],),  # One weight per feature
                               initializer='ones',  # Start with equal attention
                               constraint=tf.keras.constraints.NonNeg(),  # Ensure positive weights
                               trainable=True)
        super(AttentionLayer, self).build(input_shape)
        
    def call(self, x):
        # Apply softplus to ensure positive weights with a smooth gradient
        attention_weights = tf.nn.softplus(self.W)
        
        # Add L1 regularization to encourage sparsity while maintaining distribution
        l1_reg = 0.01 * tf.reduce_sum(tf.abs(attention_weights))
        self.add_loss(l1_reg)
        
        # Normalize weights to sum to 1
        attention_weights = attention_weights / tf.reduce_sum(attention_weights)
        
        # Reshape for broadcasting
        attention_weights = tf.reshape(attention_weights, (1, -1))
        
        # Apply attention weights to input
        attended_features = x * attention_weights
        return attended_features, attention_weights
    
    def compute_output_shape(self, input_shape):
        return [(input_shape[0], input_shape[1]), (input_shape[0], input_shape[1])]

class DQNAgent:
    """Deep Q-Learning agent with experience replay and attention mechanism"""
    
    def __init__(self, state_size: int, action_size: int):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.tau = 0.001  # Soft update parameter
        
        # Create models
        self.model = self.build_model()
        self.target_model = tf.keras.models.clone_model(self.model)
        self.target_model.set_weights(self.model.get_weights())
        
    def build_model(self) -> models.Model:
        """Create neural network architecture with attention"""
        # Input layer
        inputs = Input(shape=(self.state_size,))
        
        # Apply attention directly to input features
        attended_features, attention_weights = AttentionLayer()(inputs)
        
        # First dense layer
        x = Dense(128, activation='relu')(attended_features)
        x = Dropout(0.1)(x)
        
        # Second dense layer
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.1)(x)
        
        # Output layer
        outputs = Dense(self.action_size, activation='linear')(x)
        
        # Create model
        model = models.Model(inputs=inputs, outputs=[outputs, attention_weights])
        model.compile(
            loss=['huber', None],  # Only compute loss for Q-values
            optimizer=optimizers.Adam(learning_rate=0.001),
            run_eagerly=True
        )
        return model
    
    def act(self, state: np.ndarray) -> tuple:
        """Epsilon-greedy action selection with attention weights"""
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size), None
        
        # Ensure state is 2D and float32
        state = np.atleast_2d(state.astype(np.float32))
        q_values, attention_weights = self.model.predict(state)
        q_values = q_values[0]
        attention_weights = attention_weights[0]
        
        # Get prescription indicator from state (last feature)
        has_prescription = state[0, -1]
        
        # If patient has prescription, discourage 'none' action by reducing its Q-value
        if has_prescription > 0:
            q_values[0] -= 5.0  # Reduce Q-value for 'none' action
        
        # Add small random noise to break ties
        q_values += np.random.normal(0, 0.01, size=q_values.shape)
        return np.argmax(q_values), attention_weights
    
    def remember(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        """Store experience in replay buffer"""
        # Ensure states are 1D arrays
        state = np.asarray(state).flatten()
        next_state = np.asarray(next_state).flatten()
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self, batch_size: int):
        """Train on past experiences"""
        if len(self.memory) < batch_size:
            return
        
        # Sample mini-batch from memory
        minibatch = random.sample(self.memory, batch_size)
        
        # Extract states, ensure float32 type and reshape
        states = np.array([x[0] for x in minibatch], dtype=np.float32)
        next_states = np.array([x[3] for x in minibatch], dtype=np.float32)
        
        # Ensure states have correct shape (batch_size, state_size)
        states = states.reshape(batch_size, self.state_size)
        next_states = next_states.reshape(batch_size, self.state_size)
        
        # Extract other components
        actions = np.array([x[1] for x in minibatch])
        rewards = np.array([x[2] for x in minibatch], dtype=np.float32)
        dones = np.array([x[4] for x in minibatch], dtype=np.float32)
        
        # Predict Q-values for current and next states
        current_q_values, _ = self.model.predict(states)
        next_q_values, _ = self.target_model.predict(next_states)
        
        # Compute target Q values with prescription masking
        targets = current_q_values.copy()
        
        for i in range(batch_size):
            # If done, use reward. If not, use reward + discounted max Q
            if dones[i]:
                targets[i][actions[i]] = rewards[i]
            else:
                # Apply prescription masking to next state Q-values
                if next_states[i, -1] > 0:  # If patient has prescription
                    next_q_values[i, 0] -= 5.0  # Reduce Q-value for 'none' action
                targets[i][actions[i]] = rewards[i] + self.gamma * np.max(next_q_values[i])
        
        # Fit the model with dummy attention weights (not used in loss)
        dummy_attention = np.zeros((batch_size, 1))  # Dummy target for attention
        self.model.fit(states, [targets, dummy_attention], epochs=1, verbose=0)
        
        # Soft update target network
        self._update_target_network()
        
        # Decay exploration
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def _update_target_network(self):
        """Soft update target network weights"""
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        
        for i in range(len(weights)):
            target_weights[i] = self.tau * weights[i] + (1 - self.tau) * target_weights[i]
        
        self.target_model.set_weights(target_weights)
        
    def get_feature_importance(self, state: np.ndarray) -> dict:
        """Get feature importance scores using attention weights"""
        state = np.atleast_2d(state.astype(np.float32))
        _, attention_weights = self.model.predict(state)
        attention_weights = attention_weights[0].flatten()
        
        # Normalize attention weights to sum to 1
        attention_weights = np.abs(attention_weights)  # ensure positive weights
        attention_weights = attention_weights / np.sum(attention_weights)  # normalize to sum to 1
        
        # Map attention weights to feature names
        feature_names = [
            'gender_M', 'gender_F', 'gender_OTHER',
            'length_of_stay', 'mortality',
            'has_prescription'
        ]
        
        return dict(zip(feature_names, attention_weights))