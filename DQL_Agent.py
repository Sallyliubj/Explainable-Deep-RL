import numpy as np
import random
from collections import deque
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.models import Sequential

class DQNAgent:
    """Deep Q-Learning agent with experience replay"""
    
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
        
    def build_model(self) -> Sequential:
        """Create neural network architecture"""
        model = Sequential([
            layers.Input(shape=(self.state_size,)),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.1),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.1),
            layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(
            loss='huber',  # More robust than MSE
            optimizer=optimizers.Adam(learning_rate=0.001),
            run_eagerly=True
        )
        return model
    
    def act(self, state: np.ndarray) -> int:
        """Epsilon-greedy action selection with Q-value masking"""
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)
        
        # Ensure state is 2D and float32
        state = np.atleast_2d(state.astype(np.float32))
        q_values = self.model.predict(state)[0]
        
        # Get prescription indicator from state (last feature)
        has_prescription = state[0, -1]
        
        # If patient has prescription, discourage 'none' action by reducing its Q-value
        if has_prescription > 0:
            q_values[0] -= 5.0  # Reduce Q-value for 'none' action
        
        # Add small random noise to break ties
        q_values += np.random.normal(0, 0.01, size=q_values.shape)
        return np.argmax(q_values)
    
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
        current_q_values = self.model.predict(states)
        next_q_values = self.target_model.predict(next_states)
        
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
        
        # Fit the model
        self.model.fit(states, targets, epochs=1, verbose=0)
        
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