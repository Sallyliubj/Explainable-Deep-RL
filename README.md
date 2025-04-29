# Explainable Deep Reinforcement Learning for Antibiotic Prescription

This project implements a Proximal Policy Optimization (PPO) based reinforcement learning system for antibiotic prescription recommendations. The system learns from historical prescription data in the MIMIC-III clinical database and aims to provide explainable recommendations while balancing between effective treatment and antibiotic stewardship.

## 🔧 Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Prepare the MIMIC-III dataset:
   - Download the MIMIC-III clinical database here [https://mimic.mit.edu/](https://mimic.mit.edu/)
   - Place the CSV files in the directory `dataset/`

## 🔭 Usage

1. Train the model:
```bash
python src/train.py
```
This will save training progress and visualizations in the `train_logs/` directory.

2. Evaluate the model:
```bash
python src/evaluate.py
```
This will save evaluation results, explanations, and visualizations in the `test_logs/` directory.

3. Start the recommendation server:
```bash
python src/server.py
```

4. Run the UI (in a separate terminal):
```bash
cd UI/explainable-ppo-ui
npm install
npm run dev
```

## 💾 Pre-trained Models

If you don't want to train the model from scratch, pre-trained models are available in the `checkpoints/` directory:
- `checkpoints/best_accuracy_ppo_model.weights.h5`: Model checkpoint with the highest accuracy
- `checkpoints/final_ppo_model.weights.h5`: Model checkpoint from the end of training

The server and evaluation scripts are configured to use these pre-trained models by default.

## 🗂️ Directory Structure

```
├── src/                          # Core Python code
│   ├── models/                   # Neural network models
│   │   ├── ppo_agent.py          # PPO agent implementation
│   │   └── ppo_network.py        # Neural network architecture
│   ├── utils/                    # Utility functions
│   │   ├── explainability.py     # Explainability modules (SHAP, attention)
│   │   ├── preprocessing.py      # Data preprocessing functions
│   │   ├── reward.py             # Reward function implementation
│   │   ├── antibiotic_analysis.py # Antibiotic-specific analysis
│   │   └── custom_logging.py     # Logging and visualization utilities
│   ├── data/                     # Data loading and processing
│   ├── train.py                  # Training script
│   ├── evaluate.py               # Evaluation script
│   └── server.py                 # API server for UI integration
├── UI/                           # User interface
│   └── explainable-ppo-ui/       # React-based frontend
│       ├── src/                  # UI source code
│       ├── public/               # Static assets
│       └── package.json          # UI dependencies
├── DQN/                          # Baseline implementation using DQN
│   ├── DQL_Agent.py              # DQN agent implementation
│   ├── antibioticEnv.py          # Custom environment for antibiotics
│   ├── train_dql.py              # DQN training script
│   ├── test_dql.py               # DQN evaluation script
│   └── explainability.py         # Basic explainability for DQN
├── dataset/                      # Dataset storage
├── checkpoints/                  # Saved model checkpoints
├── train_logs/                   # Training logs and visualizations
└── test_logs/                    # Evaluation logs and explanations
```

## 💡 Features

- PPO-based reinforcement learning for antibiotic prescription
- Comprehensive reward function considering:
  - Match with historical prescriptions
  - Patient outcomes
  - Organism coverage
  - Antibiotic spectrum appropriateness
  - Clinical guidelines
- Explainable decision-making process through:
  - SHAP values for feature importance
  - Attention weights for interpretability
  - Hybrid importance scoring
  - Natural language explanations
- Integration with MIMIC-III clinical database
- Interactive UI for real-time recommendations with explanations

## 🧩 Model Architecture

The model uses a PPO architecture with:
- Shared layers for feature extraction
- Separate policy (actor) and value (critic) networks
- Gaussian action distribution for continuous action spaces
- Clipped surrogate objective for stable training
- Attention mechanism for improved explainability

## 🎯 Evaluation Metrics

The system is evaluated on:
- Prescription accuracy
- Average reward
- Appropriate coverage rate
- Narrow spectrum usage rate
- Feature importance metrics
- Explainability assessment

**Attribution**

The preprocessing methodologies and reward formula used in this project are adapted from [Kaggle](https://www.kaggle.com/code/yepvaishz/rl-research). The original implementation of double DQN agent provided valuable insights and serves as our benchmark.