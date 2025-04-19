# Explainable Deep Reinforcement Learning for Antibiotic Prescription

This project implements a Proximal Policy Optimization (PPO) based reinforcement learning system for antibiotic prescription recommendations. The system learns from historical prescription data in the MIMIC-III clinical database and aims to provide explainable recommendations while balancing between effective treatment and antibiotic stewardship.

## Setup

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
   - Download the MIMIC-III clinical database
   - Place the CSV files in a directory named `dataset/mimic-iii-clinical-database-demo-1.4/`

## Usage

1. Train the model:
```bash
python src/train.py
```

2. Evaluate the model:
```bash
python src/evaluate.py
```

## Features

- PPO-based reinforcement learning for antibiotic prescription
- Comprehensive reward function considering:
  - Match with historical prescriptions
  - Patient outcomes
  - Organism coverage
  - Antibiotic spectrum appropriateness
  - Clinical guidelines
- Explainable decision-making process
- Integration with MIMIC-III clinical database

## Model Architecture

The model uses a PPO architecture with:
- Shared layers for feature extraction
- Separate policy (actor) and value (critic) networks
- Gaussian action distribution for continuous action spaces
- Clipped surrogate objective for stable training

## Evaluation Metrics

The system is evaluated on:
- Prescription accuracy
- Average reward
- Appropriate coverage rate
- Narrow spectrum usage rate