from flask import Flask, request, jsonify
import numpy as np
from flask_cors import CORS

from models.ppo_agent import PPOAgent
from utils.explainability import explain_recommendation
import tensorflow as tf
import shap
import os

app = Flask(__name__)
CORS(app)

# Updated to match the new feature size
state_size = 19
n_actions = 6
feature_names = [
    'age', 'wbc', 'lactate', 'creatinine',
    'pneumonia', 'uti', 'sepsis', 'skin_soft_tissue', 'intra_abdominal', 'meningitis',
    'los', 'expire_flag',
    'has_staph', 'has_strep', 'has.e.coli', 'has_pseudomonas', 'has_klebsiella',
    'gender_M', 'gender_F'
]

idx_to_antibiotic = {
    0: 'none',
    1: 'piperacillin/tazobactam',
    2: 'vancomycin',
    3: 'cefepime',
    4: 'meropenem',
    5: 'levofloxacin'
}

# Initialize agent and network
ppo_agent = PPOAgent(state_size, n_actions)

# Build ALL model branches
dummy_input = tf.convert_to_tensor(np.zeros((1, state_size), dtype=np.float32))

ppo_agent.network.build(input_shape=(None, state_size))

# Now load weights (no expect_partial!)
ppo_agent.network.load_weights("checkpoints/best_accuracy_ppo_model.weights.h5")


def model_wrapper(x):
    x_tensor = tf.convert_to_tensor(x, dtype=tf.float32)
    return ppo_agent.network.get_policy(x_tensor).numpy()


background = np.random.rand(100, state_size).astype(np.float32)
explainer = shap.KernelExplainer(model_wrapper, background)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        patients = request.json['patients']
        results = []

        for patient in patients:
            gender = patient['gender']
            gender_feat = [1, 0] if gender == 'M' else [0, 1]

            features = [
                           float(patient['age']),
                           float(patient['wbc']),
                           float(patient['lactate']),
                           float(patient['creatinine']),
                           int(patient['pneumonia']),
                           int(patient['uti']),
                           int(patient['sepsis']),
                           int(patient['skin_soft_tissue']),
                           int(patient['intra_abdominal']),
                           int(patient['meningitis']),
                           float(patient['los']),
                           int(patient['expire_flag']),
                           int(patient['has_staph']),
                           int(patient['has_strep']),
                           int(patient['has_e_coli']),
                           int(patient['has_pseudomonas']),
                           int(patient['has_klebsiella'])
                       ] + gender_feat

            state = np.array(features, dtype=np.float32)
            state_tensor = tf.convert_to_tensor([state], dtype=tf.float32)

            action, probs = ppo_agent.get_action(state, training=False)
            q_value = probs[action]
            antibiotic = idx_to_antibiotic[action]

            explanation = explain_recommendation((antibiotic, q_value), patient)

            shap_vals = explainer.shap_values(state.reshape(1, -1))
            shap_array = np.squeeze(shap_vals, axis=0).T
            mean_shap = np.mean(np.abs(shap_array), axis=0)

            attention = ppo_agent.network.get_attention_weights(state_tensor).numpy()[0]
            hybrid = (mean_shap + attention) / 2

            results.append({
                'recommendation': antibiotic,
                'q_value': float(q_value),
                'explanation': explanation,
                'feature_importance': [
                    {
                        'feature': feature_names[i],
                        'shap': float(mean_shap[i]),
                        'attn': float(attention[i]),
                        'hybrid': float(hybrid[i])
                    } for i in range(len(feature_names))
                ]
            })

        return jsonify({'results': results})


    except Exception as e:
        import traceback

        traceback.print_exc()

        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5001)
