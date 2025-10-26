import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import torch
from typing import Dict, List, Any


def run_shap_analysis(model, X_train, X_test, y_test, feature_names, player_ids):
    """
    Run SHAP analysis on a trained model.

    Args:
        model: The trained model to analyze
        X_train: Training data used to fit the model
        X_test: Test data to analyze
        y_test: True labels for test data
        feature_names: List of feature names
        player_ids: List/array of player IDs aligned with X_test

    Returns:
        dict: Dictionary containing SHAP values and related data
    """
    # Initialize SHAP explainer
    if hasattr(model, 'predict_proba'):
        explainer = shap.KernelExplainer(
            model.predict_proba, shap.sample(X_train, 100))
    else:
        # For PyTorch models
        def f(x):
            with torch.no_grad():
                return model(torch.tensor(x, dtype=torch.float32)).numpy()
        explainer = shap.KernelExplainer(f, shap.sample(X_train.values, 100))

    # Calculate SHAP values for each class
    shap_values = explainer.shap_values(X_test)

    # Get predictions
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(X_test)
    else:
        with torch.no_grad():
            logits = model(torch.tensor(X_test.values, dtype=torch.float32))
            proba = torch.softmax(logits, dim=1).numpy()

    return {
        "shap_values_list": shap_values,
        "proba": proba,
        "feature_names": feature_names
    }


def inspect_player(shap_values_list, X_test, feature_names, proba, y_test, player_ids, target_player=None):
    """
    Analyze SHAP values for a specific player.
    """
    # 1) pick the row index (target player or worst prediction by loss)
    if target_player is not None:
        idxs = np.where(np.asarray(player_ids) == target_player)[0]
        if len(idxs) == 0:
            raise ValueError(f"Player {target_player} not found")
        idx = int(idxs[0])
    else:
        losses = -np.log(proba[np.arange(len(y_test)), y_test.astype(int)])
        idx = int(np.argmax(losses))

    # 2) choose class to explain: predicted class
    pred_class = int(np.argmax(proba[idx]))

    # 3) robustly get the SHAP vector for this row
    sv = shap_values_list
    if isinstance(sv, list):
        # classic multiclass: list length C, each (N, F)
        player_shap = sv[pred_class][idx]
    elif isinstance(sv, np.ndarray):
        if sv.ndim == 2 and sv.shape[0] == len(X_test):
            # (N, F): per-row attributions (e.g., ranked_outputs=1)
            player_shap = sv[idx]
        elif sv.ndim == 3:
            # Try (C, N, F)
            if sv.shape[0] <= 10 and sv.shape[1] == len(X_test):
                player_shap = sv[pred_class, idx, :]
            # Try (N, C, F)  <-- your case: (6, 51, 110)
            elif sv.shape[0] == len(X_test) and sv.shape[2] == len(feature_names):
                player_shap = sv[idx, pred_class, :]
            # Try (N, F, C)
            elif sv.shape[0] == len(X_test) and sv.shape[1] == len(feature_names):
                player_shap = sv[idx, :, pred_class]
            else:
                raise ValueError(f"Unexpected 3D SHAP shape {sv.shape}")
        else:
            raise ValueError(f"Unexpected SHAP shape {sv.shape}")
    else:
        raise TypeError(f"Unsupported SHAP container type: {type(sv)}")
    assert player_shap.shape[0] == len(feature_names), "SHAP vector length mismatch"

    # 4) plot a compact bar chart of top contributions (instead of summary_plot on a single row)
    contrib = pd.DataFrame({
        "feature": feature_names,
        "shap_value": player_shap,
        "abs_val": np.abs(player_shap)
    }).sort_values("abs_val", ascending=False).head(20)

    plt.figure(figsize=(10, 6))
    plt.barh(contrib["feature"][::-1], contrib["shap_value"][::-1])
    plt.xlabel("SHAP contribution (predicted class)")
    plt.title(f"{player_ids[idx]} — row {idx} — pred class {pred_class}")
    plt.tight_layout()

    import os
    os.makedirs('outputs', exist_ok=True)
    outpath = f'outputs/example_player_shap_{str(player_ids[idx]).replace(" ", "_")}.png'
    plt.savefig(outpath, dpi=200)
    plt.close()

    return {
        'player_id': player_ids[idx],
        'row_index': idx,
        'predicted_class': pred_class,
        'true_label': int(y_test[idx]),
        'predicted_proba': proba[idx].tolist(),
        'top_features': contrib[["feature", "shap_value"]].to_dict(orient="records"),
        'plot_path': outpath
    }
