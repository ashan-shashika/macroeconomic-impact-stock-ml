"""
Load saved model results, SHAP values, and models for comparison analysis.
"""

import os
import pickle
import glob


def load_evaluations(base_dir="models"):
    """
    Load all evaluation results.

    Returns
    results : dict
        {model_name: {train_metrics, test_metrics,
                      train_pred, test_pred, feature_cols}}
    """
    results = {}
    results_dir = os.path.join(base_dir, "results")

    for path in sorted(glob.glob(os.path.join(results_dir, "*_results.pkl"))):
        with open(path, "rb") as f:
            data = pickle.load(f)
            results[data["model_name"]] = data

    print(f"Loaded {len(results)} evaluations: {list(results.keys())}")
    return results


def load_shap_values(base_dir="models"):
    """
    Load all SHAP values.

    Returns
    shap_data : dict
        {model_name: {shap_test_df, mean_abs_shap, feature_cols}}
    """
    shap_data = {}
    shap_dir = os.path.join(base_dir, "shap")

    for path in sorted(glob.glob(os.path.join(shap_dir, "*_shap.pkl"))):
        with open(path, "rb") as f:
            data = pickle.load(f)
            shap_data[data["model_name"]] = data

    print(f"Loaded {len(shap_data)} SHAP files: {list(shap_data.keys())}")
    return shap_data


def load_models(base_dir="models"):
    """
    Load all trained model objects.

    Returns
    models : dict
        {model_name: trained_model_object}
    """
    models = {}
    model_dir = os.path.join(base_dir, "trained")

    for path in sorted(glob.glob(os.path.join(model_dir, "*_model.pkl"))):
        with open(path, "rb") as f:
            model = pickle.load(f)
            # extract name from filename: "ridge_model.pkl" → "ridge"
            name = os.path.basename(path).replace("_model.pkl", "")
            models[name] = model

    print(f"Loaded {len(models)} models: {list(models.keys())}")
    return models


def get_y_test(results):
    """
    Safely extract y_test from results.
    Picks the one that matches test_pred size.
    """
    for name, r in results.items():
        if len(r["y_test"]) == len(r["test_pred"]):
            return r["y_test"]

    raise ValueError("No y_test found matching test_pred size")


def load_feature_importances(base_dir="models_sectors", model=None):
    results = {}

    model_dirs = (
        [os.path.join(base_dir, model)]
        if model
        else [d for d in sorted(glob.glob(os.path.join(base_dir, "*")))
              if os.path.isdir(d)]
    )

    for model_dir in model_dirs:
        model_name = os.path.basename(model_dir)
        fi_dir = os.path.join(model_dir, "feature_importance")

        pattern = os.path.join(fi_dir, "*_feature_importance.pkl")
        for path in sorted(glob.glob(pattern)):
            with open(path, "rb") as f:
                data = pickle.load(f)

            # extract sector from filename if not stored in data
            # energy_feature_importance.pkl → "energy"
            sector = data.get("sector") or \
                os.path.basename(path).replace("_feature_importance.pkl", "")

            if model_name not in results:
                results[model_name] = {}
            results[model_name][sector] = data

    print("Loaded feature importances:")
    for m, sectors in results.items():
        print(f"  {m:10s} → {list(sectors.keys())}")

    return results
