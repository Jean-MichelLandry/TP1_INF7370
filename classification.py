import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, roc_auc_score
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, AdaBoostClassifier,
    GradientBoostingClassifier, BaggingClassifier
)
from sklearn.inspection import permutation_importance


# ---------------- Utils ---------------- #

def slugify(name: str) -> str:
    return name.lower().replace(" ", "_")


def get_scores(clf, X):
    """Retourne un score continu pour ROC/thresholding."""
    if hasattr(clf, "predict_proba"):
        p = clf.predict_proba(X)
        if p.ndim == 2 and p.shape[1] > 1:
            return p[:, 1]
        return p.ravel()
    if hasattr(clf, "decision_function"):
        s = clf.decision_function(X)
        if s.ndim == 1:
            return s
        return s[:, 1]
    # fallback (pas l’idéal pour AUC, mais au cas où)
    return clf.predict(X)


def compute_feature_importances(clf, X, y, feature_names):
    """Retourne un DataFrame trié des importances (native, coef_, ou permutation)."""
    if hasattr(clf, "feature_importances_"):
        vals = np.asarray(clf.feature_importances_, dtype=float)
    elif hasattr(clf, "coef_"):
        c = clf.coef_.astype(float)
        vals = np.abs(c) if c.ndim == 1 else np.linalg.norm(c, axis=0)
    else:
        pi = permutation_importance(
            clf, X, y, n_repeats=5, random_state=42, scoring="f1"
        )
        vals = pi.importances_mean.astype(float)

    vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
    vals = np.maximum(vals, 0.0)
    s = vals.sum()
    norm = vals / s if s > 0 else vals

    df = pd.DataFrame({
        "feature": feature_names,
        "importance": vals,
        "normalized": norm
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)
    return df


def save_feature_reports(df_imp, model_name, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    slug = slugify(model_name)
    csv_path = os.path.join(out_dir, f"{slug}_features.csv")
    df_imp.to_csv(csv_path, index=False)

    topk = df_imp.head(20)
    plt.figure(figsize=(10, 6))
    plt.barh(topk["feature"][::-1], topk["importance"][::-1])
    plt.xlabel("Importance")
    plt.title(f"Top 20 features — {model_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{slug}_features_top20.png"))
    plt.close()

    print(f"\n=== {model_name} — Top 20 features ===")
    print(topk[["rank", "feature", "importance", "normalized"]].to_string(index=False))


def save_roc_curve(y_true, y_score, model_name, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"AUC={auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title(f"ROC — {model_name}")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{slugify(model_name)}_roc.png"))
    plt.close()
    return auc


# -------------- Évaluation par modèle -------------- #

def evaluate_model(model_name, X_train, y_train, X_test, y_test, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    filenames = {
        "Decision Tree": "decision_tree_results.csv",
        "Random Forest": "random_forest_results.csv",
        "AdaBoost": "adaboost_results.csv",
        "Gradient Boosting": "gradient_boosting_results.csv",
        "Bagging": "bagging_results.csv"
    }

    classifiers = {
        "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=42),

        "Random Forest": RandomForestClassifier(
            n_estimators=250,
            max_depth=10,
            min_samples_leaf=3,
            max_features="sqrt",
            random_state=42
        ),

        "AdaBoost": AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=2, random_state=42),
            n_estimators=250,
            learning_rate=0.2,
            random_state=42
        ),

        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=400,
            learning_rate=0.1,
            max_depth=1,           # arbres faibles (stumps profonds=1)
            random_state=42
        ),

        "Bagging": BaggingClassifier(
            estimator=DecisionTreeClassifier(max_depth=2, random_state=42),
            n_estimators=150,
            random_state=42
        ),
    }

    clf = classifiers[model_name]
    clf.fit(X_train, y_train)

    # Scores test
    y_pred = clf.predict(X_test)
    y_score = get_scores(clf, X_test)
    auc = roc_auc_score(y_test, y_score)

    # Scores train (pour détecter l’overfit)
    y_pred_train = clf.predict(X_train)
    f1_train = f1_score(y_train, y_pred_train, zero_division=0)
    f1_test = f1_score(y_test, y_pred, zero_division=0)

    print(f"[{model_name}] F1 train = {f1_train:.4f} | F1 test = {f1_test:.4f} | Δ = {f1_train - f1_test:+.4f}")

    row = pd.DataFrame([{
        "model": model_name,
        "Exactitude": accuracy_score(y_test, y_pred),
        "Précision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_test,
        "auc": auc
    }])

    # Sauvegardes
    row.to_csv(os.path.join(out_dir, filenames[model_name]), index=False)

    feature_names = X_train.columns if hasattr(X_train, "columns") else [f"f{i}" for i in range(X_train.shape[1])]
    df_imp = compute_feature_importances(clf, X_test, y_test, feature_names)
    save_feature_reports(df_imp, model_name, out_dir)
    save_roc_curve(y_test, y_score, model_name, out_dir)

    return row


# -------------- Agrégation & Viz -------------- #

def save_results(res_df, out_dir="resultats"):
    os.makedirs(out_dir, exist_ok=True)

    print("\n=== Leaderboard (trié par Exactitude) ===")
    print(res_df.to_string(index=False))

    res_df.to_csv(os.path.join(out_dir, "leaderboard.csv"), index=False)

    # barplot des métriques
    plt.figure(figsize=(10, 6))
    res_df.plot(x="model", y=["Exactitude", "Précision", "recall", "f1", "auc"], kind="bar")
    plt.title("Comparaison des modèles")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.xticks(rotation=30, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "comparaison_modeles.png"))
    plt.close()


# -------------- Pipeline principal -------------- #

def classify_data(train_data_file: str):
    df = pd.read_csv(train_data_file)

    drop_cols = {"ItemID", "SentimentText", "Sentiment"}
    feature_cols = [c for c in df.columns if c not in drop_cols]
    X = df[feature_cols]
    y = df["Sentiment"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    model_names = ["Decision Tree", "Random Forest", "AdaBoost", "Gradient Boosting", "Bagging"]

    rows = []
    for name in model_names:
        rows.append(evaluate_model(name, X_train, y_train, X_test, y_test, out_dir="resultats"))

    # Tri par Exactitude (métrique pilote demandée)
    res = pd.concat(rows, ignore_index=True).sort_values("Exactitude", ascending=False).reset_index(drop=True)

    # Affichage du meilleur selon l’exactitude
    best = res.iloc[0]
    print(f"\n>>> Meilleur modèle (Exactitude) : {best['model']} — {best['Exactitude']:.4f}")

    print("\n=== Features utilisées ===")
    print(pd.Series(feature_cols).to_string(index=False))

    save_results(res, out_dir="resultats")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classification de sentiments.")
    parser.add_argument("training_data", type=str, help="Chemin vers le fichier d'entraînement CSV.")
    args = parser.parse_args()
    classify_data(args.training_data)
