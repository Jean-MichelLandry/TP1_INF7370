import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, BaggingClassifier

def evaluate_model(model_name, X_train, y_train, X_test, y_test, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    filenames = {
        "Decision Tree": "decision_tree_results.csv",
        "Random Forest": "random_forest_results.csv",
        "AdaBoost": "adaboost_results.csv",
        "Gradient Boosting": "gradient_boosting_results.csv",
        "Bagging": "bagging_results.csv"
    }
    builders = {
        "Decision Tree": DecisionTreeClassifier,
        "Random Forest": RandomForestClassifier,
        "AdaBoost": AdaBoostClassifier,
        "Gradient Boosting": GradientBoostingClassifier,
        "Bagging": BaggingClassifier
    }
    clf = builders[model_name]()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    row = pd.DataFrame([{
        "model": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred)
    }])
    row.to_csv(os.path.join(out_dir, filenames[model_name]), index=False)
    return row

def save_results(res_df, out_dir="resultats"):
    os.makedirs(out_dir, exist_ok=True)
    print(res_df.to_string(index=False))
    res_df.to_csv(os.path.join(out_dir, "leaderboard.csv"), index=False)
    plt.figure(figsize=(10,6))
    res_df.plot(x="model", y=["accuracy","precision","recall","f1"], kind="bar")
    plt.title("Comparaison des modèles")
    plt.ylabel("Score")
    plt.ylim(0,1)
    plt.xticks(rotation=30, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "comparaison_modeles.png"))

def classify_data(train_data_file:str):
    df = pd.read_csv(train_data_file)
    drop_cols = {"ItemID","SentimentText","Sentiment"}
    feature_cols = [c for c in df.columns if c not in drop_cols]
    X = df[feature_cols]
    y = df["Sentiment"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
    model_names = ["Decision Tree","Random Forest","AdaBoost","Gradient Boosting","Bagging"]
    rows = []
    for name in model_names:
        rows.append(evaluate_model(name, X_train, y_train, X_test, y_test, out_dir="resultats"))
    res = pd.concat(rows, ignore_index=True).sort_values("f1", ascending=False).reset_index(drop=True)
    print("Features utilisées:", feature_cols)
    save_results(res, out_dir="resultats")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classification de sentiments.")
    parser.add_argument("training_data", type=str, help="Chemin vers le fichier d'entraînement CSV.")
    args = parser.parse_args()
    classify_data(args.training_data)
