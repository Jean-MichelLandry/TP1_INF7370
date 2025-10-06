import argparse
import os
import pandas as pd
from sklearn.base import clone
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, BaggingClassifier
from features_extraction import extract_features

MODELS = {
    "decision_tree": DecisionTreeClassifier(),
    "random_forest": RandomForestClassifier(),
    "adaboost": AdaBoostClassifier(),
    "gradient_boosting": GradientBoostingClassifier(),
    "bagging": BaggingClassifier(),
}

def evaluation(train_features_path, test_csv, out_dir, model_key):
    os.makedirs(out_dir, exist_ok=True)


    test_feat_path = os.path.join(out_dir, "test_data.csv")
    extract_features(test_csv, test_feat_path)

    df_train = pd.read_csv(train_features_path, encoding="utf-8", encoding_errors="replace", index_col="ItemID")
    df_test  = pd.read_csv(test_feat_path,       encoding="utf-8", encoding_errors="replace", index_col="ItemID")

    drop_cols = {"SentimentText","Sentiment"}   
    feature_cols = sorted((set(df_train.columns) & set(df_test.columns)) - drop_cols)

    X_train = df_train[feature_cols]
    y_train = df_train["Sentiment"]
    X_test  = df_test[feature_cols]

    clf = clone(MODELS[model_key])
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)


    pred_df = pd.DataFrame(
        {
            "SentimentText": df_test["SentimentText"] if "SentimentText" in df_test.columns else "",
            "PredictedSentiment": y_pred.astype(int),
        },
        index=df_test.index,
    )
    pred_df.index.name = "ItemID"
    pred_df.to_csv(os.path.join(out_dir, "test_predictions.csv"), index=True, index_label="ItemID")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ajoute PredictedSentiment aux tweets du test (ItemID en index).")
    parser.add_argument("train_features_path")
    parser.add_argument("test_csv")
    parser.add_argument("out_dir")
    parser.add_argument("model", choices=list(MODELS.keys()))
    args = parser.parse_args()
    evaluation(args.train_features_path, args.test_csv, args.out_dir, args.model)
