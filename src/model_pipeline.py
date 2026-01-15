import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, average_precision_score, RocCurveDisplay

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "sample_data.csv"
ASSETS_DIR = PROJECT_ROOT / "assets"
ASSETS_DIR.mkdir(exist_ok=True)


CAT_COLS = ["gender", "department", "role_level"]
DROP_COLS = ["employee_id", "attrition"]


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df.dropna()


def split_xy(df: pd.DataFrame):
    y = df["attrition"]
    X = df.drop(columns=DROP_COLS)
    num_cols = [c for c in X.columns if c not in CAT_COLS]
    return X, y, num_cols


def make_preprocessor(num_cols):
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CAT_COLS),
            ("num", StandardScaler(), num_cols),
        ]
    )


def evaluate_model(name, model, X_train, y_train, X_test, y_test, preprocessor):
    pipe = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", model),
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_roc = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc")
    cv_pr = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="average_precision")

    pipe.fit(X_train, y_train)

    y_prob = pipe.predict_proba(X_test)[:, 1]
    test_roc = roc_auc_score(y_test, y_prob)
    test_pr = average_precision_score(y_test, y_prob)

    return pipe, {
        "model": name,
        "cv_roc_auc_mean": float(cv_roc.mean()),
        "cv_roc_auc_std": float(cv_roc.std()),
        "cv_pr_auc_mean": float(cv_pr.mean()),
        "cv_pr_auc_std": float(cv_pr.std()),
        "test_roc_auc": float(test_roc),
        "test_pr_auc": float(test_pr),
    }


def save_feature_importance(pipe, num_cols, filename: Path):
    # For Logistic Regression and RandomForest we can extract importance
    model = pipe.named_steps["model"]
    pre = pipe.named_steps["preprocess"]

    # Build feature names
    ohe = pre.named_transformers_["cat"]
    cat_names = list(ohe.get_feature_names_out(CAT_COLS))
    feat_names = cat_names + num_cols

    importances = None
    if hasattr(model, "coef_"):
        importances = model.coef_[0]
    elif hasattr(model, "feature_importances_"):
        importances = model.feature_importances_

    if importances is None:
        return

    df_imp = pd.DataFrame({
        "feature": feat_names,
        "importance": importances
    }).sort_values("importance", ascending=False)

    df_imp.to_csv(filename, index=False)


def plot_roc_curves(pipes, X_test, y_test, outpath: Path):
    plt.figure()
    for name, pipe in pipes.items():
        RocCurveDisplay.from_estimator(pipe, X_test, y_test, name=name)
    plt.savefig(outpath, bbox_inches="tight")
    plt.close()


def main():
    df = load_data(DATA_PATH)
    X, y, num_cols = split_xy(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = make_preprocessor(num_cols)

    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, random_state=42, class_weight="balanced"
        ),
    }

    results = []
    pipes = {}

    for name, model in models.items():
        pipe, metrics = evaluate_model(
            name, model, X_train, y_train, X_test, y_test, preprocessor
        )
        results.append(metrics)
        pipes[name] = pipe

        save_feature_importance(
            pipe, num_cols, ASSETS_DIR / f"{name}_feature_importance.csv"
        )

    plot_roc_curves(pipes, X_test, y_test, ASSETS_DIR / "roc_curve.png")

    summary = {
        "n_rows": int(df.shape[0]),
        "attrition_rate": float(y.mean()),
        "results": results,
    }

    with open(ASSETS_DIR / "results.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Print a clean summary for terminal
    for r in results:
        print(
            f"{r['model']}: "
            f"CV ROC-AUC {r['cv_roc_auc_mean']:.3f}±{r['cv_roc_auc_std']:.3f}, "
            f"Test ROC-AUC {r['test_roc_auc']:.3f}, "
            f"Test PR-AUC {r['test_pr_auc']:.3f}"
        )


if __name__ == "__main__":
    main()
