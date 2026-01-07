import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def load_data(path):
    return pd.read_csv(path)


def preprocess_data(df):
    df_encoded = pd.get_dummies(
        df,
        columns=["gender", "department", "role_level"],
        drop_first=True
    )
    X = df_encoded.drop(columns=["employee_id", "attrition"])
    y = df_encoded["attrition"]
    return X, y


def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression()
    model.fit(X_train_scaled, y_train)

    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    score = roc_auc_score(y_test, y_prob)

    return model, score


if __name__ == "__main__":
    data = load_data("../data/sample_data.csv")
    X, y = preprocess_data(data)
    model, roc_auc = train_model(X, y)
    print(f"Model ROC-AUC: {roc_auc:.2f}")
