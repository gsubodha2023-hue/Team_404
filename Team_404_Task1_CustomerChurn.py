

# # Section 1: Imports and configuration
import os
import glob
import numpy as np
import pandas as pd
from IPython.display import display

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

TARGET_COLUMN = "Churn"
TRAIN_PATH = "/content/dataset-1.csv"
EVALUATION_SPLIT_PATH = "./evaluation_split.csv"
RANDOM_STATE = 42

print("Imports completed successfully.")

# ==============================================================================

# Section 2: Locate and load dataset
def find_training_file():
    candidates = [
        TRAIN_PATH,
        "./dataset-1.csv",
        "/content/ dataset-1.csv",
    ]

    for path in candidates:
        if os.path.exists(path):
            return path

    matches = glob.glob("/content/*dataset-1*.csv") + glob.glob("./*dataset-1*.csv")
    if matches:
        return matches[0]

    return None

actual_train_path = find_training_file()

if actual_train_path is None:
    raise FileNotFoundError(
        "dataset-1.csv was not found. In Google Colab, upload dataset-1.csv "
        "using the Files panel and then run this cell again."
    )

df = pd.read_csv(actual_train_path)
df.columns = df.columns.str.strip()

print("Loaded:", actual_train_path)
print("Dataset shape:", df.shape)
print("Columns:", df.columns.tolist())
display(df.head())

if TARGET_COLUMN not in df.columns:
    raise KeyError(f"Target column '{TARGET_COLUMN}' was not found in the dataset.")

print("\nRaw target distribution:")
print(df[TARGET_COLUMN].value_counts(dropna=False))

# ==============================================================================

# Section 3: Data cleaning
def data_cleaning(data):
    data = data.copy()
    data.columns = data.columns.str.strip()
    data = data.drop_duplicates().reset_index(drop=True)

    # customerID is an identifier, not a useful predictive feature
    if "customerID" in data.columns:
        data = data.drop(columns=["customerID"])

    # Telco TotalCharges may contain blank strings
    if "TotalCharges" in data.columns:
        data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")

    # Strip spaces from text values while preserving missing values
    object_cols = data.select_dtypes(include=["object", "string"]).columns
    for col in object_cols:
        data[col] = data[col].map(lambda x: x.strip() if isinstance(x, str) else x)

    # Robustly convert target to 0/1 when target is present
    if TARGET_COLUMN in data.columns:
        target = data[TARGET_COLUMN]

        if pd.api.types.is_numeric_dtype(target):
            data[TARGET_COLUMN] = pd.to_numeric(target, errors="coerce")
        else:
            normalized = target.astype("string").str.strip().str.lower()
            target_map = {
                "yes": 1, "no": 0,
                "true": 1, "false": 0,
                "1": 1, "0": 0
            }
            mapped = normalized.map(target_map)

            if mapped.isna().any():
                bad_values = sorted(normalized[mapped.isna()].dropna().unique().tolist())
                raise ValueError(
                    f"Unsupported values found in '{TARGET_COLUMN}': {bad_values}. "
                    "Expected Yes/No or 1/0 values."
                )

            data[TARGET_COLUMN] = mapped.astype(int)

        if data[TARGET_COLUMN].isna().any():
            raise ValueError(f"Missing or invalid values found in target column '{TARGET_COLUMN}'.")

        data[TARGET_COLUMN] = data[TARGET_COLUMN].astype(int)

    return data

cleaned_df = data_cleaning(df)

print("Original shape:", df.shape)
print("Cleaned shape:", cleaned_df.shape)
print("Duplicate rows after cleaning:", cleaned_df.duplicated().sum())
print("\nMissing values:")
print(cleaned_df.isnull().sum())
print("\nCleaned target distribution:")
print(cleaned_df[TARGET_COLUMN].value_counts())

# ==============================================================================

# Section 4: Feature engineering
def feature_engineering(data):
    data = data.copy()

    if "TotalCharges" in data.columns and "tenure" in data.columns:
        tenure_safe = pd.to_numeric(data["tenure"], errors="coerce").replace(0, 1)
        data["AvgMonthlySpend"] = data["TotalCharges"] / tenure_safe

    service_columns = [
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies"
    ]
    available_services = [c for c in service_columns if c in data.columns]

    if available_services:
        data["NumServices"] = (data[available_services] == "Yes").sum(axis=1)

    return data


def scale_features(data):
    # Scaling is intentionally handled later by StandardScaler inside the pipeline.
    return data.copy()


def preprocess_data(data):
    data = data_cleaning(data)
    data = feature_engineering(data)
    data = scale_features(data)
    return data

processed_df = preprocess_data(df)
print("Processed shape:", processed_df.shape)
display(processed_df.head())

# ==============================================================================

# Section 5: Train / validation / test split
def split_data(data):
    data = data.copy()

    train_val, test_split = train_test_split(
        data,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=data[TARGET_COLUMN]
    )

    # 25% of remaining 80% = 20% of full data
    train_split, validation_split = train_test_split(
        train_val,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=train_val[TARGET_COLUMN]
    )

    return train_split, validation_split, test_split

train_split, validation_split, test_split = split_data(processed_df)

print("Full dataset:", processed_df.shape)
print("Training set:", train_split.shape)
print("Validation set:", validation_split.shape)
print("Test set:", test_split.shape)

# ==============================================================================

# Section 6: Separate features and labels
X_train = train_split.drop(columns=[TARGET_COLUMN])
y_train = train_split[TARGET_COLUMN]

X_val = validation_split.drop(columns=[TARGET_COLUMN])
y_val = validation_split[TARGET_COLUMN]

X_test = test_split.drop(columns=[TARGET_COLUMN])
y_test = test_split[TARGET_COLUMN]

print("X_train:", X_train.shape, " y_train:", y_train.shape)
print("X_val  :", X_val.shape, " y_val  :", y_val.shape)
print("X_test :", X_test.shape, " y_test :", y_test.shape)

# ==============================================================================

# Section 7: Identify numeric and categorical columns
numeric_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = X_train.select_dtypes(include=["object", "string", "category", "bool"]).columns.tolist()

print("Numerical features:")
print(numeric_features)
print("\nCategorical features:")
print(categorical_features)

# ==============================================================================

# Section 8: Build preprocessing transformer
numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ],
    remainder="drop"
)

print("Preprocessor created successfully.")

# ==============================================================================

# Section 9: Evaluation helper
def evaluate_predictions(y_true, y_pred, y_proba):
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1 Score": f1_score(y_true, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, y_proba)
    }

    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    return metrics

# ==============================================================================

# Section 10: Train Logistic Regression
logistic_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE))
    ]
)

logistic_model.fit(X_train, y_train)
print("Logistic Regression model trained successfully.")

log_val_pred = logistic_model.predict(X_val)
log_val_proba = logistic_model.predict_proba(X_val)[:, 1]
print("\nLogistic Regression - Validation")
evaluate_predictions(y_val, log_val_pred, log_val_proba)

# ==============================================================================

# Section 11: Train Random Forest
rf_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=300,
                max_depth=8,
                min_samples_split=10,
                min_samples_leaf=4,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1
            )
        )
    ]
)

rf_model.fit(X_train, y_train)
print("Random Forest model trained successfully.")

rf_val_pred = rf_model.predict(X_val)
rf_val_proba = rf_model.predict_proba(X_val)[:, 1]
print("\nRandom Forest - Validation")
evaluate_predictions(y_val, rf_val_pred, rf_val_proba)

# ==============================================================================

# Section 12: Compare models and select the best by Validation ROC-AUC
models = {
    "Logistic Regression": logistic_model,
    "Random Forest": rf_model
}

results = []

for name, current_model in models.items():
    train_pred = current_model.predict(X_train)
    train_proba = current_model.predict_proba(X_train)[:, 1]

    val_pred = current_model.predict(X_val)
    val_proba = current_model.predict_proba(X_val)[:, 1]

    results.append({
        "Model": name,
        "Train Accuracy": accuracy_score(y_train, train_pred),
        "Train F1": f1_score(y_train, train_pred, zero_division=0),
        "Train ROC-AUC": roc_auc_score(y_train, train_proba),
        "Validation Accuracy": accuracy_score(y_val, val_pred),
        "Validation Precision": precision_score(y_val, val_pred, zero_division=0),
        "Validation Recall": recall_score(y_val, val_pred, zero_division=0),
        "Validation F1": f1_score(y_val, val_pred, zero_division=0),
        "Validation ROC-AUC": roc_auc_score(y_val, val_proba)
    })

results_df = pd.DataFrame(results).sort_values("Validation ROC-AUC", ascending=False).reset_index(drop=True)
display(results_df)

best_model_name = results_df.loc[0, "Model"]
model = models[best_model_name]
print("Selected model:", best_model_name)

# ==============================================================================

# Section 13: Final prediction function
def predict_churn(new_data):
    """
    Accepts raw customer data (without requiring the Churn column).
    Returns: (class_predictions, churn_probabilities)
    """
    processed = preprocess_data(new_data)

    if TARGET_COLUMN in processed.columns:
        processed = processed.drop(columns=[TARGET_COLUMN])

    predictions = model.predict(processed)
    probabilities = model.predict_proba(processed)[:, 1]
    return predictions, probabilities

print("predict_churn() is ready.")

# ==============================================================================

# Section 14: Final held-out test results
# X_test is already cleaned/engineered, but predict_churn is intentionally robust
# to either raw or processed input.
test_predictions, test_probabilities = predict_churn(X_test)

print("Final Held-Out Test Results")
test_metrics = evaluate_predictions(y_test, test_predictions, test_probabilities)
display(pd.DataFrame([test_metrics]))

# ==============================================================================

# Section 15: Hidden evaluation split (runs only when evaluation_split.csv exists)
assert "predict_churn" in dir(), "predict_churn() must be defined before evaluation."

evaluation_candidates = [
    EVALUATION_SPLIT_PATH,
    "/content/evaluation_split.csv"
]
actual_evaluation_path = next((p for p in evaluation_candidates if os.path.exists(p)), None)

if actual_evaluation_path is not None:
    evaluation_df = pd.read_csv(actual_evaluation_path)
    evaluation_df.columns = evaluation_df.columns.str.strip()

    if TARGET_COLUMN not in evaluation_df.columns:
        raise KeyError(f"'{TARGET_COLUMN}' is missing from evaluation_split.csv")

    # Convert the hidden target using the same cleaning logic.
    evaluation_clean = data_cleaning(evaluation_df)
    y_hidden = evaluation_clean[TARGET_COLUMN]

    # Use the original/raw evaluation features so predict_churn performs the full preprocessing.
    X_hidden = evaluation_df.drop(columns=[TARGET_COLUMN])
    y_pred, y_proba = predict_churn(X_hidden)

    print("Hidden Evaluation Results")
    evaluation_metrics = evaluate_predictions(y_hidden, y_pred, y_proba)
    display(pd.DataFrame([evaluation_metrics]))
else:
    print(
        "evaluation_split.csv was not found. This is OK for normal Colab testing. "
        "When the evaluator provides the file, this cell will run automatically."
    )

# ==============================================================================

# Section 16: Workspace check
variables_to_check = [
    "df", "processed_df", "train_split", "X_train",
    "preprocessor", "logistic_model", "rf_model",
    "model", "evaluate_predictions", "predict_churn"
]

print("=== Workspace Variable Status ===")
for var in variables_to_check:
    status = "✅ Defined" if var in globals() else "❌ Undefined"
    print(f"{var:<22}: {status}")