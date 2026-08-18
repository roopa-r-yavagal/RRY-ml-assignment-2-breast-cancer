import streamlit as st
import pandas as pd
import joblib

# ============================================================
# App skeleton — upload CSV + select a model
# ============================================================

st.set_page_config(page_title="Breast Cancer Classifier", layout="wide")

st.title("Breast Cancer Classification")
st.write(
    "Upload test data (CSV) and select a model to see its predictions "
    "and performance on the uploaded data."
)

# ------------------------------------------------------------
# Model selection dropdown
# ------------------------------------------------------------
model_options = {
    "Logistic Regression": "model/logistic_regression.pkl",
    "Decision Tree": "model/decision_tree.pkl",
    "kNN": "model/knn.pkl",
    "Naive Bayes": "model/naive_bayes.pkl",
    "Random Forest": "model/random_forest.pkl",
}

selected_model_name = st.selectbox("Select a model", list(model_options.keys()))

# ------------------------------------------------------------
# CSV upload
# ------------------------------------------------------------
uploaded_file = st.file_uploader("Upload test data (CSV)", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    #st.write("Preview of uploaded data:")
    #st.dataframe(data.head())

    # Load the selected model and the scaler
    model = joblib.load(model_options[selected_model_name])
    scaler = joblib.load("model/scaler.pkl")

    st.success(f"Loaded model: {selected_model_name}")
    #st.write(f"Uploaded data shape: {data.shape}")
    st.caption(f"Loaded {data.shape[0]} rows × {data.shape[1]} columns — columns: {', '.join(data.columns)}")

    # ------------------------------------------------------------
    # Separate target from features
    # ------------------------------------------------------------
    if "target" not in data.columns:
        st.error("Uploaded CSV must contain a 'target' column (1=malignant, 0=benign).")
        st.stop()

    y_true = data["target"]
    X = data.drop(columns=["target"])

    # ------------------------------------------------------------
    # Scale features using the saved (fitted) scaler
    # ------------------------------------------------------------
    X_scaled = scaler.transform(X)

    # ------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------
    y_pred = model.predict(X_scaled)
    y_proba = model.predict_proba(X_scaled)[:, 1]

    # ------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------
    from sklearn.metrics import (
        accuracy_score, roc_auc_score, precision_score,
        recall_score, f1_score, matthews_corrcoef, confusion_matrix
    )

    accuracy = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_proba)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)

    st.subheader("Evaluation Metrics")
    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
    metrics_col1.metric("Accuracy", f"{accuracy:.3f}")
    metrics_col1.metric("AUC", f"{auc:.3f}")
    metrics_col2.metric("Precision", f"{precision:.3f}")
    metrics_col2.metric("Recall", f"{recall:.3f}")
    metrics_col3.metric("F1 Score", f"{f1:.3f}")
    metrics_col3.metric("MCC", f"{mcc:.3f}")

    # ------------------------------------------------------------
    # Confusion Matrix
    # ------------------------------------------------------------
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(
        cm,
        index=["Actual: Benign (0)", "Actual: Malignant (1)"],
        columns=["Predicted: Benign (0)", "Predicted: Malignant (1)"]
    )
    st.dataframe(cm_df)
   
    # ------------------------------------------------------------
    # Predictions table
    # ------------------------------------------------------------
    st.subheader("Predictions")
    results_df = data.copy()
    results_df["Predicted"] = y_pred
    results_df["Predicted Probability (Malignant)"] = y_proba
    st.dataframe(results_df)

else:
    st.info("Please upload a CSV file to continue.")
