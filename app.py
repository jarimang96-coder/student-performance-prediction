from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Student Performance Prediction", page_icon="🎓", layout="wide")

MODEL_PATH = Path("student_model.pkl")
DATA_PATH = Path("student_data.csv")

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

st.title("🎓 Student Performance Prediction System")
st.caption("Classroom prototype for exploring potential Mathematics academic risk.")

if not MODEL_PATH.exists() or not DATA_PATH.exists():
    st.error("Project files are missing. Run `python train_model.py` first.")
    st.stop()

model = load_model()
data = load_data()

st.subheader("Dataset overview")
c1, c2, c3 = st.columns(3)
c1.metric("Historical records", len(data))
c2.metric("Average final grade", f"{data['G3'].mean():.2f}")
c3.metric("Historical at-risk records", int((data["G3"] < 10).sum()))

with st.expander("Preview the dataset"):
    st.dataframe(data.head(10), use_container_width=True)

st.divider()
st.subheader("Predict uploaded student records")
st.write("Upload a CSV with the same predictor columns used during training. Do not include G3.")
required_columns = list(model.feature_names_in_)

sample = data.drop(columns=["G3"]).head(5)
st.download_button(
    "Download sample input CSV",
    sample.to_csv(index=False).encode("utf-8"),
    file_name="sample_prediction_input.csv",
    mime="text/csv"
)

uploaded = st.file_uploader("Choose student CSV", type=["csv"])
if uploaded is not None:
    try:
        new_data = pd.read_csv(uploaded)
        missing = [col for col in required_columns if col not in new_data.columns]

        if missing:
            st.error("Missing required columns: " + ", ".join(missing))
        elif new_data.empty:
            st.warning("The uploaded CSV contains no records.")
        else:
            predictions = model.predict(new_data[required_columns])
            results = new_data.copy()
            results["Prediction"] = [
                "AT RISK" if int(value) == 1 else "NOT AT RISK"
                for value in predictions
            ]
            st.success("Prediction completed.")
            st.dataframe(results, use_container_width=True)
            left, right = st.columns(2)
            left.metric("Predicted AT RISK", int((predictions == 1).sum()))
            right.metric("Predicted NOT AT RISK", int((predictions == 0).sum()))
            st.download_button(
                "Download prediction results",
                results.to_csv(index=False).encode("utf-8"),
                file_name="student_predictions.csv",
                mime="text/csv"
            )
    except pd.errors.EmptyDataError:
        st.error("The uploaded CSV is empty.")
    except Exception as error:
        st.error(f"Could not process the uploaded file: {error}")

st.divider()
st.subheader("Responsible use")
st.info(
    "This model uses historical data from two Portuguese secondary schools. "
    "It may not generalize to another school or population. Predictions are not "
    "diagnoses and must never be the sole basis for decisions about students."
)
