# Student Performance Prediction System
A classroom prototype using Random Forest to classify historical Mathematics student records.
## Rule
G3 < 10 = AT RISK (1); G3 >= 10 = NOT AT RISK (0). G3 is excluded from model inputs to prevent target leakage.
## Dataset
UCI Student Performance, ID 320: https://archive.ics.uci.edu/dataset/320/student+performance
## Setup
Python 3.10+. Run `python -m pip install -r requirements.txt`, then `python train_model.py`, then `streamlit run app.py`. Open the local URL shown, normally http://localhost:8501.
## Outputs
Training generates `student_data.csv`, `student_model.pkl`, and `model_metrics.json`. The app previews data, supplies a sample CSV, validates uploaded columns, predicts risk, and downloads results.
## GitHub
Create a public repository named `student-performance-prediction`. Push source files and documentation. Follow instructor rules for generated files. Never publish credentials.
## Limitations
Data represents two Portuguese secondary schools and may not generalize. Predictions are not diagnoses and must not be the sole basis for decisions.
## References
UCI: https://archive.ics.uci.edu/dataset/320/student+performance
Scikit-learn: https://scikit-learn.org/stable/
Streamlit: https://docs.streamlit.io/
GitHub: https://docs.github.com/en/get-started
