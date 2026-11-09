# SAD Mini-Activity and Reflection
## Problem
Academic risk is often noticed after poor results. This prototype explores early identification using available student indicators.
## Users
Faculty members or academic analysts.
## Inputs
Student attributes and indicators such as study time, absences, previous failures, G1 and G2. G3 is target only.
## Process
Validate data → preprocess → apply trained Random Forest → classify risk.
## Outputs
AT RISK / NOT AT RISK and model evaluation summary.
## Limitation
Data represents two Portuguese secondary schools and may not generalize to another institution.
## Reflection
1. **Target and G3 exclusion:** The target is G3 < 10 (AT RISK); G3 >= 10 (NOT AT RISK). G3 is excluded because it defines the outcome. Including it would cause target leakage.
2. **Recall:** Recall for AT RISK is the proportion of actually at-risk test records correctly identified. Add the actual recall from your run.
3. **Different institution:** Student backgrounds, school resources, grading practices, and learning environments may differ from the two Portuguese schools represented in the dataset.
4. **Faculty action:** Review the student's circumstances, speak with the student, identify learning needs, and consider tutoring or consultation. Never decide solely from the prediction.
## Actual evaluation — fill from your own run
| Metric | Result |
|---|---|
| Accuracy | __________ |
| Precision (AT RISK) | __________ |
| Recall (AT RISK) | __________ |
| F1-score (AT RISK) | __________ |
## Evidence
- [ ] Dataset inspection terminal
- [ ] Model evaluation terminal
- [ ] Streamlit dashboard
- [ ] Prediction results
- [ ] Missing-column error
- [ ] Downloaded results CSV
- [ ] Public GitHub repository
