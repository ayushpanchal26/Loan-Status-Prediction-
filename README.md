# Loan Status Prediction (Flask App)

A small Flask web app that loads an SVM loan-approval model (saved as `model.pkl`) and exposes a web form to predict whether a loan will be approved.

---

## Project structure
loan-pred-flask/
- ├─ app.py
- ├─ model.pkl
- ├─ requirements.txt
- └─ templates/
- └─ index.html

## About the model
- The model was trained in `loan_status_prediction.ipynb`.
- Model type: `sklearn.svm.SVC(kernel='linear')` (saved to `model.pkl`).
- Target: `Loan_Status` mapped `{ 'N': 0, 'Y': 1 }`.
- Categorical encodings used (IMPORTANT — the Flask app uses the same mapping):
  - `Gender`: `Male -> 1`, `Female -> 0`
  - `Married`: `Yes -> 1`, `No -> 0`
  - `Dependents`: `'3+'` was converted to `4` (others numeric: 0,1,2,...)
  - `Education`: `Graduate -> 1`, `Not Graduate -> 0`
  - `Self_Employed`: `Yes -> 1`, `No -> 0`
  - `Property_Area`: `Rural -> 0`, `Semiurban -> 1`, `Urban -> 2`

Feature order expected by the model:
['Gender','Married','Dependents','Education','Self_Employed','ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term','Credit_History','Property_Area']
