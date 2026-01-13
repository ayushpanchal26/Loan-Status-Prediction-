from flask import Flask, render_template, request
import numpy as np
import pickle
import os

app = Flask(__name__)

# --- Load model (expects model.pkl in the same folder) ---
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"model.pkl not found at {MODEL_PATH}. Please place the trained model there.")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# --- Mappings used in the original notebook ---
mappings = {
    "Gender": {"Male": 1, "Female": 0},
    "Married": {"Yes": 1, "No": 0},
    "Self_Employed": {"Yes": 1, "No": 0},
    "Education": {"Graduate": 1, "Not Graduate": 0},
    "Property_Area": {"Rural": 0, "Semiurban": 1, "Urban": 2},
    # Note: Dependents mapping is handled below (convert '3+' -> 4)
}

# Feature order expected by the model (must match training)
feature_order = [
    "Gender", "Married", "Dependents", "Education", "Self_Employed",
    "ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term",
    "Credit_History", "Property_Area"
]

def preprocess_form(form):
    """
    Take form values and return a 2D numpy array shaped (1, n_features)
    in the same order as feature_order.
    """
    values = []
    for feat in feature_order:
        if feat in ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term"]:
            # numeric fields: cast to float or int
            v_raw = form.get(feat, "").strip()
            try:
                # LoanAmount or incomes could be float
                v = float(v_raw)
            except ValueError:
                v = 0.0
            values.append(v)
        elif feat == "Dependents":
            dep = form.get("Dependents", "").strip()
            # Notebook converted '3+' to 4, keep same behaviour
            if dep == "3+":
                dep_val = 4
            else:
                try:
                    dep_val = int(dep)
                except ValueError:
                    dep_val = 0
            values.append(dep_val)
        elif feat == "Credit_History":
            # Accept '0','1' or 'No','Yes'
            ch = form.get("Credit_History", "").strip()
            if ch.lower() in ("1", "yes", "y", "true"):
                ch_val = 1
            elif ch.lower() in ("0", "no", "n", "false"):
                ch_val = 0
            else:
                # default: 0
                try:
                    ch_val = int(ch)
                except:
                    ch_val = 0
            values.append(ch_val)
        else:
            # categorical mappings
            raw = form.get(feat, "").strip()
            if raw == "":
                # fallback default
                mapped = 0
            else:
                mapped = mappings.get(feat, {}).get(raw, None)
                if mapped is None:
                    # try if user entered numeric string already
                    try:
                        mapped = int(raw)
                    except:
                        mapped = 0
            values.append(mapped)
    arr = np.asarray(values, dtype=float).reshape(1, -1)
    return arr

@app.route("/", methods=["GET", "POST"])
def index():
    result_text = None
    input_values = {}
    if request.method == "POST":
        # Preprocess form and make prediction
        input_values = request.form.to_dict()
        X = preprocess_form(request.form)
        pred = model.predict(X)  # model returns 0 or 1 based on notebook
        if hasattr(pred, "__len__"):
            pred_val = int(pred[0])
        else:
            pred_val = int(pred)
        if pred_val == 0:
            result_text = "Not approved (Loan Status = 0)"
        else:
            result_text = "Approved (Loan Status = 1)"
    return render_template("index.html", result=result_text, input_values=input_values)

if __name__ == "__main__":
    app.run(debug=True)
