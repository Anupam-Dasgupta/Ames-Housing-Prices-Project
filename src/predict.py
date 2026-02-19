from pathlib import Path
import joblib
import os
import pandas as pd
import numpy as np

def main():
    BASE_DIR = Path(__file__).resolve().parent.parent

    MODEL_PATH = BASE_DIR / "models" / "stack_model.joblib"

    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model not found. Run train.py first")
    
    TEST_PATH = BASE_DIR / "data" / "test_cleaned.csv"
    OUT_PATH = BASE_DIR / "submissions" / "submission.csv"
    
    model = joblib.load(MODEL_PATH)
    X_test = pd.read_csv(TEST_PATH)

    raw_preds = model.predict(X_test)
    preds = np.expm1(raw_preds)

    sub = pd.DataFrame({"Id":X_test["Id"],
                        "SalePrice":preds})

    os.makedirs(BASE_DIR / "submissions",exist_ok=True)
    sub.to_csv(OUT_PATH,index=False)
    print(f"Saved: {OUT_PATH}")

if __name__ == "__main__":
    main()
    