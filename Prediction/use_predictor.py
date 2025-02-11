import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

def load_data(test_path):
    test_data = pd.read_csv(test_path).sample(frac=1, random_state=45).reset_index(drop=True)
    X_test = test_data.iloc[:, 1:].values
    y_test = test_data.iloc[:, 0].values
    return X_test, y_test

def main():
    filepath = ''                   #you should use ur own path
    X_test, y_test = load_data(filepath)                  

    scaler = StandardScaler()
    X_test_scaled = scaler.fit_transform(X_test)

    loaded_model = joblib.load('')                  #you should use ur own path

    y_pred = loaded_model.predict(X_test_scaled)
    y_pred_proba = loaded_model.predict_proba(X_test_scaled)[:, 1]

    result_df = pd.DataFrame({
        'True Label': y_test,
        'Predicted Label': y_pred,
        'Predicted Probability': y_pred_proba
    })
    result_df.to_csv('prediction_results.csv', index=False)
    print(f"reult saved to: {filepath}")

if __name__ == "__main__":
    main()

