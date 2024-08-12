import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import pandas as pd
import csv
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def save_confution_matrix(cm, file_path):
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.savefig(file_path)
    
    
def save_report(report, file_path):
    with open(file_path, 'w') as f:
        f.write(report)
        
        
def save_dataframe_as_csv(df, file_path):
    df.to_csv(file_path, index=False)


def save_model(model, path):
    joblib.dump(model, path)
    

def load_model(path):
    loaded_model = joblib.load(path)
    return loaded_model

def set_pandas_options():
    #setting the maximum number of printing columns 
    pd.set_option('display.max_columns', 20)
    # Increase the maximum width of the display
    pd.set_option('display.width', 1000)
    
    
def get_Errors(y, y_pred, x):
    mse = mean_squared_error(y, y_pred)
    print(f"Mean Squared Error: {mse:.2f}")
    
    # Calculating Mean Absolute Error (MAE)
    mae = mean_absolute_error(y, y_pred)
    print(f"Mean Absolute Error (MAE): {mae:.2f}")

    # Calculating Root Mean Squared Error (RMSE)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

    # Calculating R-squared (Coefficient of Determination)
    r2 = r2_score(y, y_pred)
    print(f"R-squared: {r2:.2f}")
    
    # Calculating Adjusted R-squared
    n = len(y)
    p = x.shape[1]  # Number of predictors
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    print(f"Adjusted R-squared: {adj_r2:.2f}")
    
def load_data(path):
    csv_file = pd.read_csv(path)
    return csv_file