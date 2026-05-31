import numpy as np
import pandas as pd
from xgboost import XGBRegressor as xgb_model
from sklearn.metrics import root_mean_squared_error #metric for the competition

def prepare_target(df: pd.DataFrame):

    df = df.copy()
    df['TVT_prev'] = df['TVT'].shift(1)
    df = df.dropna(subset=['TVT', 'TVT_prev'])
    
    y = df['TVT']
    X = df.drop(columns=['TVT'])
    
    return X, y


def train_xgb(path_train_csvs:str, model:xgb_model):

    final_df = pd.concat(path_train_csvs)

    k = 0
    