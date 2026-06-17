import numpy as np
import pandas as pd
from xgboost import XGBRegressor as xgb_model
from sklearn.metrics import root_mean_squared_error #metric for the competition
from src.prepare_data import prepare_target
from sklearn.preprocessing import StandardScaler
from src.features import full_preprocessing_pipeline

def validate_model(model, t_X, t_y, v_X, v_y):

    predictions_train = model.predict(t_X)
    predictions_val = model.predict(v_X)
    rmse_train = root_mean_squared_error(t_y, predictions_train)
    rmse_val = root_mean_squared_error(v_y, predictions_val)

    print(f'train rmse: {rmse_train}')
    print(f'val rmse: {rmse_val}')

    return rmse_train, rmse_val


def prepare_correction_train_data(model, X_tr, y_tr, X_val, y_val):

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_val = scaler.fit_transform(X_val)

    model.fit(X_tr, y_tr)

    quarter = int(len(X_tr)/3)

    X_tr_real = X_tr[:quarter]
    y_tr_real = y_tr[:quarter]

    X_val_real = np.concatenate([X_tr[quarter:], X_val], axis=0)
    y_val_real = np.concatenate([y_tr[quarter:], y_val], axis=0)
    

    rmse_train, rmse_val = validate_model(model, 
                            X_tr_real, y_tr_real, X_val_real, y_val_real)
    return model, rmse_train, rmse_val


def prepare_correction_test_data(df, model):
    df = df.copy()

    df['TVT_prev'] = df['TVT_input'].shift(1)
    df['TVT_prev'] = df['TVT_prev'].where(df['TVT_input'].notna(), df['TVT_prev'])

    df = df.dropna(subset=['TVT_input', 'TVT_prev', 'GR'])

    known_tvt = df['TVT_input'].notna().sum()
    
    y = df.loc[df['TVT_input'].notna(), 'TVT_input']
    X = df.loc[df['TVT_input'].notna()].drop(columns=['TVT_input'])

    scaler = StandardScaler()

    X = scaler.fit_transform()

    threshold = int(known_tvt * 0.8)
    train_X, train_y = X.iloc[:threshold], y.iloc[:threshold]
    val_X, val_y = X.iloc[threshold:], y.iloc[threshold:]

    model.fit(train_X, train_y)
    rmse_train, rmse_val = validate_model(model, train_X, train_y, val_X, val_y)

    return model, rmse_train, rmse_val