import numpy as np
import pandas as pd
from xgboost import XGBRegressor as xgb_model
from sklearn.metrics import root_mean_squared_error #metric for the competition
from prepare_data import prepare_target


def validate_model(model, t_X, t_y, v_X, v_y):

    predictions_train = model.predict(t_X)
    predictions_val = model.predict(v_X)
    rmse_train = root_mean_squared_error(t_y, predictions_train)
    rmse_val = root_mean_squared_error(v_y, predictions_val)

    print(f'train rmse: {rmse_train}')
    print(f'val rmse: {rmse_val}')


def prepare_correction_train(df, model):
    X, y = prepare_target(df)

    fifth = int(len(y)/5)
    train_X, train_y = X[:-fifth], y[:-fifth]
    val_X, val_y = X[-fifth:], y[-fifth:]

    model.fit(train_X, train_y)
    predictions = model.predict(val_X)
    rmse = root_mean_squared_error()
    

def prepare_correction_test(df, model):

    df = df.copy()
    df['TVT_prev'] = df['TVT_input'].shift(1)
    df = df.dropna(subset=['TVT_input', 'TVT_prev', 'GR'])
    
    y = df['TVT_input']
    X = df.drop(columns=['TVT_input'])

    model.fit(X, y)

    return model