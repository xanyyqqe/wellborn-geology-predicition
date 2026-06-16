from src.features import full_preprocessing_pipeline
import pandas as pd


def train_test_split_paths(paths:list):
    X_data, y_data = [], []
    k = 0
    for path in paths:
        new_df = full_preprocessing_pipeline(*path)
        X, y = prepare_target(new_df)

        X_data.append(X)
        y_data.append(y)
        k += 1
        if k % 100 == 0 or path == paths[-1]:
            print(f"processed {k} .csv's")

    sqrt_len = int(len(y_data)/4)
    train_X, train_y = X_data[:-sqrt_len], y_data[:-sqrt_len]
    val_X, val_y = X_data[-sqrt_len:], y_data[-sqrt_len:]
    return train_X, train_y, val_X, val_y


def prepare_target(df: pd.DataFrame):

    df = df.copy()
    df['TVT_prev'] = df['TVT'].shift(1)
    df = df.dropna(subset=['TVT', 'TVT_prev'])
    
    y = df['TVT']
    X = df.drop(columns=['TVT', 'TVT_input'])
    
    return X, y