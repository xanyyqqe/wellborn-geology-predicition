import pandas as pd
import numpy as np
from scipy.spatial import cKDTree


def generate_signal_features(df: pd.DataFrame, window_sizes=[5, 11, 21]) -> pd.DataFrame:

    df = df.copy()
    
    for w in window_sizes:
        df[f'gr_roll_mean_{w}'] = df['GR'].rolling(window=w, min_periods=1, center=True).mean()
        df[f'gr_roll_std_{w}'] = df['GR'].rolling(window=w, min_periods=1, center=True).std().fillna(0)
        df[f'gr_roll_min_{w}'] = df['GR'].rolling(window=w, min_periods=1, center=True).min()
        df[f'gr_roll_max_{w}'] = df['GR'].rolling(window=w, min_periods=1, center=True).max()
        
        df[f'gr_diff_{w}'] = df['GR'] - df[f'gr_roll_mean_{w}']
        
    for lag in [1, 2, 3]:
        df[f'gr_lag_{lag}'] = df['GR'].shift(lag).fillna(method='bfill')
        df[f'gr_lead_{lag}'] = df['GR'].shift(-lag).fillna(method='ffill')
        
    return df


def generate_spatial_features(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    df['delta_X'] = df['X'].diff().fillna(0)
    df['delta_Y'] = df['Y'].diff().fillna(0)
    df['delta_Z'] = df['Z'].diff().fillna(0)
    
    horizontal_dist = np.sqrt(df['delta_X']**2 + df['delta_Y']**2)
    df['trajectory_slope'] = df['delta_Z'] / (horizontal_dist + 1e-6)
    
    return df


def integrate_with_typewells(hz_df: pd.DataFrame, tw_df: pd.DataFrame) -> pd.DataFrame:

    hz_df = hz_df.copy()
    
    hz_start = np.array([hz_df['X'].iloc[0], hz_df['Y'].iloc[0]])
    tw_start = np.array([tw_df['X'].iloc[0], tw_df['Y'].iloc[0]])
    hz_df['dist_to_closest_typewell'] = np.linalg.norm(hz_start - tw_start)
    
    tw_tree = cKDTree(tw_df[['GR']].values)
    distances, indices = tw_tree.query(hz_df[['GR']].values, k=1)
    
    hz_df['tw_reference_Z'] = tw_df['Z'].iloc[indices].values
    hz_df['tw_reference_TVT'] = tw_df['TVT'].iloc[indices].values
    hz_df['gr_difference_to_tw'] = distances
    
    return hz_df


def full_preprocessing_pipeline(hz_path: str, tw_path: str) -> pd.DataFrame:

    from src.data_loader import load_horizontal_well, load_typewell
    hz_df = load_horizontal_well(hz_path)
    tw_df = load_typewell(tw_path)

    hz_df = generate_signal_features(hz_df)
    hz_df = generate_spatial_features(hz_df)

    processed_with_typewell = integrate_with_typewells(hz_df, tw_df)
    
    return processed_with_typewell
