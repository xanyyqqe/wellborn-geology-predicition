import os
import pandas as pd


def prepare_paths(path:str) -> list:

    wells = {}

    for root, _, files in os.walk(path):
        for file in sorted(files):
            if not file.endswith('.csv'):
                continue

            if file.endswith('__horizontal_well.csv'):
                well_id = file.replace('__horizontal_well.csv', '')
                wells.setdefault(well_id, {})['horizontal'] = os.path.join(root, file)
            elif file.endswith('__typewell.csv'):
                well_id = file.replace('__typewell.csv', '')
                wells.setdefault(well_id, {})['typewell'] = os.path.join(root, file)

    paths = []
    for well_id in sorted(wells):
        well_files = wells[well_id]
        if 'horizontal' in well_files and 'typewell' in well_files:
            paths.append([well_files['horizontal'], well_files['typewell']])

    return paths


def load_horizontal_well(path:str) -> pd.DataFrame:

    df = pd.read_csv(path)
    if df.empty:
        raise Exception('wrong path for horizontal .csv')
    
    df = df.sort_values(by='MD').reset_index(drop=True)
    
    formations_to_drop = ['ANCC', 'ASTNU', 'ASTNL', 'EGFDU', 'EGFDL', 'BUDA']
    if 'ANCC' in df.columns:
        df = df.drop(columns=formations_to_drop)

    if 'GR' in df.columns:
        df['GR'] = (df['GR'].
                    interpolate(method='linear').
                    bfill().
                    ffill())
        
    return df


def load_typewell(path:str) -> pd.DataFrame:

    df = pd.read_csv(path)
    if df.empty:
        raise Exception('wrong path for typewell .csv')
    
    df['GR'] = df['GR'].fillna(df['GR'].median())

    return df