import os
import pandas as pd

# rogii-wellbore-prediction/

# ├── .gitignore               # Игнорирование тяжелых данных и кэша
# ├── README.md                # Главное описание проекта (метрики, подходы)
# ├── requirements.txt         # Список библиотек (pip freeze > requirements.txt)
# │
# ├── data/                    # ЭТОЙ ПАПКИ НЕ ДОЛЖНО БЫТЬ НА GITHUB
# │   ├── raw/                 # Сюда скачиваете оригинальные файлы с Kaggle
# │   └── processed/           # Сюда сохраняете обработанные признаки
# │
# ├── notebooks/               # Папка для экспериментов
# │   ├── 01_eda_geology.ipynb # Исследовательский анализ данных (графики, логи)
# │   └── 02_prototyping.ipynb # Черновики моделей и проверка валидации
# │
# ├── src/                     # Основной модуль автоматизации (чистый Python)
# │   ├── __init__.py
# │   ├── config.py            # Пути к файлам, гиперпараметры, SEED
# │   ├── data_loader.py       # Загрузка и первичная очистка данных
# │   ├── features.py          # Расчет признаков (rolling GR stats, cKDTree)
# │   └── models.py            # Архитектуры моделей (LightGBM / PyTorch 1D-CNN)
# │
# ├── train.py                 # Скрипт локального обучения и кросс-валидации
# └── submission_builder.py    # Скрипт сборки финального файла для Kaggle

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