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

    paths = []
    for root, _ ,files in os.walk(path):

        current_well = []
        for file in files:

            if not(file.endswith('.csv')):
                continue
            
            current_well.append(os.path.join(root, file))
            if len(current_well) == 2:
                paths.append(current_well)
                current_well.clear()
    
    return paths


def prepare_horizontal(path:str) -> pd.DataFrame:

    df = pd.read_csv(path)
    if not(df):
        raise Exception('wrong path for horizontal .csv')
    
    df = df.sort_values(by='MD').reset_index(drop=True)
    if 'GR' in df.columns:
        df['GR'] = (df['GR'].
                    interpolate(method='linear').
                    bfill().
                    ffill())
        
    return df


def prepare_typewell(path:str) -> pd.DataFrame:

    df = pd.read_csv(path)
    if not(df):
        raise Exception('wrong path for typewell .csv')
    
    df['GR'] = df['GR'].fillna(df['GR'].median())

    return df


