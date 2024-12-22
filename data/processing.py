import pandas as pd


def normalize_data(data: pd.DataFrame) -> pd.DataFrame:
    return (data - data.mean()) / data.std()
