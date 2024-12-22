from pathlib import Path
import pandas as pd
from data.base import BaseTimeSeriesData
import data.processing as P


class PhuLienData(BaseTimeSeriesData):
    data_path = Path(__file__).parent / 'PhuLien.csv'

    def __init__(self, config, data=None):
        # Initialize the dataframe
        self.dataframe = data or pd.read_csv(self.data_path, index_col=0)
        super().__init__(config, self.dataframe)

    def prepare(self, data):
        return P.normalize_data(data)
