from typing import Any
import numpy.typing as npt
import numpy as np
import pandas as pd
import torch
from pydantic import BaseModel, Field
from utils.config import GlobalConfig, RandomMode


class SimulationDataset(BaseModel):
    # Dataset
    dataset: pd.DataFrame = Field(...,
                                  description='The dataset to simulate missing values')
    processed_dataset: pd.DataFrame = Field(...,
                                            description='The dataset after applying the processing')
    missing_dataset: pd.DataFrame = Field(...,
                                          description='The dataset with missing values')
    missing_indices: tuple[int, int] = Field(...,
                                             description='The indices of missing values')

    def __len__(self):
        return len(self.dataset)

    class Config:
        arbitrary_types_allowed = True


class BaseTimeSeriesData:
    '''
    Base class for time series data & loader
    '''

    def __init__(self, config: GlobalConfig, data: pd.DataFrame = None):
        self.config = config

        # Validate the config
        if self.config.window_size < self.config.missing_size:
            raise ValueError(
                'Window size must be greater than missing size')

        # Simulate missing values in the dataset
        self.data = self.__simulate_missing_dataset(
            dataset=data,
            missing_size=config.missing_size,
            mode=config.data_simulate_random_mode,
            replacement_value=config.data_simulate_replacement_value,
            seed=config.seed
        )

        # Get samples index based on the mode
        if self.config.data_batch_mode == 'random':
            self.samples_index = self.get_samples_index_random_mode()
        elif self.config.data_batch_mode == 'seasonal':
            self.samples_index = self.get_samples_index_seasonal_mode()
        elif self.config.data_batch_mode == 'brute':
            self.samples_index = self.get_samples_index_brute_mode()
        else:
            raise ValueError(
                'Invalid mode. Choose from random, seasonal, brute')

    def prepare(self, data: pd.DataFrame) -> pd.DataFrame:
        '''
        Prepare the dataset.
        '''
        return data

    def __simulate_missing_dataset(
        self,
        dataset: pd.DataFrame,
        missing_size: int,
        mode: RandomMode = 'middle',
        replacement_value: Any = None,
        seed: int = 42
    ) -> SimulationDataset:
        # Set random seed if mode is random
        if mode == 'random':
            np.random.seed(seed)

        # Preprocessing
        processed_dataset = self.prepare(dataset.copy())

        # Get replacement value is mean of dataset if None
        if replacement_value is None:
            replacement_value = processed_dataset.mean()

        # Create a copy of the dataset
        missing_dataset = processed_dataset.copy()

        # Safe random range
        end_idx = len(missing_dataset) - missing_size

        # If mode is random, randomly select indices to replace
        if mode == 'random':
            missing_point = np.random.randint(0, end_idx)

            # Replace values with NaN or replacement value
            missing_dataset.iloc[missing_point:missing_point +
                                 missing_size] = replacement_value

            missing_indices = (missing_point, missing_point + missing_size)

        # If mode is middle, replace the middle values with NaN or replacement value
        elif mode == 'middle':
            missing_point = end_idx // 2 - missing_size // 2

            # Replace values with NaN or replacement value
            missing_dataset.iloc[missing_point:missing_point +
                                 missing_size] = replacement_value

            missing_indices = (missing_point, missing_point + missing_size)

        # If mode is start, replace the start values with NaN or replacement value
        elif mode == 'start':
            missing_point = 0

            # Replace values with NaN or replacement value
            missing_dataset.iloc[missing_point:missing_point +
                                 missing_size] = replacement_value

            missing_indices = (missing_point, missing_point + missing_size)

        # If mode is end, replace the end values with NaN or replacement value
        elif mode == 'end':
            missing_point = end_idx

            # Replace values with NaN or replacement value
            missing_dataset.iloc[missing_point:missing_point +
                                 missing_size] = replacement_value

            missing_indices = (missing_point, missing_point + missing_size)

        else:
            raise ValueError(
                'Invalid mode. Choose from random, middle, start, end')

        return SimulationDataset(dataset=dataset, processed_dataset=processed_dataset, missing_dataset=missing_dataset, missing_indices=missing_indices)

    def collate(self, data: npt.ArrayLike) -> torch.Tensor:
        return torch.tensor(data, dtype=self.config.dtype, device=self.config.device)

    def __len__(self):
        return self.config.data_num_samples // self.config.batch_size

    def get_samples_index_random_mode(self) -> list[int]:
        # Sample before and after the missing values
        _samples_before_missing = range(
            0, self.data.missing_indices[0] - self.config.window_size)
        _samples_after_missing = range(
            self.data.missing_indices[1] + 1, len(self.data) - self.config.window_size)
        return np.random.choice(list(_samples_before_missing) + list(_samples_after_missing), self.config.data_num_samples).tolist()

    def get_samples_index_seasonal_mode(self) -> list[int]:
        raise NotImplementedError('Seasonal mode not implemented')

    def get_samples_index_brute_mode(self) -> list[int]:
        raise NotImplementedError('Brute mode not implemented')

    def __getitem__(self, idx: int) -> torch.Tensor:
        if idx >= len(self):
            raise IndexError(
                f"Index {idx} out of range. Max length is {len(self)}")

        # Get the samples
        samples = []
        for i in self.samples_index[idx * self.config.batch_size:(idx + 1) * self.config.batch_size]:
            samples.append(
                self.data.missing_dataset.iloc[i:i + self.config.window_size].values.reshape(1, -1))

        return self.collate(np.array(samples))

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def get_missing_window_and_mask(self) -> tuple[torch.Tensor, torch.Tensor]:
        _window_center = (
            self.data.missing_indices[0] + self.data.missing_indices[1]) // 2
        _window_start = _window_center - self.config.window_size // 2
        _window_end = _window_center + self.config.window_size // 2
        window = self.collate(
            self.data.missing_dataset.iloc[_window_start:_window_end].values.reshape(1, -1)).unsqueeze(0)
        mask = torch.zeros_like(window)
        mask[:, :, self.config.window_size // 2 - self.config.missing_size //
             2:self.config.window_size // 2 + self.config.missing_size // 2] = 1
        return window, mask

    def merge_results(self, results: torch.Tensor) -> pd.DataFrame:
        '''
        Merge the results into the dataset
        '''
        # Create a copy of the dataset
        dataset = self.data.missing_dataset.copy()

        # Merge the results
        dataset.iloc[self.data.missing_indices[0]:self.data.missing_indices[1]] = results

        return dataset

    def visualize(self):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(3, 1, figsize=(15, 9))
        ax[0].plot(self.data.dataset.to_numpy(), label='Original')
        ax[0].set_title('Original Dataset')
        ax[0].legend()

        ax[1].plot(self.data.processed_dataset.to_numpy(), label='Processed')
        ax[1].set_title('Processed Dataset')
        ax[1].legend()

        ax[2].plot(self.data.missing_dataset.to_numpy(), label='Missing')
        ax[2].set_title('Missing Dataset')
        ax[2].legend()

        plt.tight_layout()
        plt.show()
        plt.close(fig)
