from typing import Literal, Any, Union
from pydantic import BaseModel, Field
import numpy as np
import torch


RandomMode = Literal['random', 'middle', 'start', 'end']
GetBatchMode = Literal['random', 'seasonal', 'brute']


def get_random_seed():
    return np.random.randint(0, 99999)


def get_available_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class GlobalConfig(BaseModel):
    # General configs
    seed: int = Field(
        default_factory=get_random_seed, description='The random seed to use for reproducibility')
    device: Union[torch.device, str] = Field(
        default_factory=get_available_device, description='The device to use for training')
    dtype: torch.dtype = Field(torch.float32,
                               description='The data type to use for training')

    # Data configs
    window_size: int = Field(...,
                             description='The size of the window to perform the prediction')
    missing_size: int = Field(...,
                              description='The size of the missing values to simulate')
    data_simulate_random_mode: RandomMode = Field(
        'middle', description='The model to simulate missing values')
    data_simulate_replacement_value: Any = Field(
        None, description='The value to replace the missing values. If None, it will be the mean of dataset')
    data_batch_mode: GetBatchMode = Field(
        'random', description='The mode to get the batch data')
    data_num_samples: int = Field(
        100, description='The number of samples to generate')

    # Backbone model configs
    backbone_config: dict = Field(
        default_factory=dict, description='The configuration for the model')

    # Scheduler configs
    timesteps: int = Field(
        100, description='The number of timesteps to diffuse')
    trace_interval: int = Field(
        20, description='The interval to save the trace samples')
    scheduler_config: dict = Field(
        default_factory=dict, description='The configuration for the scheduler')

    # Training configs
    epochs: int = Field(..., description='The number of epochs for training')
    batch_size: int = Field(..., description='The batch size for training')
    save_path: str = Field(
        None, description='The path to save the model checkpoints')
    save_interval: int = Field(
        4, description='The interval to save the model checkpoints')
    use_early_stopping: bool = Field(
        True, description='Whether to use early stopping')
    early_stopping_patience: int = Field(
        10, description='The patience for early stopping')

    # Loss function configs
    loss_function_config: dict = Field(
        default_factory=dict, description='The configuration for the loss function')
    optimizer_config: dict = Field(
        default_factory=dict, description='The configuration for the optimizer')

    class Config:
        arbitrary_types_allowed = True
