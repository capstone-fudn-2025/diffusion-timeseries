import os
import time
import torch
from tqdm import tqdm
from data.base import BaseTimeSeriesData
from backbones.base import TripleB
from schedulers.base import BaseScheduler
from utils.config import GlobalConfig


class Trainer:
    def __init__(
        self,
        config: GlobalConfig,
        dataset: BaseTimeSeriesData,
        model: TripleB,
        scheduler: BaseScheduler,
        loss_function: torch.nn.Module = None,
        optimizer: torch.optim.Optimizer = None
    ):
        self.config = config

        # Initialize dataset
        assert isinstance(
            dataset, BaseTimeSeriesData), f"Invalid dataset type, expected BaseTimeSeriesData but got {type(dataset)}"
        self.dataset = dataset

        # Initialize model
        assert isinstance(
            model, TripleB), f"Invalid model type, expected TripleB but got {type(model)}"
        self.model = model
        self.model.to(config.device)

        # Initialize scheduler
        assert isinstance(
            scheduler, BaseScheduler), f"Invalid scheduler type, expected BaseScheduler but got {type(scheduler)}"
        self.scheduler = scheduler

        # Initialize loss function and optimizer
        self.loss_function = loss_function or torch.nn.MSELoss(
            **config.loss_function_config)
        self.optimizer = optimizer or torch.optim.Adam(
            self.model.parameters(), **config.optimizer_config)

        # Variables
        self.losses = []

    def train(self):
        # Set model to train mode
        self.model.train()

        # Print training process
        print(f"🧠 Training process begin.")
        print(f"- Timesteps: {self.config.timesteps}")
        print(f"- Batch size: {self.config.batch_size}")
        print(f"- Epochs: {self.config.epochs}")
        print(f"- Device: {self.config.device}")
        print(f"- Dataset: {self.dataset.__class__.__name__}")
        print(f"- Model: {self.model.__class__.__name__}")
        print(f"- Scheduler: {self.scheduler.__class__.__name__}")

        # Define variables
        running_loss = 0.0

        # Create checkpoint directory
        if self.config.save_path:
            os.makedirs(self.config.save_path, exist_ok=True)

        # Iterate over epochs
        _s = time.perf_counter()
        for epoch in range(self.config.epochs):
            progress_bar = tqdm(
                self.dataset, desc=f'🚅 Epoch {epoch+1}/{self.config.epochs}')
            for batch in progress_bar:
                # Zero gradients
                self.optimizer.zero_grad()

                # Create noise sample
                noise = torch.randn_like(batch).to(self.config.device)
                time_tensor = torch.randint(
                    1, self.config.timesteps + 1, (batch.shape[0], 1, 1)).to(self.config.device)
                noise_sample = self.scheduler.training_diffusion(
                    sample=batch, timestep=time_tensor, noise=noise)

                # Predict noise
                output = self.model(
                    noise_sample, time_tensor / self.config.timesteps)

                # Calculate loss
                loss = self.loss_function(output, noise)

                # Backward pass
                loss.backward()
                self.optimizer.step()

                # Update progress bar
                progress_bar.set_postfix(loss=loss.item())

                # Update total loss
                running_loss += loss.item()

            # Update losses
            self.losses.append(running_loss / len(self.dataset))
            running_loss = 0

            # Save model after interval or last epoch
            if self.config.save_path and (epoch % self.config.save_interval == 0 or epoch == self.config.epochs - 1):
                torch.save(self.model.state_dict(),
                           f"{self.config.save_path}/{self.model.__class__.__name__}_{epoch+1}.pt")
                print(f"📦 Model saved at epoch {epoch+1}")

            # Early stopping
            if self.config.use_early_stopping and max(self.losses[-self.config.early_stopping_patience:]) == self.losses[-1]:
                print(
                    f"🛑 Early stopping at epoch {epoch+1} due to no improvement in loss")
                break

        print(
            f"🏁 Training process complete at {round(time.perf_counter() - _s, 2)}s")
        self.model.eval()

    def visualize_history(self, clear: bool = True):
        '''
        Visualize loss history
        - clear (bool): Clear the losses after visualization (default: True)
        '''

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=(15, 5))
        ax.plot(self.losses)
        ax.set_title('Loss history')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')
        plt.show()
        plt.close(fig)
        if clear:
            self.losses = []
