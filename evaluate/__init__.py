import pandas as pd
from tqdm import tqdm
from utils.config import GlobalConfig
from data.base import BaseTimeSeriesData
from backbones.base import TripleB
from schedulers.base import BaseScheduler
import evaluate.metrics as M


class Evaluator:
    def __init__(
        self,
        config: GlobalConfig,
        dataset: BaseTimeSeriesData,
        backbone: TripleB,
        scheduler: BaseScheduler,
    ):
        self.config = config
        self.dataset = dataset
        self.backbone = backbone
        self.scheduler = scheduler

        self.metrics_result = pd.DataFrame(columns=[
            "Similarity", "NMAE", "R2", "RMSE", "FSD", "FB", "FA2"
        ])
        self.plot_result = []

    def evaluate(self, repeat: int = 1):
        for i in tqdm(range(repeat), desc="Evaluating"):
            # Get the next batch
            _, mask = self.dataset.get_missing_window_and_mask()

            # Get the imputed values
            output = self.scheduler(self.backbone, mask)

            # Get original part
            original_index = (
                self.dataset.data.missing_indices[0] + self.dataset.data.missing_indices[1]) // 2 - (self.config.missing_size // 2)
            original = self.dataset.data.processed_dataset[original_index:
                                                           original_index + self.config.missing_size].values

            # Get predicted part
            predicted_index = self.config.window_size // 2 - self.config.missing_size // 2
            predicted = output[:, :, predicted_index:predicted_index +
                               self.config.missing_size].detach().cpu().numpy().reshape(-1, 1)

            # Calculate the metrics
            similarity = M.similarity(predicted, original)
            nmae = M.nmae(predicted, original)
            r2 = M.r2(predicted, original)
            rmse = M.rmse(predicted, original)
            fsd = M.fsd(predicted, original)
            fb = M.fb(predicted, original)
            fa2 = M.fa2(predicted, original)

            # Append the result
            self.metrics_result.loc[i] = [
                similarity, nmae, r2, rmse, fsd, fb, fa2]

            # Append the plot
            self.plot_result.append((original, predicted))

        return self.metrics_result

    def visualize_result(self):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, len(self.plot_result),
                               figsize=(5 * len(self.plot_result), 5))
        for i, (original, predicted) in enumerate(self.plot_result):
            if len(self.plot_result) == 1:
                ax.plot(original, label='Original')
                ax.plot(predicted, label='Predicted')
                ax.set_title(f"Result {i + 1}")
                ax.legend()
            else:
                ax[i].plot(original, label='Original')
                ax[i].plot(predicted, label='Predicted')
                ax[i].set_title(f"Result {i + 1}")
                ax[i].legend()
        plt.tight_layout()
        plt.show()
        plt.close(fig)
