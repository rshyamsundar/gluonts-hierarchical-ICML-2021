import inspect
from typing import Callable, Dict, NamedTuple, Optional, Type, Union

import numpy as np
import pandas as pd

from gluonts.dataset.common import Dataset, ListDataset
from gluonts.evaluation import MultivariateEvaluator
from gluonts.evaluation.backtest import backtest_metrics
from gluonts.model.deephier import DeepHierEstimator
from gluonts.model.estimator import Estimator
from gluonts.model.predictor import Predictor
from gluonts.model.r_forecast import RHierarchicalForecastPredictor
from gluonts.mx.trainer import Trainer

from config.dataset_config import dataset_config_dict


class HierarchicalDatasetInfo(NamedTuple):
    train_dataset: Dataset
    test_dataset: Dataset
    S: np.ndarray
    index: pd.Index
    freq: str
    target_dim: int
    target_train: np.ndarray
    target_test: np.ndarray


class Experiment:

    def __init__(
        self,
        dataset_path: str,
        estimator: Union[Type[Estimator], Type[Predictor]],
        hyper_params: Dict,
        job_config: Dict,
    ) -> None:
        self.dataset_path = dataset_path
        self.estimator = estimator
        self.hyper_params = hyper_params
        self.job_config = job_config

    def _get_matching_params(self, func: Callable):
        """
        Returns the hyper-params that are applicable to `func`.

        Parameters
        ----------
        func

        Returns
        -------

        """
        func_args = inspect.signature(func).parameters
        return {
            arg: self.hyper_params[arg]
            for arg in func_args.keys() & self.hyper_params.keys()
        }

    def _get_hierarchical_dataset(self):
        """
        Get the dataset given the location.

        Returns
        -------

        """
        data = pd.read_csv(f'{self.dataset_path}/data.csv', index_col=0)
        Y = data.values.transpose()
        index = data.index
        freq = pd.to_datetime(index).freq
        if freq is None:
            freq = pd.infer_freq(index)

            # hack to deal with quarterly data (infer_freq doesn't recognize '3M')
            if freq[0] == 'Q':
                freq = '3M'
            elif freq == 'MS':  # Month-start frequency
                freq = 'M'

        try:
            import scipy.sparse
            S = scipy.sparse.load_npz(f'{self.dataset_path}/agg_mat.npz')
            S = S.todense()
            S = np.array(S)
        except:
            S = pd.read_csv(f'{self.dataset_path}/agg_mat.csv', index_col=0).values

        # Split into train and test sets
        dataset_str = self.dataset_path.split('/')[-1]
        prediction_length = dataset_config_dict[dataset_str]["prediction_length"]

        # Get the right split for validation or test: `evaluation_length` time steps will be evaluated.
        if "validation" in self.job_config and self.job_config["validation"]:
            target_train = Y[:, :len(index) - 2 * prediction_length]
            target_test = Y[:, :len(index) - prediction_length]
            valid_plus_test_length = 2 * prediction_length
        else:
            target_train = Y[:, :len(index) - prediction_length]
            target_test = Y
            valid_plus_test_length = prediction_length

        train_dataset = ListDataset(
            [{"start": index[0], "item_id": "all_items", "target": target_train}],
            freq=freq,
            one_dim_target=False
        )
        test_dataset = ListDataset(
            [{"start": index[0], "item_id": "all_items", "target": target_test}],
            freq=freq,
            one_dim_target=False
        )

        for _data in train_dataset.list_data:
            print(f'train target shape: {_data["target"].shape}')

            # confirm train target has the right shape!
            assert _data["target"].shape == (Y.shape[0], Y.shape[1] - valid_plus_test_length)

        for _data in test_dataset.list_data:
            print(f'test target shape: {_data["target"].shape}')

            # confirm test target has the right shape!
            assert _data["target"].shape == (Y.shape[0], Y.shape[1])

        return HierarchicalDatasetInfo(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            S=S,
            index=index,
            freq=freq,
            target_dim=Y.shape[0],
            target_train=target_train,
            target_test=target_test,
        )

    def run(self):
        hier_dataset = self._get_hierarchical_dataset()
        train_dataset, test_dataset, S = hier_dataset.train_dataset, hier_dataset.test_dataset, hier_dataset.S
        target_dim = hier_dataset.target_dim
        freq = hier_dataset.freq

        trainer = Trainer(
            **self._get_matching_params(Trainer)
        )

        dataset_str = self.dataset_path.split('/')[-1]

        if self.estimator != RHierarchicalForecastPredictor:
            estimator = self.estimator(
                freq=freq,
                prediction_length=dataset_config_dict[dataset_str]["prediction_length"],
                target_dim=target_dim,
                S=S,
                **self._get_matching_params(self.estimator),
                trainer=trainer,
            )

            predictor = estimator.train(train_dataset)
        else:
            tags = pd.read_csv(f'{self.dataset_path}/tags.csv', header=None).to_numpy()
            predictor = self.estimator(
                freq=freq,
                prediction_length=dataset_config_dict[dataset_str]["prediction_length"],
                target_dim=target_dim,
                num_bottom_ts=S.shape[1],
                nodes=dataset_config_dict[dataset_str]["nodes"],
                is_hts=dataset_config_dict[dataset_str]["is_hts"],
                **self._get_matching_params(self.estimator),
                tags=tags,
            )

        evaluator = MultivariateEvaluator(quantiles=(np.arange(20) / 20.0)[1:], )
        agg_metrics, item_metrics = backtest_metrics(
            test_dataset,
            predictor,
            evaluator,
        )

        if dataset_str == "tourismlarge":
            cum_num_nodes_per_level = dataset_config_dict[dataset_str]["cum_num_nodes_per_level"]
        else:
            nodes = dataset_config_dict[dataset_str]['nodes']
            num_nodes_per_level = [1] + [np.asscalar(np.sum(n)) for n in nodes]
            cum_num_nodes_per_level = np.cumsum(num_nodes_per_level)

        metric_name = "mean_wQuantileLoss"
        level_wise_agg_metrics = {}
        start_ix = 0
        for l, end_ix in enumerate(cum_num_nodes_per_level):
            agg_metrics_level, _ = evaluator.get_aggregate_metrics(item_metrics.iloc[start_ix:end_ix])
            print(f"level_{l}_{metric_name}", agg_metrics_level[metric_name])

            level_wise_agg_metrics[f"level_{l}_{metric_name}"] = agg_metrics_level[metric_name]
            start_ix = end_ix

        for name in self.job_config["metrics"]:
            print(f"{name}", agg_metrics[name])

        return agg_metrics, level_wise_agg_metrics


def main(dataset_path: str, estimator: Type[Estimator], hyper_params: Dict, job_config: Optional[Dict] = None):
    if not job_config:
        job_config = dict(
            metrics=["mean_wQuantileLoss"],
            validation=False,
        )

    expt = Experiment(
        dataset_path=dataset_path,
        estimator=estimator,
        hyper_params=hyper_params,
        job_config=job_config,
    )

    agg_metrics, level_wise_agg_metrics = expt.run()
    return agg_metrics, level_wise_agg_metrics
