import argparse
import json
import numpy as np
import os

from gluonts.model.deephier import DeepHierEstimator
from gluonts.model.r_forecast import RHierarchicalForecastPredictor

from config.dataset_config import dataset_config_dict
from config.method_config import *
import experiment


def get_best_hps(dataset, method):
    cwd = os.path.dirname(os.path.realpath(__file__))
    ix = cwd.rfind('/')
    source_dir = cwd[: ix + 1]

    best_hps_by_dataset = {}
    with open(f'{source_dir}/experiments/best_hps/best_hps_{method}.jsonl') as fp:
        for line in fp.readlines():
            d = json.loads(line)
            best_hps_by_dataset.update(json.loads(d))

    return best_hps_by_dataset[dataset]


if __name__ == "__main__":
    """
    Run experiments from the paper as follows:
    
    python run_experiment_with_best_hps.py --dataset dataset --method method
    
    where
     
        dataset is one of: [labour, traffic, tourism, tourismlarge, wiki]
        (see config/dataset_config.py)
        
        and 
        
        method is one of: [HierE2E, DeepVAR, DeepVARPlus, 
                            ETS_BU, ARIMA_BU,
                            ETS_MINT_shr, ETS_MINT_ols, ARIMA_MINT_shr, ARIMA_MINT_ols,
                            ETS_ERM, ARIMA_ERM,
                            DEPBU_MINT, 
                          ]
        (see config/method_config.py) 
    
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--method", type=str, required=True)
    parser.add_argument("--num-runs", type=int, required=False, default=5)

    args, _ = parser.parse_known_args()

    dataset = args.dataset
    if dataset == "wiki":
        dataset = "wiki2"
    elif dataset == "tourism":
        dataset = "tourismsmall"

    method = args.method
    if method in ["hierE2E", "DeepVAR", "DeepVARPlus"]:
        hyper_params = get_best_hps(dataset=dataset, method=method)
        estimator = DeepHierEstimator
    else:
        # Each combination of forecasting method and reconciliation strategy is run separately.
        # Base forecasting method is auto-tuned by the R package internally.
        hyper_params = eval(method)
        estimator = RHierarchicalForecastPredictor

    num_runs = args.num_runs
    print(f"Running {method} on {dataset} dataset {num_runs} time(s)")
    print(hyper_params)

    job_config=dict(
        metrics=["mean_wQuantileLoss"],
        validation=False,
    )

    agg_metrics_ls = []
    level_wise_agg_metrics_ls = []
    for i in range(num_runs):
        print(f"********* Run {i+1} *********")
        agg_metrics, level_wise_agg_metrics = experiment.main(
            dataset_path=f'./experiments/data/{dataset}',
            estimator=estimator,
            hyper_params=hyper_params,
            job_config=job_config
        )

        agg_metrics_ls.append(agg_metrics)
        level_wise_agg_metrics_ls.append(level_wise_agg_metrics)

    print(f"\n****** Results averaged over {num_runs} runs "
          f"(level-wise errors are shown first followed by the overall error): ******")

    for metric_name in job_config["metrics"]:
        for level_metric_name in level_wise_agg_metrics_ls[0].keys():
            errors = [
                level_wise_agg_metric[level_metric_name]
                for level_wise_agg_metric in level_wise_agg_metrics_ls
            ]
            print(f"Mean +/- std. of {level_metric_name} over {num_runs} num_runs: {np.mean(errors)} +/- {np.std(errors)}")

    for metric_name in job_config["metrics"]:
        errors = [
            agg_metrics[metric_name]
            for agg_metrics in agg_metrics_ls
        ]
        print(f"Mean +/- std. of {metric_name} over {num_runs} num_runs: {np.mean(errors)} +/- {np.std(errors)}")
