import argparse
from datetime import datetime
from pathlib import Path
import pickle

from gluonts.model.deephier import DeepHierEstimator
from gluonts.model.r_forecast import RHierarchicalForecastPredictor

from config.dataset_config import dataset_config_dict
from config.method_config import *
import experiment
import utils


if __name__ == "__main__":
    """
    Run experiments from the paper as follows:
    
    python run_experiment_with_best_hps.py --dataset dataset --method method --num-runs 5
    
    where
     
        dataset is one of: [labour, traffic, tourism, tourismlarge, wiki]
        (see config/dataset_config.py)
                
        method is one of: [HierE2E, DeepVAR, DeepVARPlus, 
                            ETS_NaiveBU, ARIMA_NaiveBU,
                            ETS_MINT_shr, ETS_MINT_ols, ARIMA_MINT_shr, ARIMA_MINT_ols,
                            ETS_ERM, ARIMA_ERM,
                            PERMBU_MINT, 
                          ]
        (see config/method_config.py)
        
        num-runs: number of re-runs; default 5 
    
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
    if method in ["HierE2E", "DeepVAR", "DeepVARPlus"]:
        hyper_params = utils.get_best_hps(dataset=dataset, method=method)
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

    results_path = f"./experiments/results/{method}/{dataset}"
    Path(results_path).mkdir(parents=True, exist_ok=True)

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

        # Save results
        unique_id = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        with open(f"{results_path}/run_{unique_id}.pkl", "wb") as fp:
            pickle.dump([agg_metrics, level_wise_agg_metrics], fp)

    utils.print_results(agg_metrics_ls=agg_metrics_ls, level_wise_agg_metrics_ls=level_wise_agg_metrics_ls)