import argparse
import json
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

    args, _ = parser.parse_known_args()
    print(args)

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

    print(hyper_params)

    job_config=dict(
        metrics=["mean_wQuantileLoss"],
        validation=False,
    )

    NUM_RUNS = 5
    for i in range(NUM_RUNS):
        experiment.main(
            dataset_path=f'./data/{dataset}',
            estimator=estimator,
            hyper_params=hyper_params,
            job_config=job_config
        )
