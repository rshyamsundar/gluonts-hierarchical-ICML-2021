import glob
import json
import numpy as np
import os
import pickle


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


def parse_results(dataset, method, num_runs):
    results_path = f"./experiments/results/{method}/{dataset}"
    result_files = glob.glob(f"{results_path}/*.pkl")

    agg_metrics_ls, level_wise_agg_metrics_ls = [], []
    num_runs = min(num_runs, len(result_files))
    for file in result_files[:num_runs]:
        with open(file, "rb") as fp:
            agg_metrics, level_wise_agg_metrics = pickle.load(fp)
        agg_metrics_ls.append(agg_metrics)
        level_wise_agg_metrics_ls.append(level_wise_agg_metrics)

    return agg_metrics_ls, level_wise_agg_metrics_ls


def print_results(agg_metrics_ls, level_wise_agg_metrics_ls):
    num_runs = len(agg_metrics_ls)

    print(f"\n****** Results averaged over {num_runs} runs "
          f"(level-wise CRPS scores are shown first followed by the overall CRPS score): ******")

    for level_metric_name in level_wise_agg_metrics_ls[0].keys():
        level_wise_errors = [
            level_wise_agg_metric[level_metric_name]
            for level_wise_agg_metric in level_wise_agg_metrics_ls
        ]
        level_wise_mean, level_wise_std = np.mean(level_wise_errors), np.std(level_wise_errors)
        print(f"Mean +/- std. of {level_metric_name} over {num_runs} num_runs: "
              f"{level_wise_mean:.4f} +/- {level_wise_std:.4f}")

    metric_name = "mean_wQuantileLoss"
    errors = [
        agg_metrics[metric_name]
        for agg_metrics in agg_metrics_ls
    ]
    mean, std = np.mean(errors), np.std(errors)
    print(f"Mean +/- std. of {metric_name} over {num_runs} num_runs: {mean:.4f} +/- {std:.4f}")
