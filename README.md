# Code for "End-to-End Learning of Coherent Probabilistic Forecasts for Hierarchical Time Series" 


This is a fork of [GluonTS](https://github.com/awslabs/gluon-ts/tree/master) accompanying the paper 
"End-to-End Learning of Coherent Probabilistic Forecasts for Hierarchical Time Series" presented at ICML 2021.

## Setup

The code is written in [GluonTS](https://github.com/awslabs/gluon-ts/tree/master), 
we recommend installing it the following way 

```
pip install --upgrade mxnet
git clone https://github.com/rshyamsundar/gluonts-hierarchical-ICML-2021.git
cd gluonts-hierarchical-ICML-2021
pip install -e .
```

The model will also be released in GluonTS mainline, this fork is created to keep a version with results as close as possible to 
the one published in the paper. We do encourage you, however, to try out GluonTS mainline as well; due to code changes on mainline, results may 
change over time there.

(**Skip this step if you want to run only our method.**) We also provide a python wrapper for running existing hierarchical methods that were implemented in R.
To run them, `rpy2` must be installed along with R and `hts` package: 

```
pip install rpy2==2.9
pip install jinja2
R -e 'install.packages(c("hts"), repos="https://cloud.r-project.org")'
``` 
For running the competing method `PERMBU_MINT`, more packages need to be installed; in case of any issues, check R-specific [README.md](https://github.com/rshyamsundar/gluonts-hierarchical-ICML-2021/tree/master/src/gluonts/model/r_forecast/R), which provides help for both ubuntu and mac environments.
``` 
R -e 'install.packages(c("here", "SGL", "matrixcalc", "igraph", "gsl", "copula", "sn", "scoringRules", "fBasics", "msm", "gtools", "lubridate", "forecast", "abind", "glmnet", "propagate", "SuppDists"))'
```

## Running

All the methods compared in the paper can be run as follows. Our method is denoted as "HierE2E".

```
python experiments/run_experiment_with_best_hps.py --dataset dataset --method method
```
where dataset is one of `{labour, traffic, tourism, tourismlarge, wiki}` and method is one of `{HierE2E, DeepVAR, DeepVARPlus, ETS_NaiveBU, ARIMA_NaiveBU, ETS_MINT_shr, ETS_MINT_ols, ARIMA_MINT_shr, ARIMA_MINT_ols, ETS_ERM, ARIMA_ERM, PERMBU_MINT}`.                        


This will run the selected method 5 times on the selected dataset with the hyperparameters used in the paper. This script also saves the results (level-wise as well as overall scores) in `experiments/results`.

One can also limit the number of repetitions of the same method using the command line argument `num-runs`:

```
python experiments/run_experiment_with_best_hps.py --dataset dataset --method method --num-runs 1
```
This allows doing the multiple runs of the same method in parallel.

The following script fetches the saved results of previous runs and prints the mean and standard deviation over multiple runs (controlled by `num-runs`):

```
python experiments/show_results.py --dataset dataset --method method --num-runs 5 
```
If results are available for fewer number of runs, then mean/std is calculated over only those results available in `experiments/results` folder.

## Citing

If the datasets, benchmark, or methods are useful for your research, you can cite the following paper:

```

@InProceedings{pmlr-v139-rangapuram21a,
  title = 	 {End-to-End Learning of Coherent Probabilistic Forecasts for Hierarchical Time Series},
  author =       {Rangapuram, Syama Sundar and Werner, Lucien D and Benidis, Konstantinos and Mercado, Pedro and Gasthaus, Jan and Januschowski, Tim},
  booktitle = 	 {Proceedings of the 38th International Conference on Machine Learning},
  pages = 	 {8832--8843},
  year = 	 {2021},
  editor = 	 {Meila, Marina and Zhang, Tong},
  volume = 	 {139},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {18--24 Jul},
  publisher =    {PMLR},
  pdf = 	 {http://proceedings.mlr.press/v139/rangapuram21a/rangapuram21a.pdf},
  url = 	 {http://proceedings.mlr.press/v139/rangapuram21a.html},
  abstract = 	 {This paper presents a novel approach for hierarchical time series forecasting that produces coherent, probabilistic forecasts without requiring any explicit post-processing reconciliation. Unlike the state-of-the-art, the proposed method simultaneously learns from all time series in the hierarchy and incorporates the reconciliation step into a single trainable model. This is achieved by applying the reparameterization trick and casting reconciliation as an optimization problem with a closed-form solution. These model features make end-to-end learning of hierarchical forecasts possible, while accomplishing the challenging task of generating forecasts that are both probabilistic and coherent. Importantly, our approach also accommodates general aggregation constraints including grouped and temporal hierarchies. An extensive empirical evaluation on real-world hierarchical datasets demonstrates the advantages of the proposed approach over the state-of-the-art.}
}

```
