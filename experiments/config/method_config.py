DeepVAR = dict(
    coherent_train_samples=False,
    coherent_pred_samples=False,
    likelihood_weight=1.0,
    CRPS_weight=0.0,
)

DeepVARPlus = dict(
    coherent_train_samples=False,
    coherent_pred_samples=True,
    likelihood_weight=1.0,
    CRPS_weight=0.0,
)

Ours = dict(
    coherent_train_samples=True,
    coherent_pred_samples=True,
    likelihood_weight=0.0,
    CRPS_weight=1.0,
    num_samples_for_loss=200,
    rec_weight=0.0,
    seq_axis=None,  # None for no axis iteration, choose any axis combination in a list, e.g., [1], [0, 2], [2, 1]
)

OursWarmStart = dict(
    **Ours,
    warmstart_epoch_frac=0.1,  # if `CRPS_weight > 0.0 then after `epoch_frac x epochs` CRPS loss is used.
)

OursLH = dict(
    coherent_train_samples=True,
    coherent_pred_samples=True,
    likelihood_weight=1.0,
    CRPS_weight=0.0,
    num_samples_for_loss=50,
    sample_LH=True,
)

OursWarmStartLH = dict(
    **OursLH,
    warmstart_epoch_frac=0.1,  # if `CRPS_weight > 0.0 then after `epoch_frac x epochs` CRPS loss is used.
)

ETS_BU = dict(
    method_name="naive_bottom_up",
    fmethod="ets",
    nonnegative=True,
)

ARIMA_BU = dict(
    method_name="naive_bottom_up",
    fmethod="arima",
    nonnegative=True,
)

ETS_MINT_shr = dict(
    method_name="mint",
    fmethod="ets",
    algorithm="cg",
    covariance="shr",
    nonnegative=True,
)

ETS_MINT_ols = dict(
    method_name="mint",
    fmethod="ets",
    algorithm="cg",
    covariance="ols",
    nonnegative=True,
)

ARIMA_MINT_shr = dict(
    method_name="mint",
    fmethod="arima",
    algorithm="cg",
    covariance="shr",
    nonnegative=True,
)

ARIMA_MINT_ols = dict(
    method_name="mint",
    fmethod="arima",
    algorithm="cg",
    covariance="ols",
    nonnegative=True,
)

ETS_ERM = dict(
    method_name="ermParallel", # "erm",
    fmethod="ets",
)

ARIMA_ERM = dict(
    method_name="ermParallel", # "erm",
    fmethod="arima",
)

DEPBU_MINT = dict(
    method_name="depbu_mint",
    correction=False,
    seasonal=True,
    stationary=False,
)

