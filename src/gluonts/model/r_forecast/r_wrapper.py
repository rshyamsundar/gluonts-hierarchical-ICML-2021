from rpy2 import robjects, rinterface
import rpy2.robjects.packages as rpackages
import rpy2.robjects.numpy2ri
from gluonts.time_feature.seasonality import get_seasonality

import os
import numpy as np


y_bottom = np.array(
    [
        [1, 2.1, 3, 4, 5],
        [1, 2.2, 3, 4, 5],
        [1, 2.3, 3, 4, 5],
        [1, 2.4, 3, 4, 5],
    ]
)

y_bottom_r = robjects.r.matrix(
    robjects.FloatVector(y_bottom.flatten()),
    ncol=y_bottom.shape[1], nrow=y_bottom.shape[0], byrow=True
).transpose()


stats_pkg = rpackages.importr("stats")
period = get_seasonality("H")
y_bottom_r = stats_pkg.ts(y_bottom_r, frequency=period)

print(y_bottom_r)


hts_pkg = rpackages.importr('hts')
nodes = np.array([2, [2, 2]])
robjects.numpy2ri.activate()
# nodes_r = robjects.r.matrix(nodes, nrow=2, ncol=2)

nodes_r = rpy2.robjects.ListVector.from_length(2)
nodes_r[0] = rpy2.robjects.IntVector([2])
nodes_r[1] = rpy2.robjects.IntVector([2, 2])

print(nodes_r)

hts = hts_pkg.hts(y_bottom_r, nodes=nodes_r)
print(hts)


this_dir = os.path.dirname(os.path.realpath(__file__))
r_files = [
    n[:-2] for n in os.listdir(f"{this_dir}/R/") if "hierarchical" in n
]

for n in r_files:
    #try:
    robjects.r(f'source("{this_dir}/R/{n}.R")')
    #except RRuntimeError as er:
     #   raise RRuntimeError(str(er) + USAGE_MESSAGE) from er


naive_bottom_up = robjects.r["naive_bottom_up"]
params = {
    "prediction_length": 4,
    "fmethod": "arima",
}
r_params = robjects.vectors.ListVector(params)
print(r_params)

forecast = naive_bottom_up(hts, r_params)
print(forecast)


print("mint")
params = {
    "prediction_length": 4,
    "fmethod": "ets",
    "covariance": "shr",
    "nonnegative": True,
    "algorithm": "cg",
}
r_params = robjects.vectors.ListVector(params)
mint = robjects.r["mint"]
forecast = mint(hts, r_params)
print(forecast)