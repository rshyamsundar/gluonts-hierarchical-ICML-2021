# Standard library imports
import os
from typing import Dict, Iterator, List, Optional

# Third-party imports
import numpy as np
import pandas as pd
from pathlib import Path

# First-party imports
from gluonts.core.component import validated
from gluonts.dataset.common import Dataset
from gluonts.time_feature.seasonality import get_seasonality
from gluonts.model.forecast import SampleForecast
from gluonts.model.predictor import RepresentablePredictor


# https://stackoverflow.com/questions/25329955/check-if-r-is-installed-from-python
from subprocess import Popen, PIPE

proc = Popen(["which", "R"], stdout=PIPE, stderr=PIPE)
R_IS_INSTALLED = proc.wait() == 0

try:
    import rpy2.robjects.packages as rpackages
    from rpy2 import rinterface, robjects
    from rpy2.rinterface import RRuntimeError
except ImportError as e:
    rpy2_error_message = str(e)
    RPY2_IS_INSTALLED = False
else:
    RPY2_IS_INSTALLED = True

USAGE_MESSAGE = """
The RHierarchicalForecastPredictor is a thin wrapper for calling the R HTS package.
In order to use it you need to install R and run

pip install rpy2  # version: rpy2>=2.9.*,<3.*

R -e 'install.packages(c("hts"), repos="https://cloud.r-project.org")'
"""

POINT_HIERARCHICAL_METHODS = [
    "naive_bottom_up", "mint", "erm", "ermParallel"
]

SAMPLE_HIERARCHICAL_METHODS = [
    "depbu_mint"
]

SUPPORTED_METHODS = POINT_HIERARCHICAL_METHODS + SAMPLE_HIERARCHICAL_METHODS


class RHierarchicalForecastPredictor(RepresentablePredictor):
    """
    The RHierarchicalForecastPredictor is a thin wrapper for calling the R HTS package.
    In order to use it you need to install R and run

    pip install rpy2

    R -e 'install.packages(c("hts"), repos="https://cloud.r-project.org")'

    Parameters
    ----------
    freq
    prediction_length
    target_dim
    num_bottom_ts
    nodes
    tags
    is_hts
    method_name
        hierarchical forecast or reconciliation method to be used; mutst be one of:
        "mint", "naive_bottom_up", "erm", "mint_ols", "depbu_mint", "ermParallel"
    fmethod
    algorithm
    covariance
    nonnegative
    period
    correction
    seasonal
    stationary
    params
    lead_time
    """

    @validated()
    def __init__(
        self,
        freq: str,
        prediction_length: int,
        target_dim: int,
        num_bottom_ts: int,
        nodes: List,
        tags: np.ndarray,
        is_hts: bool,
        method_name: str = "mint",
        fmethod: str = "arima",
        algorithm: str = "cg",
        covariance: str = "shr",
        nonnegative: bool = False,
        period: int = None,
        correction: bool = True,
        seasonal: bool = True,
        stationary: bool = False,
        params: Optional[Dict] = None,
        lead_time: int = 0,
    ) -> None:

        super().__init__(freq=freq, prediction_length=prediction_length)

        try:
            from rpy2 import robjects, rinterface
            import rpy2.robjects.packages as rpackages
        except ImportError as e:
            raise ImportError(str(e) + USAGE_MESSAGE) from e

        if not R_IS_INSTALLED:
            raise ImportError("R is not Installed! \n " + USAGE_MESSAGE)

        if not RPY2_IS_INSTALLED:
            raise ImportError(rpy2_error_message + USAGE_MESSAGE)

        self._robjects = robjects
        self._rinterface = rinterface
        self._rinterface.initr()
        self._rpackages = rpackages

        this_dir = os.path.dirname(os.path.realpath(__file__))
        this_dir = this_dir.replace("\\", "/")  # for windows
        r_files = [
            n[:-2] for n in os.listdir(f"{this_dir}/R/") if "hierarchical" in n
        ]

        for n in r_files:
            try:
                path = Path(this_dir, "R", f"{n}.R")
                robjects.r(f'source("{path}")'.replace("\\", "\\\\"))
            except RRuntimeError as er:
                raise RRuntimeError(str(er) + USAGE_MESSAGE) from er

        assert (
            method_name in SUPPORTED_METHODS
        ), f"method {method_name} is not supported please use one of {SUPPORTED_METHODS}"

        self.method_name = method_name

        self._stats_pkg = rpackages.importr('stats')
        self._hts_pkg = rpackages.importr('hts')
        self._r_method = robjects.r[method_name]

        self.prediction_length = prediction_length
        self.freq = freq
        self.period = period if period is not None else get_seasonality(freq)
        self.target_dim = target_dim
        self.num_bottom_ts = num_bottom_ts
        self.nodes = nodes
        self.tags = tags
        self.params = {
            'prediction_length': self.prediction_length,
            'output_types': ['samples'],
            'frequency': self.period,
            "fmethod": fmethod,
            "algorithm": algorithm,
            "covariance": covariance,
            "nonnegative": nonnegative,
            "seasonal": seasonal,
            "stationary": stationary,
            "correction": correction,
        }
        if params is not None:
            self.params.update(params)
        self.lead_time = lead_time
        self.is_hts = is_hts


    def _unlist(self, l):
        if type(l).__name__.endswith("Vector") and type(l).__name__ != "ListVector":
            return [self._unlist(x) for x in l]
        elif type(l).__name__ == "ListVector":
            return [self._unlist(x) for x in l]
        elif type(l).__name__ == "Matrix":
            return np.array(l)
        else:
            return l

    def _run_r_forecast(self, d, params, save_info):
        buf = []

        def save_to_buf(x):
            buf.append(x)

        def dont_save(x):
            pass

        f = save_to_buf if save_info else dont_save

        # save output from the R console in buf
        self._rinterface.set_writeconsole_regular(f)
        self._rinterface.set_writeconsole_warnerror(f)

        params["freq_period"] = self.period
        print("method_name:", self.method_name)

        r_params = self._robjects.vectors.ListVector(params)

        if self.is_hts:
            nodes = self._robjects.ListVector.from_length(len(self.nodes))
            for idx, elem in enumerate(self.nodes):
                if not isinstance(elem, list):
                    elem = [elem]
                nodes[idx] = self._robjects.IntVector(elem)
        else:
            nodes_temp = []
            for idx, elem in enumerate(self.nodes):
                nodes_temp.extend(elem)
            nodes = self._robjects.r.matrix(self._robjects.IntVector(nodes_temp),
                                            ncol=len(self.nodes), nrow=len(self.nodes[0]), byrow=False).transpose()

        num_ts, nobs = d["target"].shape
        tags_r = self._robjects.r.matrix(self._robjects.StrVector(self.tags.flatten()),
                                           ncol=self.tags.shape[1], nrow=self.tags.shape[0], byrow=True)

        y_bottom = self._robjects.r.matrix(self._robjects.FloatVector(d["target"].flatten()),
                                      ncol=nobs, nrow=num_ts, byrow=True).transpose()

        y_bottom_ts = self._stats_pkg.ts(y_bottom, frequency=self.period)

        if self.is_hts:
            hier_ts = self._hts_pkg.hts(y_bottom_ts, nodes=nodes)  # get nodes correctly here
        else:
            hier_ts = self._hts_pkg.gts(y_bottom_ts, groups=nodes)  # get nodes correctly here

        if self.method_name == "depbu_mint":
            forecast = self._r_method(y_bottom_ts, tags_r, r_params)
        else:
            forecast = self._r_method(hier_ts, r_params)

        all_forecasts = list(forecast)
        if self.method_name in POINT_HIERARCHICAL_METHODS:
            assert len(all_forecasts) == self.target_dim * self.prediction_length
            hier_point_forecasts = np.reshape(
                list(forecast), (self.target_dim, self.prediction_length)
            ).transpose()
            hier_forecasts = np.tile(
                hier_point_forecasts, (params["num_samples"], 1, 1)
            )
        else:
            hier_forecasts = np.reshape(
                list(forecast), (params["num_samples"], self.prediction_length, self.target_dim), order='F'
            )
        forecast_dict = dict(
            samples=hier_forecasts
        )

        self._rinterface.set_writeconsole_regular(
            self._rinterface.consolePrint
        )
        self._rinterface.set_writeconsole_warnerror(
            self._rinterface.consolePrint
        )
        return forecast_dict, buf

    def predict(
        self, dataset: Dataset, num_samples=100, save_info=True, **kwargs
    ) -> Iterator[SampleForecast]:
        for data in dataset:
            data["target"] = data["target"][-self.num_bottom_ts:, :]

            if self.method_name == "depbu_mint":
                train_length = data["target"].shape[1]

                # this method expects train + test target; add dummy values for the "test" part
                data["target"] = np.concatenate(
                    (data["target"], data["target"][:, -self.prediction_length:]),
                    axis=1
                )
                self.params["train_length"] = train_length

            params = self.params.copy()
            params['num_samples'] = num_samples

            forecast_dict, console_output = self._run_r_forecast(
                data, params, save_info=save_info
            )

            if self.method_name == "depbu_mint":
                # remove the extra `prediction_length` target that was added
                data["target"] = data["target"][:, :-self.prediction_length]
                assert data['target'].shape[1] == train_length

            samples = np.array(forecast_dict['samples'])
            expected_shape = (params['num_samples'], self.prediction_length, self.target_dim)
            assert (samples.shape == expected_shape),\
                f"Expected shape {expected_shape} but found {samples.shape}"

            info = (
                {'console_output': '\n'.join(console_output)}
                if save_info
                else None
            )

            yield SampleForecast(
                samples,
                # TODO: to use gluonts.support.pandas.forecast_start(data), after updating it so that it understands
                #  multivariate target!
                pd.date_range(data["start"], periods=data["target"].shape[-1] + 1, freq=self.freq)[-1],
                self.freq,
                info=info,
                item_id=data.get("item_id", None),
            )
