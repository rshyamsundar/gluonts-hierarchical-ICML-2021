# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

# Standard library imports
from typing import List, Tuple
from itertools import product

# Third-party imports
import mxnet as mx
import numpy as np

# First-party imports
from gluonts.core.component import validated
from gluonts.mx import Tensor
from gluonts.mx.distribution import DistributionOutput
from gluonts.mx.util import assert_shape, weighted_average
from gluonts.mx.distribution import LowrankMultivariateGaussian
from gluonts.model.deepvar._network import DeepVARNetwork


class DeepHierNetwork(DeepVARNetwork):
    @validated()
    def __init__(
        self,
        M,
        A,
        num_layers: int,
        num_cells: int,
        cell_type: str,
        history_length: int,
        context_length: int,
        prediction_length: int,
        distr_output: DistributionOutput,
        dropout_rate: float,
        lags_seq: List[int],
        target_dim: int,
        conditioning_length: int,
        cardinality: List[int] = [1],
        embedding_dimension: int = 1,
        scaling: bool = True,
        seq_axis: List[int] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            num_layers=num_layers,
            num_cells=num_cells,
            cell_type=cell_type,
            history_length=history_length,
            context_length=context_length,
            prediction_length=prediction_length,
            distr_output=distr_output,
            dropout_rate=dropout_rate,
            lags_seq=lags_seq,
            target_dim=target_dim,
            conditioning_length=conditioning_length,
            cardinality=cardinality,
            embedding_dimension=embedding_dimension,
            scaling=scaling,
            **kwargs
        )

        self.M = M
        self.A = A
        self.seq_axis = seq_axis

    def reconcile_samples(self, samples):
        """
        Computes coherent samples by projecting unconstrained `samples` using the matrix `self.M`.

        Parameters
        ----------
        samples
            Unconstrained samples.
            Shape: (num_samples, batch_size, seq_len, num_ts) during training and
                   (num_parallel_samples x batch_size, seq_len, num_ts) during prediction.

        Returns
        -------
        Coherent samples
            Tensor, shape same as that of `samples`.

        """

        if self.seq_axis:
            # bring the axis to iterate in the beginning
            samples = mx.nd.moveaxis(samples, self.seq_axis, list(range(len(self.seq_axis))))

            out = [
                mx.nd.dot(samples[idx], self.M, transpose_b=True)
                for idx in product(*[range(x) for x in [samples.shape[d] for d in range(len(self.seq_axis))]])
            ]

            # put the axis in the correct order again
            out = mx.nd.concat(*out, dim=0).reshape(samples.shape)
            out = mx.nd.moveaxis(out, list(range(len(self.seq_axis))), self.seq_axis)
            return out
        else:
            return mx.nd.dot(samples, self.M, transpose_b=True)

    def train_hybrid_forward(
        self,
        F,
        target_dimension_indicator: Tensor,
        past_time_feat: Tensor,
        past_target_cdf: Tensor,
        past_observed_values: Tensor,
        past_is_pad: Tensor,
        future_time_feat: Tensor,
        future_target_cdf: Tensor,
        future_observed_values: Tensor,
        epoch_frac: float,
    ) -> Tuple[Tensor, ...]:
        """
        Computes the loss for training DeepVAR, all inputs tensors representing
        time series have NTC layout.

        Parameters
        ----------
        F
        target_dimension_indicator
            Indices of the target dimension (batch_size, target_dim)
        past_time_feat
            Dynamic features of past time series (batch_size, history_length,
            num_features)
        past_target_cdf
            Past marginal CDF transformed target values (batch_size,
            history_length, target_dim)
        past_observed_values
            Indicator whether or not the values were observed (batch_size,
            history_length, target_dim)
        past_is_pad
            Indicator whether the past target values have been padded
            (batch_size, history_length)
        future_time_feat
            Future time features (batch_size, prediction_length, num_features)
        future_target_cdf
            Future marginal CDF transformed target values (batch_size,
            prediction_length, target_dim)
        future_observed_values
            Indicator whether or not the future values were observed
            (batch_size, prediction_length, target_dim)

        Returns
        -------
        distr
            Loss with shape (batch_size, 1)
        likelihoods
            Likelihoods for each time step
            (batch_size, context + prediction_length, 1)
        distr_args
            Distribution arguments (context + prediction_length,
            number_of_arguments)
        """

        seq_len = self.context_length + self.prediction_length

        # unroll the decoder in "training mode", i.e. by providing future data
        # as well
        rnn_outputs, _, scale, lags_scaled, inputs = self.unroll_encoder(
            F=F,
            past_time_feat=past_time_feat,
            past_target_cdf=past_target_cdf,
            past_observed_values=past_observed_values,
            past_is_pad=past_is_pad,
            future_time_feat=future_time_feat,
            future_target_cdf=future_target_cdf,
            target_dimension_indicator=target_dimension_indicator,
        )

        # put together target sequence
        # (batch_size, seq_len, target_dim)
        target = F.concat(
            past_target_cdf.slice_axis(
                axis=1, begin=-self.context_length, end=None
            ),
            future_target_cdf,
            dim=1,
        )

        # assert_shape(target, (-1, seq_len, self.target_dim))

        distr, distr_args = self.distr(
            time_features=inputs,
            rnn_outputs=rnn_outputs,
            scale=scale,
            lags_scaled=lags_scaled,
            target_dimension_indicator=target_dimension_indicator,
            seq_len=self.context_length + self.prediction_length,
        )

        # Assert CRPS_weight, likelihood_weight, and coherent_train_samples have harmonious values
        assert self.CRPS_weight >= 0.0, 'CRPS weight must be non-negative'
        assert self.likelihood_weight >= 0.0, 'Likelihood weight must be non-negative!'
        assert self.likelihood_weight + self.CRPS_weight > 0.0, 'At least one of CRPS or likelihood weights must be non-zero'
        if self.CRPS_weight == 0.0 and self.coherent_train_samples:
            assert 'No sampling being performed. coherent_train_samples flag is ignored'
        if not self.sample_LH == 0.0 and self.coherent_train_samples:
            assert 'No sampling being performed. coherent_train_samples flag is ignored'
        if self.likelihood_weight == 0.0 and self.sample_LH:\
            assert 'likelihood_weight is 0 but sample likelihoods are still being calculated. Set sample_LH=0 when likelihood_weight=0'

        # Sample from multivariate Gaussian distribution if we are using CRPS or LH-sample loss
        # dim: (num_samples, batch_size, seq_len, m)
        if self.sample_LH or (self.CRPS_weight > 0.0):
            raw_samples = distr.sample_rep(num_samples=self.num_samples_for_loss, dtype='float32')

            # Only project during training if we have already sampled
            if self.coherent_train_samples and epoch_frac > self.warmstart_epoch_frac:
                coherent_samples = self.reconcile_samples(raw_samples)
                assert_shape(coherent_samples, raw_samples.shape)
                samples = coherent_samples
            else:
                samples = raw_samples

        # Compute likelihoods (always do this step)
        # we sum the last axis to have the same shape for all likelihoods
        # (batch_size, seq_len, 1)
        # calculates likelihood of NN prediction under the current learned distribution parameters
        if self.sample_LH: # likelihoods on samples
            # Compute mean and variance
            mu = samples.mean(axis=0)
            var = mx.nd.square(samples - samples.mean(axis=0)).mean(axis=0)
            likelihoods = -LowrankMultivariateGaussian(
                        dim=samples.shape[-1], rank=0, mu=mu, D=var
                            ).log_prob(target).expand_dims(axis=-1)
        else: # likelihoods on network params
            likelihoods = -distr.log_prob(target).expand_dims(axis=-1)
        assert_shape(likelihoods, (-1, seq_len, 1))

        # Pick loss function approach. This avoids sampling if we are only training with likelihoods on params
        if self.CRPS_weight > 0.0:  # and epoch_frac > self.warmstart_epoch_frac:
            loss_CRPS = distr.crps(samples, target)
            loss_unmasked = self.CRPS_weight * loss_CRPS + self.likelihood_weight * likelihoods
        else:  # CRPS_weight = 0.0 (asserted non-negativity above)
          loss_unmasked = likelihoods
              
        # get mask values
        past_observed_values = F.broadcast_minimum(
            past_observed_values, 1 - past_is_pad.expand_dims(axis=-1)
        )

        # (batch_size, subseq_length, target_dim)
        observed_values = F.concat(
            past_observed_values.slice_axis(
                axis=1, begin=-self.context_length, end=None
            ),
            future_observed_values,
            dim=1,
        )

        # mask the loss at one time step if one or more observations is missing
        # in the target dimensions (batch_size, subseq_length, 1)
        loss_weights = observed_values.min(axis=-1, keepdims=True)

        assert_shape(loss_weights, (-1, seq_len, 1)) #-1 is batch axis size

        loss = weighted_average(
            F=F, x=loss_unmasked, weights=loss_weights, axis=1
        )

        assert_shape(loss, (-1, -1, 1))

        self.distribution = distr

        return (loss, likelihoods) + distr_args

    def reconciliation_error(self, samples):
        r"""
        Computes the maximum relative reconciliation error among all the aggregated time series

        .. math::

                        \max_i \frac{|y_i - s_i|} {|y_i|},

        where :math:`i` refers to the aggregated time series index, :math:`y_i` is the (direct) forecast obtained for
        the :math:`i^{th}` time series and :math:`s_i` is its aggregated forecast obtained by summing the corresponding
        bottom-level forecasts. If :math:`y_i` is zero, then the absolute difference, :math:`|s_i|`, is used instead.

        This can be comupted as follows given the constraint matrix A:

        .. math::

                        \max \frac{|A \times samples|} {|samples[:r]|},

        where :math:`r` is the number aggregated time series.

        Parameters
        ----------
        samples
            Samples. Shape: `(*batch_shape, target_dim)`.

        Returns
        -------
        Float
            Reconciliation error


        """

        num_agg_ts = self.A.shape[0]
        forecasts_agg_ts = samples.slice_axis(
            axis=-1, begin=0, end=num_agg_ts
        ).asnumpy()

        abs_err = mx.nd.abs(mx.nd.dot(samples, self.A, transpose_b=True)).asnumpy()
        rel_err = np.where(
            forecasts_agg_ts == 0,
            abs_err,
            abs_err / np.abs(forecasts_agg_ts),
        )

        return np.max(rel_err)

    def sampling_decoder(
        self,
        F,
        past_target_cdf: Tensor,
        target_dimension_indicator: Tensor,
        time_feat: Tensor,
        scale: Tensor,
        begin_states: List[Tensor],
    ) -> Tensor:
        """
        Computes sample paths by unrolling the RNN starting with a initial
        input and state.

        Parameters
        ----------
        past_target_cdf
            Past marginal CDF transformed target values (batch_size,
            history_length, target_dim)
        target_dimension_indicator
            Indices of the target dimension (batch_size, target_dim)
        time_feat
            Dynamic features of future time series (batch_size, history_length,
            num_features)
        scale
            Mean scale for each time series (batch_size, 1, target_dim)
        begin_states
            List of initial states for the RNN layers (batch_size, num_cells)
        Returns
        --------
        sample_paths : Tensor
            A tensor containing sampled paths. Shape: (1, num_sample_paths,
            prediction_length, target_dim).
        """

        def repeat(tensor):
            return tensor.repeat(repeats=self.num_parallel_samples, axis=0)

        # blows-up the dimension of each tensor to
        # batch_size * self.num_sample_paths for increasing parallelism
        repeated_past_target_cdf = repeat(past_target_cdf)
        repeated_time_feat = repeat(time_feat)
        repeated_scale = repeat(scale)
        repeated_target_dimension_indicator = repeat(
            target_dimension_indicator
        )

        # slight difference for GPVAR and DeepVAR, in GPVAR, its a list
        repeated_states = self.make_states(begin_states)

        future_samples = []

        # for each future time-units we draw new samples for this time-unit
        # and update the state
        for k in range(self.prediction_length):
            lags = self.get_lagged_subsequences(
                F=F,
                sequence=repeated_past_target_cdf,
                sequence_length=self.history_length + k,
                indices=self.shifted_lags,
                subsequences_length=1,
            )

            rnn_outputs, repeated_states, lags_scaled, inputs = self.unroll(
                F=F,
                begin_state=repeated_states,
                lags=lags,
                scale=repeated_scale,
                time_feat=repeated_time_feat.slice_axis(
                    axis=1, begin=k, end=k + 1
                ),
                target_dimension_indicator=repeated_target_dimension_indicator,
                unroll_length=1,
            )

            distr, distr_args = self.distr(
                time_features=inputs,
                rnn_outputs=rnn_outputs,
                scale=repeated_scale,
                target_dimension_indicator=repeated_target_dimension_indicator,
                lags_scaled=lags_scaled,
                seq_len=1,
            )

            # (num_parallel_samples*batch_size, 1, m)
            # new_samples are not coherent (initially)
            new_incoherent_samples = distr.sample()

            # reconcile new_incoherent_samples if coherent_pred_samples=True, use new_incoherent_samples if False
            if self.coherent_pred_samples:
                new_coherent_samples = self.reconcile_samples(new_incoherent_samples)

                assert_shape(new_coherent_samples, new_incoherent_samples.shape)

                if self.compute_reconciliation_error:
                    recon_err = self.reconciliation_error(samples=new_coherent_samples)
                    print(f"Reconciliation error for prediction time step t={k + 1}: {recon_err}")

                new_samples = new_coherent_samples
            else:
                new_samples = new_incoherent_samples

            # (batch_size, seq_len, target_dim)
            future_samples.append(new_samples)
            repeated_past_target_cdf = F.concat(
                repeated_past_target_cdf, new_samples, dim=1
            )

        # (batch_size * num_samples, prediction_length, target_dim)
        samples = F.concat(*future_samples, dim=1)

        # (batch_size, num_samples, prediction_length, target_dim)
        return samples.reshape(
            shape=(
                -1,
                self.num_parallel_samples,
                self.prediction_length,
                self.target_dim,
            )
        )


class DeepHierTrainingNetwork(DeepHierNetwork):

    def __init__(
        self,
        num_samples_for_loss: int,
        likelihood_weight: float,
        CRPS_weight: float,
        coherent_train_samples: bool,
        warmstart_epoch_frac: float,
        sample_LH: bool,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.num_samples_for_loss = num_samples_for_loss
        self.likelihood_weight = likelihood_weight
        self.CRPS_weight = CRPS_weight
        self.coherent_train_samples = coherent_train_samples
        self.warmstart_epoch_frac = warmstart_epoch_frac
        self.sample_LH = sample_LH

        # noinspection PyMethodOverriding,PyPep8Naming
    def hybrid_forward(
        self,
        F,
        target_dimension_indicator: Tensor,
        past_time_feat: Tensor,
        past_target_cdf: Tensor,
        past_observed_values: Tensor,
        past_is_pad: Tensor,
        future_time_feat: Tensor,
        future_target_cdf: Tensor,
        future_observed_values: Tensor,
        epoch_frac: float,
    ) -> Tuple[Tensor, ...]:
        """
        Computes the loss for training DeepVAR, all inputs tensors representing
        time series have NTC layout.

        Parameters
        ----------
        F
        target_dimension_indicator
            Indices of the target dimension (batch_size, target_dim)
        past_time_feat
            Dynamic features of past time series (batch_size, history_length,
            num_features)
        past_target_cdf
            Past marginal CDF transformed target values (batch_size,
            history_length, target_dim)
        past_observed_values
            Indicator whether or not the values were observed (batch_size,
            history_length, target_dim)
        past_is_pad
            Indicator whether the past target values have been padded
            (batch_size, history_length)
        future_time_feat
            Future time features (batch_size, prediction_length, num_features)
        future_target_cdf
            Future marginal CDF transformed target values (batch_size,
            prediction_length, target_dim)
        future_observed_values
            Indicator whether or not the future values were observed
            (batch_size, prediction_length, target_dim)

        Returns
        -------
        distr
            Loss with shape (batch_size, 1)
        likelihoods
            Likelihoods for each time step
            (batch_size, context + prediction_length, 1)
        distr_args
            Distribution arguments (context + prediction_length,
            number_of_arguments)
        """
        return self.train_hybrid_forward(
            F,
            target_dimension_indicator,
            past_time_feat,
            past_target_cdf,
            past_observed_values,
            past_is_pad,
            future_time_feat,
            future_target_cdf,
            future_observed_values,
            epoch_frac,
        )


class DeepHierPredictionNetwork(DeepHierNetwork):
    @validated()
    def __init__(
        self,
        num_parallel_samples: int,
        compute_reconciliation_error: bool,
        coherent_pred_samples: bool,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.num_parallel_samples = num_parallel_samples
        self.compute_reconciliation_error = compute_reconciliation_error
        self.coherent_pred_samples=coherent_pred_samples

        # for decoding the lags are shifted by one,
        # at the first time-step of the decoder a lag of one corresponds to
        # the last target value
        self.shifted_lags = [l - 1 for l in self.lags_seq]

    # noinspection PyMethodOverriding,PyPep8Naming
    def hybrid_forward(
        self,
        F,
        target_dimension_indicator: Tensor,
        past_time_feat: Tensor,
        past_target_cdf: Tensor,
        past_observed_values: Tensor,
        past_is_pad: Tensor,
        future_time_feat: Tensor,
    ) -> Tensor:
        """
        Predicts samples given the trained DeepVAR model.
        All tensors should have NTC layout.
        Parameters
        ----------
        F
        target_dimension_indicator
            Indices of the target dimension (batch_size, target_dim)
        past_time_feat
            Dynamic features of past time series (batch_size, history_length,
            num_features)
        past_target_cdf
            Past marginal CDF transformed target values (batch_size,
            history_length, target_dim)
        past_observed_values
            Indicator whether or not the values were observed (batch_size,
            history_length, target_dim)
        past_is_pad
            Indicator whether the past target values have been padded
            (batch_size, history_length)
        future_time_feat
            Future time features (batch_size, prediction_length, num_features)

        Returns
        -------
        sample_paths : Tensor
            A tensor containing sampled paths (1, num_sample_paths,
            prediction_length, target_dim).

        """
        return self.predict_hybrid_forward(
            F=F,
            target_dimension_indicator=target_dimension_indicator,
            past_time_feat=past_time_feat,
            past_target_cdf=past_target_cdf,
            past_observed_values=past_observed_values,
            past_is_pad=past_is_pad,
            future_time_feat=future_time_feat,
        )
