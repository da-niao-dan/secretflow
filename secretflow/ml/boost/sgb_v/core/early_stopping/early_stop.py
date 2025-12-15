# Copyright 2025 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Early stopping callback that manages early stopping logic on a PYU device.
"""

from typing import Optional

from secretflow.device import PYU, reveal
from secretflow.ml.boost.core.callback import CallBackCompatibleModel, EarlyStopping

from .early_stop_record import EarlyStopRecord


class EarlyStopActorCallback(EarlyStopping):
    """
    Early stopping callback that executes early stopping logic on a PYU device.

    This class inherits from EarlyStopping and overrides only the __init__ and
    _update_rounds methods to manage the early stopping state on a specific device
    using PYUObject proxy.

    Parameters
    ----------
    device : PYU
        The PYU device where the early stopping logic will be executed.
    rounds : int
        Number of rounds without improvement before stopping.
    metric_name : Optional[str], optional
        Name of the metric to use for early stopping, by default None.
    data_name : Optional[str], optional
        Name of the dataset, by default None.
    maximize : Optional[bool], optional
        Whether to maximize the metric. If None, will be inferred from metric name, by default None.
    save_best : bool, optional
        Whether to save the best model, by default False.
    min_delta : float, optional
        Minimum change in score to be considered an improvement, by default 0.0.

    Examples
    --------
    >>> from secretflow import PYU
    >>> alice = PYU('alice')
    >>> early_stop = EarlyStopActorCallback(
    ...     device=alice,
    ...     rounds=10,
    ...     metric_name="rmse",
    ...     maximize=False,
    ...     save_best=True,
    ...     min_delta=0.001
    ... )
    """

    def __init__(
        self,
        device: PYU,
        rounds: int,
        metric_name: Optional[str] = None,
        data_name: Optional[str] = None,
        maximize: Optional[bool] = None,
        save_best: bool = False,
        min_delta: float = 0.0,
    ):
        # Call parent constructor
        super().__init__(
            rounds=rounds,
            metric_name=metric_name,
            data_name=data_name,
            maximize=maximize,
            save_best=save_best,
            min_delta=min_delta,
        )

        # Additional attributes for device-based execution
        self.device = device

        # Create the early stop record on the device
        # Since EarlyStopRecord is now a @proxy(PYUObject), we can instantiate it directly
        self.record = EarlyStopRecord(
            rounds=rounds,
            maximize=maximize,
            min_delta=min_delta,
            metric_name=metric_name,
            device=device,
        )

    def _update_rounds(
        self,
        score: float,
        name: str,
        metric: str,
        model: CallBackCompatibleModel,
        epoch: int,
    ) -> bool:
        """
        Override parent's _update_rounds to use device-based record.

        Parameters
        ----------
        score : float
            The metric score for the current iteration.
        name : str
            Name of the dataset.
        metric : str
            Name of the metric.
        model : CallBackCompatibleModel
            The model being trained.
        epoch : int
            The current iteration number.

        Returns
        -------
        bool
            True if training should stop, False otherwise.
        """
        # Handle tuple scores (e.g., from cross-validation)
        if isinstance(score, tuple):
            score = score[0]

        # Call record method and reveal the results in a single round trip
        should_stop, best_iteration, best_score = reveal(
            self.record.add_score(float(score), epoch)
        )

        # Update model with best iteration info
        model.set_best_iteration_score(
            iteration=best_iteration,
            score=best_score,
        )

        return bool(should_stop)

    def __repr__(self) -> str:
        """String representation of the early stop actor callback."""
        return (
            f"EarlyStopActorCallback(device={self.device}, "
            f"rounds={self.rounds}, "
            f"metric_name={self.metric_name}, "
            f"data_name={self.data}, "
            f"save_best={self.save_best}, "
            f"min_delta={self._min_delta})"
        )
