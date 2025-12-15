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


from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from secretflow.device import PYUObject, proxy


@proxy(PYUObject)
@dataclass
class EarlyStopRecord:
    """
    A pure Python data structure for tracking early stopping metrics and decisions.

    This class maintains the history of metric scores and determines when training
    should stop based on the improvement (or lack thereof) of the metric.

    Parameters
    ----------
    rounds : int
        Number of rounds without improvement before stopping.
    maximize : Optional[bool]
        Whether to maximize the metric. If None, will be inferred from metric name.
    min_delta : float
        Minimum change in score to be considered an improvement.
    metric_name : Optional[str]
        Name of the metric being tracked.

    Attributes
    ----------
    history : List[float]
        Complete history of all metric scores.
    best_scores : List[float]
        History of best scores seen so far.
    best_iteration : int
        The iteration number where the best score was achieved.
    best_score : Optional[float]
        The best score achieved so far.
    current_rounds : int
        Number of consecutive rounds without improvement.
    should_stop : bool
        Whether training should stop based on current state.
    """

    rounds: int
    maximize: Optional[bool] = None
    min_delta: float = 0.0
    metric_name: Optional[str] = None

    # Internal state
    history: List[float] = field(default_factory=list)
    best_scores: List[float] = field(default_factory=list)
    best_iteration: int = 0
    best_score: Optional[float] = None
    current_rounds: int = 0
    should_stop: bool = False

    def __post_init__(self):
        """Validate parameters after initialization."""
        if self.rounds <= 0:
            raise ValueError("rounds must be greater than 0")
        if self.min_delta < 0:
            raise ValueError("min_delta must be greater or equal to 0")

        # Infer maximize setting if not explicitly provided
        if self.maximize is None:
            if self.metric_name is not None:
                self.maximize = self._infer_maximize(self.metric_name)
            else:
                # Default to minimize if we can't infer
                self.maximize = False

    def _infer_maximize(self, metric_name: str) -> bool:
        """
        Infer whether to maximize based on metric name.

        Parameters
        ----------
        metric_name : str
            Name of the metric.

        Returns
        -------
        bool
            True if metric should be maximized, False otherwise.
        """
        # Align with base implementation in callback.py
        # Only roc_auc should be maximized; mse, rmse, and tweedie metrics should be minimized
        maximize_metrics = ("roc_auc",)
        return any(metric_name.startswith(x) for x in maximize_metrics)

    def _is_improvement(self, new_score: float, best_score: float) -> bool:
        """
        Check if new score is an improvement over best score.

        Parameters
        ----------
        new_score : float
            The new score to evaluate.
        best_score : float
            The current best score.

        Returns
        -------
        bool
            True if new_score is an improvement, False otherwise.
        """
        if self.maximize:
            # For maximization: new score should be greater than best + min_delta
            return np.greater(new_score - self.min_delta, best_score)
        else:
            # For minimization: new score should be less than best - min_delta
            return np.greater(best_score - self.min_delta, new_score)

    def add_score(
        self, score: float, iteration: int
    ) -> Tuple[bool, int, Optional[float]]:
        """
        Add a new score and update early stopping state.

        Parameters
        ----------
        score : float
            The metric score for the current iteration.
        iteration : int
            The current iteration number.

        Returns
        -------
        Tuple[bool, int, Optional[float]]
            A tuple containing: (if training should stop, best iteration, best score).
        """
        # Add to complete history
        self.history.append(score)

        # First iteration
        if len(self.best_scores) == 0:
            self.best_scores.append(score)
            self.best_score = score
            self.best_iteration = iteration
            self.current_rounds = 0
            self.should_stop = False
            return False, self.best_iteration, self.best_score

        # Check if this is an improvement
        if self._is_improvement(score, self.best_scores[-1]):
            # Improved - reset counter and update best
            self.best_scores.append(score)
            self.best_score = score
            self.best_iteration = iteration
            self.current_rounds = 0
        else:
            # No improvement - keep the previous best score
            self.best_scores.append(self.best_scores[-1])
            self.current_rounds += 1

        # Check if we should stop
        if self.current_rounds >= self.rounds:
            self.should_stop = True
            return True, self.best_iteration, self.best_score

        self.should_stop = False
        return False, self.best_iteration, self.best_score

    def get_best_iteration(self) -> int:
        """
        Get the iteration number where the best score was achieved.

        Returns
        -------
        int
            The best iteration number.
        """
        return self.best_iteration

    def get_best_score(self) -> Optional[float]:
        """
        Get the best score achieved.

        Returns
        -------
        Optional[float]
            The best score, or None if no scores have been added.
        """
        return self.best_score

    def get_current_rounds_without_improvement(self) -> int:
        """
        Get the number of consecutive rounds without improvement.

        Returns
        -------
        int
            Number of rounds without improvement.
        """
        return self.current_rounds

    def reset(self) -> None:
        """Reset the early stopping record to initial state."""
        self.history.clear()
        self.best_scores.clear()
        self.best_iteration = 0
        self.best_score = None
        self.current_rounds = 0
        self.should_stop = False

    def __repr__(self) -> str:
        """String representation of the early stopping record."""
        return (
            f"EarlyStopRecord(rounds={self.rounds}, "
            f"maximize={self.maximize}, "
            f"min_delta={self.min_delta}, "
            f"metric_name={self.metric_name}, "
            f"best_score={self.best_score}, "
            f"best_iteration={self.best_iteration}, "
            f"current_rounds={self.current_rounds}, "
            f"should_stop={self.should_stop})"
        )
