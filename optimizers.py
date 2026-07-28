import copy
import math
import GPyOpt
from pyDOE import lhs
from GPyOpt import Design_space
import GPy
from scipy.optimize import minimize

import itertools

from abc import abstractmethod
import time  #TODO delete this debugging only
from abc import ABC
from abc import abstractmethod
import threading

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re


class OptimizationModel():
    '''
    A model for optimizing experimental parameters within given bounds using Bayesian optimization.
    It aims to find the optimal set of parameters that reach a specified target value for a given objective.
    
    ATTRIBUTES:
    list<dict> bounds: The bounds for each parameter in the optimization.
    float target_value: The target value the optimization tries to achieve.
    dict reagent_info: Information about reagents including their volumes or other relevant properties.
    list<tuple> fixed_reagents: Fixed reagents and their volumes that are used in each experiment setup.
    GPyOpt.Design_space space: The design space defined by the bounds for the optimization.
    int initial_design_numdata: The number of points in the initial design.
    int batch_size: The number of experiments to suggest in each optimization iteration.
    list experiment_data: A dictionary to store the experimental data ('X' for inputs and 'Y' for outputs).
    float threshold: The threshold for determining when the optimization should stop.
    bool quit: Flag to indicate whether the optimization process should stop.
    list acquisition_functions: List of acquisition functions to be used for suggesting experiments.
    str acquisition_mode:
        Canonical target-aware acquisition mode.
    float balanced_exploration_weight:
        Dimensionless uncertainty weight used by balanced mode.
    float incumbent_target_error_nm:
        Best QC-approved condition-level target error available to target EI.
    int current_acquisition_index: The current index pointing to the acquisition function in use.
    int curr_iter: The current iteration of the optimization process.
    int max_iters: The maximum number of iterations for the optimization process.
    GPyOpt.models.GPModel model: The Gaussian Process model used for optimization.
    GPyOpt.core.evaluators.Sequential evaluator: The evaluator used to apply the acquisition function.
    GPyOpt.methods.ModularBayesianOptimization optimizer: The modular Bayesian optimization method.
    
    METHODS:
    __init__(self, bounds, target_value, reagent_info, fixed_reagents, initial_design_numdata=15, batch_size=3, max_iters=10) -> None: Initializes the optimization model.
    check_bounds(self, suggestion) -> bool: Checks if a suggested experiment is within the bounds.
    generate_initial_design(self) -> np.ndarray: Generates the initial design for the optimization.
    initialize_optimizer(self, X_init, Y_init) -> None: Initializes the Bayesian Optimization components with initial data.
    _update_acquisition(self) -> None: Updates the acquisition function based on the current index.
    suggest_next_locations(self) -> np.ndarray: Suggests the next locations for experimentation.
    update_experiment_data(self, X_new, Y_new) -> None: Updates the model with new experimental data.
    calc_obj(self, x) -> np.ndarray: Calculates the objective function.
    update_quit(self, X_new, Y_new) -> None: Updates the quit parameter based on optimization progress.
    '''
    _SUPPORTED_ACQUISITION_MODES = (
        'exploit',
        'explore',
        'balanced',
        'target_ei'
    )

    IMPLEMENTED_ACQUISITION_MODES = (
        'exploit',
        'explore',
        'balanced',
        'target_ei'
    )

    def __init__(
        self,
        bounds,
        target_value,
        reagent_info,
        fixed_reagents,
        variable_reagents,
        initial_design_numdata,
        batch_size,
        max_iters,
        min_conc=None,
        max_conc=None,
        total_volume=None,
        fixed_reagent_volumes=None,
        allow_true_zero=False,
        true_zero_reagents=None,
        acquisition_mode='exploit',
        balanced_exploration_weight=1.0,
        terminal_verbosity='standard'
    ):
        '''
        Initializes the Auto optimization model.

        In addition to the normalized optimization bounds, this stores physical
        recipe context needed for true-zero-aware and volume-aware optimization.
        The controller remains the final source of truth for robot execution,
        but the optimizer needs the same concentration/volume context so it can
        avoid treating non-executable regions as normal search space.

        params:
            list[dict] bounds:
                GPyOpt bounds for normalized model-space variables.

            float target_value:
                Target lambda max in nm.

            pd.DataFrame reagent_info:
                Reagent dataframe from the controller. Used to look up deck
                stock concentrations for variable reagents.

            list fixed_reagents:
                Fixed reagent names used by the controller.

            list variable_reagents:
                Names of variable reagents being optimized.

            int initial_design_numdata:
                Number of unique initial design points.

            int batch_size:
                Number of unique model-suggested recipes per iteration.

            int max_iters:
                Maximum number of Auto optimization iterations.

            list[float] min_conc:
                Minimum target concentration for each variable reagent.

            list[float] max_conc:
                Maximum target concentration for each variable reagent.

            float total_volume:
                Final/template reaction volume in uL.

            dict fixed_reagent_volumes:
                Fixed reagent names as keys and fixed transfer volumes in uL
                as values. Used to calculate remaining well volume during
                optimizer-side feasibility checks.

            bool allow_true_zero:
                If True, the mixed mask optimizer may turn variable reagents
                OFF as exact-zero transfers. If False, all variable reagents
                remain ON and are optimized only in executable transfer ranges.

            list[str] or None true_zero_reagents:
                Optional canonical variable-reagent names eligible for exact
                zero. When omitted, allow_true_zero preserves its legacy
                all-or-none behavior. Reagents not listed here remain ON.

            str acquisition_mode:
                Canonical Auto acquisition mode supplied by the controller.
                Older callers default to exploit. Modes that are recognized
                but not yet implemented remain blocked before recipe selection.

            float balanced_exploration_weight:
                Dimensionless coefficient multiplying GP predictive standard
                deviation in the balanced acquisition score. The default 1.0
                trades one nanometer of target error against one nanometer of
                predictive uncertainty. Values must be finite and nonnegative.

            str terminal_verbosity:
                Controller-normalized Auto terminal-output tier. Essential
                suppresses routine optimizer detail, standard emits concise
                acquisition summaries, and diagnostic additionally emits
                per-mask and raw normalized-recipe diagnostics.
        '''
        self.bounds = bounds
        self.target_value = target_value
        self.reagent_info = reagent_info
        self.fixed_reagents = fixed_reagents
        self.variable_reagents = variable_reagents
        self.space = GPyOpt.Design_space(bounds)

        self.initial_design_numdata = initial_design_numdata
        self.batch_size = batch_size
        self.experiment_data = {'X': [], 'Y': []}
        self.threshold = 1
        self.quit = False
        self.acquisition_functions = ['EI', 'MPI', 'LCB']
        self.current_acquisition_index = 0
        self.curr_iter = 0
        self.max_iters = max_iters

        # Physical recipe context used by true-zero-aware and volume-aware
        # optimization. These are optional for backwards compatibility, but
        # should be provided by Auto mode before physical feasibility checks are
        # enabled in the optimizer.
        self.min_conc = min_conc
        self.max_conc = max_conc
        self.total_volume = total_volume
        self.fixed_reagent_volumes = fixed_reagent_volumes
        self.allow_true_zero = bool(allow_true_zero)

        if true_zero_reagents is None:
            resolved_true_zero_reagents = (
                list(self.variable_reagents)
                if self.allow_true_zero
                else []
            )
        else:
            if isinstance(true_zero_reagents, str):
                raise ValueError(
                    "true_zero_reagents must be a list of variable reagent "
                    "names, not one unparsed string."
                )

            resolved_true_zero_reagents = [
                str(reagent_name)
                for reagent_name in true_zero_reagents
            ]
            available_reagents = {
                str(reagent_name)
                for reagent_name in self.variable_reagents
            }
            unknown_reagents = [
                reagent_name
                for reagent_name in resolved_true_zero_reagents
                if reagent_name not in available_reagents
            ]

            if unknown_reagents:
                raise ValueError(
                    "true_zero_reagents contains unknown variable reagent(s): "
                    + ', '.join(unknown_reagents)
                )

            if len(set(resolved_true_zero_reagents)) != len(
                resolved_true_zero_reagents
            ):
                raise ValueError(
                    "true_zero_reagents must not contain duplicate reagent "
                    "names."
                )

        if self.allow_true_zero and len(resolved_true_zero_reagents) == 0:
            raise ValueError(
                "allow_true_zero requires at least one eligible variable "
                "reagent."
            )

        if not self.allow_true_zero and resolved_true_zero_reagents:
            raise ValueError(
                "true_zero_reagents requires allow_true_zero to be True."
            )

        self.true_zero_reagents = resolved_true_zero_reagents
        self.true_zero_reagent_indices = [
            reagent_index
            for reagent_index, reagent_name in enumerate(
                self.variable_reagents
            )
            if str(reagent_name) in self.true_zero_reagents
        ]

        if acquisition_mode not in self._SUPPORTED_ACQUISITION_MODES:
            raise ValueError(
                "OptimizationModel acquisition_mode must be one of: "
                "exploit, explore, balanced, or target_ei. "
                f"Received: {acquisition_mode!r}."
            )

        self.acquisition_mode = acquisition_mode

        try:
            balanced_exploration_weight = float(
                balanced_exploration_weight
            )
        except (TypeError, ValueError):
            raise ValueError(
                "balanced_exploration_weight must be a finite, nonnegative "
                "number. Received: "
                f"{balanced_exploration_weight!r}."
            )

        if (
            not math.isfinite(balanced_exploration_weight)
            or balanced_exploration_weight < 0.0
        ):
            raise ValueError(
                "balanced_exploration_weight must be a finite, nonnegative "
                "number. Received: "
                f"{balanced_exploration_weight!r}."
            )

        self.balanced_exploration_weight = (
            balanced_exploration_weight
        )

        if terminal_verbosity not in (
            'essential',
            'standard',
            'diagnostic'
        ):
            raise ValueError(
                "OptimizationModel terminal_verbosity must be one of: "
                "essential, standard, or diagnostic. "
                f"Received: {terminal_verbosity!r}."
            )

        self.terminal_verbosity = terminal_verbosity

        # The controller sets this only after QC-approved condition-level data
        # has been incorporated into the GP. Replicate-level observations must
        # never be used directly as the target-EI incumbent.
        self.incumbent_target_error_nm = None

        if self.terminal_verbosity != 'essential':
            print(
                "<<optimizer>> Auto acquisition mode: "
                f"{self.acquisition_mode}"
            )

        if self.terminal_verbosity != 'essential':
            if self.true_zero_reagents:
                print(
                    "<<optimizer>> true-zero eligible variable reagents: "
                    + ', '.join(self.true_zero_reagents)
                )
            else:
                print(
                    "<<optimizer>> true-zero search disabled for all "
                    "variable reagents"
                )

        if (
            self.terminal_verbosity != 'essential'
            and self.acquisition_mode == 'balanced'
        ):
            print(
                "<<optimizer>> balanced exploration weight: "
                f"{self.balanced_exploration_weight:.4f}"
            )

        self.gp_model = None
        # The usable-spectrum classifier is deliberately independent of the
        # primary conditional lambda-max GP.  It learns whether a physically
        # executed recipe produced an interpretable interior spectrum, while
        # the primary GP continues to learn lambda max only from exact
        # eligible observations.  It is observational in this release: no
        # acquisition score, mask, volume constraint, or stop rule reads it.
        self.usable_spectrum_model = None
        self.usable_spectrum_X = np.empty(
            (0, len(self.variable_reagents)),
            dtype=float
        )
        self.usable_spectrum_Y = np.empty((0, 1), dtype=float)
        self.acquisition = None
        self.optimizer = None
        self.prediction = None
        self.predictions = None

        # Portfolio selection is opt-in. During a portfolio batch, this holds
        # earlier immutable normalized recipes so later modes can be forced to
        # contribute a meaningfully distinct feasible condition without
        # changing the fitted GP or its incumbent.
        self._portfolio_selected_normalized_recipes = []
        self.portfolio_min_distance = 0.05
        self.acquisition_modes = [self.acquisition_mode]

    def update_usable_spectrum_model(
        self,
        normalized_recipes,
        usable_spectrum_outcomes
    ):
        '''
        Rebuilds the cumulative binary usable-spectrum probability model.

        A value of one denotes an interpretable interior spectrum; zero
        denotes an objectively classified scan-boundary-censored spectrum.
        Unknown scan-quality observations must be omitted by the controller,
        rather than guessed to be failures.

        GPy's expectation-propagation classifier does not safely support
        ``set_XY`` when its observation count grows in the deployed stack.
        Therefore this method intentionally constructs a fresh
        ``GPClassification`` from the complete cumulative history each time.
        This protects history integrity and makes the update atomic: model
        attributes change only after a successful construction.

        This companion model is not an optimizer feasibility model. Physical
        feasibility remains governed exclusively by the existing mixed-mask
        and controller-side volume pathways.

        params:
            np.ndarray normalized_recipes:
                Complete cumulative recipe history in normalized 0--1 space.

            np.ndarray usable_spectrum_outcomes:
                Complete cumulative binary outcome history, one value per
                row. Values must be exactly zero or one.

        returns:
            GPy.models.GPClassification:
                Freshly fitted binary probability model.
        '''
        X = np.asarray(normalized_recipes, dtype=float)
        Y = np.asarray(usable_spectrum_outcomes, dtype=float)

        if X.ndim == 1:
            X = X.reshape(1, -1)

        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)

        if X.ndim != 2 or X.shape[1] != self._get_dimension():
            raise ValueError(
                "Usable-spectrum model recipes must have one normalized "
                "column per variable reagent."
            )

        if Y.ndim != 2 or Y.shape[1] != 1 or Y.shape[0] != X.shape[0]:
            raise ValueError(
                "Usable-spectrum outcomes must be one binary value for each "
                "normalized recipe."
            )

        if X.shape[0] == 0:
            raise ValueError(
                "Usable-spectrum model requires at least one assessed "
                "replicate."
            )

        if (
            not np.all(np.isfinite(X))
            or np.any(X < 0.0)
            or np.any(X > 1.0)
        ):
            raise ValueError(
                "Usable-spectrum model recipes must be finite normalized "
                "values between zero and one."
            )

        if (
            not np.all(np.isfinite(Y))
            or not np.all(np.isin(Y, (0.0, 1.0)))
        ):
            raise ValueError(
                "Usable-spectrum outcomes must be finite binary zero/one "
                "values."
            )

        kernel = GPy.kern.RBF(
            input_dim=self._get_dimension(),
            variance=1.0,
            lengthscale=1.0,
            ARD=False
        )

        # Do not use the variance returned by GPClassification.predict(). In
        # the deployed GPy 1.13.2 EP implementation its variance can be NaN;
        # the predictive probability mean is the only supported output here.
        fresh_model = GPy.models.GPClassification(X.copy(), Y.copy(), kernel)

        self.usable_spectrum_model = fresh_model
        self.usable_spectrum_X = X.copy()
        self.usable_spectrum_Y = Y.copy()

        if self.terminal_verbosity == 'diagnostic':
            print(
                "<<optimizer diagnostic>> rebuilt usable-spectrum "
                f"classifier from {X.shape[0]} cumulative replicate(s)"
            )

        return self.usable_spectrum_model

    def predict_usable_spectrum_probability(self, normalized_recipes):
        '''Returns finite P(interpretable interior spectrum | recipe).

        The result is a one-dimensional probability array aligned with the
        supplied normalized recipe rows.  No classifier uncertainty is
        returned because the deployed GPy EP variance is not reliable.
        '''
        if self.usable_spectrum_model is None:
            raise ValueError(
                "Usable-spectrum probability is unavailable because the "
                "classifier has not been initialized."
            )

        X = np.asarray(normalized_recipes, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        if X.ndim != 2 or X.shape[1] != self._get_dimension():
            raise ValueError(
                "Usable-spectrum probability requires one normalized column "
                "per variable reagent."
            )

        if not np.all(np.isfinite(X)):
            raise ValueError(
                "Usable-spectrum probability requires finite normalized "
                "recipes."
            )

        probability, _ = self.usable_spectrum_model.predict(X)
        probability = np.asarray(probability, dtype=float).reshape(-1)

        if not np.all(np.isfinite(probability)):
            raise ValueError(
                "Usable-spectrum classifier returned a non-finite "
                "probability."
            )

        return np.clip(probability, 0.0, 1.0)
        
    def _minimum_pairwise_distance(self, design):
        '''
        Computes the minimum Euclidean distance between any two points in a
        candidate design.

        This score is used for maximin design selection. A larger minimum
        pairwise distance means the closest two design points are farther apart,
        which indicates better space-filling behavior.

        params:
            np.ndarray design:
                Candidate design matrix with shape:
                    n_points x n_dimensions

        returns:
            float:
                The smallest Euclidean distance between any two distinct design
                points. Returns 0.0 if fewer than two points are provided.
        '''
        design = np.asarray(design, dtype=float)

        if design.shape[0] < 2:
            return 0.0

        min_distance = np.inf

        for i in range(design.shape[0]):
            for j in range(i + 1, design.shape[0]):
                distance = np.linalg.norm(design[i] - design[j])

                if distance < min_distance:
                    min_distance = distance

        return float(min_distance)
    
    def _generate_random_lhs_design(self, n_points, n_dimensions):
        '''
        Generates one random Latin hypercube design in normalized 0-1 space.

        Each dimension is divided into n_points equal intervals. The design
        samples once from each interval in each dimension, then randomly
        permutes the interval assignments independently for each dimension.

        params:
            int n_points:
                Number of design points to generate.

            int n_dimensions:
                Number of optimized variables / reagent dimensions.

        returns:
            np.ndarray:
                Latin hypercube design with shape:
                    n_points x n_dimensions
        '''
        design = np.zeros((n_points, n_dimensions), dtype=float)

        for dim in range(n_dimensions):
            # Create one random sample inside each equal interval so each
            # dimension is evenly represented across the 0-1 range.
            interval_samples = (np.arange(n_points) + np.random.random(n_points)) / n_points

            # Randomly permute the interval samples for this dimension so the
            # dimensions are not artificially correlated.
            design[:, dim] = np.random.permutation(interval_samples)

        return design
    
    def _generate_maximin_lhs_design(self, n_points, n_dimensions, n_candidates=500):
        '''
        Generates a maximin Latin hypercube design in normalized 0-1 space.

        This creates many random Latin hypercube candidate designs and selects
        the one with the largest minimum pairwise distance. The result keeps the
        per-dimension stratification of Latin hypercube sampling while improving
        global space-filling behavior.

        params:
            int n_points:
                Number of design points to generate.

            int n_dimensions:
                Number of optimized variables / reagent dimensions.

            int n_candidates:
                Number of random Latin hypercube candidates to evaluate.
                Larger values may improve the design but take longer.

        returns:
            np.ndarray:
                Selected maximin Latin hypercube design with shape:
                    n_points x n_dimensions
        '''
        best_design = None
        best_score = -np.inf

        for _ in range(n_candidates):
            candidate_design = self._generate_random_lhs_design(n_points, n_dimensions)
            candidate_score = self._minimum_pairwise_distance(candidate_design)

            if candidate_score > best_score:
                best_score = candidate_score
                best_design = candidate_design

        if best_design is None:
            raise RuntimeError("Failed to generate a maximin Latin hypercube design.")

        return best_design
    
    def generate_initial_design(self):
        '''
        Generates the initial Auto experiment design.

        The design is generated in normalized 0-1 model space. Candidate points
        are repaired according to the true-zero rule and checked against the
        well-volume constraint before being considered for the initial design.

        Instead of rejecting an entire Latin hypercube design when one point is
        volume-infeasible, this method builds a large pool of feasible candidate
        points and selects a maximin subset from that pool. This is more robust
        when independent reagent maxima create a rectangular search space whose
        high/high corners may physically overfill the reaction well.

        returns:
            np.ndarray:
                Initial design matrix with shape:
                    initial_design_numdata x number_of_variable_reagents
        '''
        n_points = int(self.initial_design_numdata)
        n_dimensions = len(self.variable_reagents)

        pool_multiplier = 500
        target_pool_size = max(n_points * pool_multiplier, 1000)
        max_attempts = target_pool_size * 20

        feasible_points = []
        attempts = 0

        while len(feasible_points) < target_pool_size and attempts < max_attempts:
            attempts += 1

            candidate = np.random.random(n_dimensions)
            repaired_candidate = self._repair_normalized_candidate_for_true_zero(
                candidate
            )

            # If true-zero is enabled, exclude the all-off variable-reagent
            # condition from Auto seed recipes. The current workflow already
            # performs blank/background subtraction, so an all-variable-off
            # Auto recipe is usually redundant.
            all_reagents_zero_eligible = (
                len(
                    getattr(
                        self,
                        'true_zero_reagent_indices',
                        list(range(n_dimensions))
                    )
                ) == n_dimensions
            )

            if (
                self.allow_true_zero
                and all_reagents_zero_eligible
                and np.allclose(
                    repaired_candidate,
                    0.0,
                    rtol=0,
                    atol=1e-9
                )
            ):
                continue

            volume_balance = self._get_candidate_volume_balance(
                repaired_candidate
            )

            if volume_balance['volume_feasible']:
                feasible_points.append(repaired_candidate)

        if len(feasible_points) < n_points:
            raise RuntimeError(
                "Could not generate enough volume-feasible initial design "
                f"points. Needed {n_points}, found {len(feasible_points)}. "
                "Try reducing initial_data, reducing variable reagent maximums, "
                "or increasing available reaction volume."
            )

        feasible_points = np.asarray(feasible_points, dtype=float)

        # Greedy maximin selection from the feasible pool.
        # Start with the feasible point farthest from the center to encourage
        # broad coverage, then repeatedly add the point whose nearest selected
        # neighbor is as far away as possible.
        center = np.full(n_dimensions, 0.5)
        first_index = int(
            np.argmax(
                np.linalg.norm(feasible_points - center, axis=1)
            )
        )

        selected_indices = [first_index]

        while len(selected_indices) < n_points:
            selected_points = feasible_points[selected_indices]

            best_index = None
            best_distance = -np.inf

            for candidate_i, candidate in enumerate(feasible_points):
                if candidate_i in selected_indices:
                    continue

                distances_to_selected = np.linalg.norm(
                    selected_points - candidate,
                    axis=1
                )
                nearest_selected_distance = float(distances_to_selected.min())

                if nearest_selected_distance > best_distance:
                    best_distance = nearest_selected_distance
                    best_index = candidate_i

            if best_index is None:
                raise RuntimeError(
                    "Failed to select a maximin subset from feasible initial "
                    "design candidate pool."
                )

            selected_indices.append(best_index)

        initial_design = feasible_points[selected_indices]

        if getattr(self, 'terminal_verbosity', 'standard') != 'essential':
            print(
                "<<optimizer>> generated volume-feasible maximin initial "
                f"design: points={n_points}, dimensions={n_dimensions}, "
                "minimum_pairwise_distance="
                f"{self._minimum_pairwise_distance(initial_design):.6f}"
            )

        if getattr(self, 'terminal_verbosity', 'standard') == 'diagnostic':
            print(
                "<<optimizer diagnostic>> feasible seed candidates="
                f"{len(feasible_points)}, generation_attempts={attempts}"
            )

        return initial_design

    def _get_dimension(self):
        '''
        Gets the number of variable reagent dimensions in the current Auto
        optimization problem.

        returns:
            int:
                Number of variable reagents being optimized.
        '''
        return len(self.variable_reagents)
    
    def _generate_reagent_masks(
        self,
        include_all_off_mask=False,
        true_zero_reagent_indices=None
    ):
        '''
        Generates binary ON/OFF masks for the variable reagents.

        A mask defines which variable reagents are allowed to be present in a
        candidate recipe.

        For each reagent:
            0 means OFF:
                reagent is forced to true zero

            1 means ON:
                reagent is optimized continuously in its executable transfer
                range

        The all-off mask is excluded by default because the current workflow
        already performs blank/background subtraction. Including an all-off Auto
        recipe would usually be redundant with that blank correction.

        params:
            bool include_all_off_mask:
                If True, include the all-zero mask. If False, exclude it.

            list[int] or None true_zero_reagent_indices:
                Optional variable-reagent indices eligible to turn OFF. When
                omitted, every dimension remains eligible for backward
                compatibility with the established global true-zero setting.

        returns:
            list[np.ndarray]:
                List of binary masks, each with shape:
                    n_dimensions
        '''
        n_dimensions = self._get_dimension()
        if true_zero_reagent_indices is None:
            true_zero_reagent_indices = list(range(n_dimensions))
        else:
            true_zero_reagent_indices = [
                int(reagent_index)
                for reagent_index in true_zero_reagent_indices
            ]

        if any(
            reagent_index < 0 or reagent_index >= n_dimensions
            for reagent_index in true_zero_reagent_indices
        ):
            raise ValueError(
                "true_zero_reagent_indices contains an index outside the "
                "optimizer dimensionality."
            )

        if len(set(true_zero_reagent_indices)) != len(
            true_zero_reagent_indices
        ):
            raise ValueError(
                "true_zero_reagent_indices must not contain duplicates."
            )

        masks = []

        for mask_int in range(2 ** len(true_zero_reagent_indices)):
            # Reagents that are not eligible for true zero remain ON in every
            # mask. Only the selected dimensions vary between OFF and ON.
            mask = np.ones(n_dimensions, dtype=int)

            for mask_position, reagent_index in enumerate(
                true_zero_reagent_indices
            ):
                mask[reagent_index] = (
                    (mask_int >> mask_position) & 1
                )

            if not include_all_off_mask and not mask.any():
                continue

            masks.append(mask)

        return masks

    def _get_reagent_masks_for_current_settings(self):
        '''
        Gets the reagent ON/OFF masks allowed by the current optimizer settings.

        If allow_true_zero is True, variable reagents may be turned OFF by the
        mask optimizer, so all non-empty ON/OFF masks are considered.

        If allow_true_zero is False, true zero is not allowed for variable
        reagents, so only the all-ON mask is considered.

        returns:
            list[np.ndarray]:
                List of allowed binary masks.
        '''
        n_dimensions = self._get_dimension()

        if self.allow_true_zero:
            return self._generate_reagent_masks(
                include_all_off_mask=False,
                true_zero_reagent_indices=getattr(
                    self,
                    'true_zero_reagent_indices',
                    list(range(n_dimensions))
                )
            )

        return [
            np.ones(n_dimensions, dtype=int)
        ]
    
    def _get_active_mask_indices(self, mask):
        '''
        Gets the indices of reagents that are ON in a binary mask.

        params:
            np.ndarray mask:
                Binary ON/OFF mask with shape:
                    n_dimensions

        returns:
            np.ndarray:
                Integer indices where mask == 1.
        '''
        mask = np.asarray(mask, dtype=int).reshape(-1)

        if mask.shape[0] != self._get_dimension():
            raise ValueError(
                "Mask length does not match optimizer dimensionality. "
                f"Mask has {mask.shape[0]} entries, but optimizer expected "
                f"{self._get_dimension()}."
            )

        return np.where(mask == 1)[0]
    
    def _expand_masked_candidate_to_full_recipe(self, x_active, mask):
        '''
        Expands an active-dimension optimizer candidate into a full normalized
        recipe vector.

        OFF reagents are forced to true zero. ON reagents receive the values
        from x_active in mask-index order.

        Example:
            mask = [1, 0, 1]
            x_active = [0.42, 0.88]
            full recipe = [0.42, 0.0, 0.88]

        params:
            np.ndarray x_active:
                Normalized candidate values for the ON reagents only.

            np.ndarray mask:
                Binary ON/OFF mask with shape:
                    n_dimensions

        returns:
            np.ndarray:
                Full normalized recipe candidate with shape:
                    n_dimensions
        '''
        mask = np.asarray(mask, dtype=int).reshape(-1)
        active_indices = self._get_active_mask_indices(mask)

        x_active = np.asarray(x_active, dtype=float).reshape(-1)

        if x_active.shape[0] != active_indices.shape[0]:
            raise ValueError(
                "Active candidate length does not match mask. "
                f"x_active has {x_active.shape[0]} values, but mask has "
                f"{active_indices.shape[0]} active reagents."
            )

        full_x = np.zeros(self._get_dimension(), dtype=float)
        full_x[active_indices] = x_active

        return full_x
    
    def _get_masked_bounds(self, mask):
        '''
        Gets normalized optimization bounds for the ON reagents in a mask.

        OFF reagents are not included in these bounds because they are forced to
        exactly zero by the mask. ON reagents are optimized continuously, but
        their lower bound is set to the normalized concentration corresponding
        to a 5 uL transfer.

        This prevents the mixed discrete/continuous optimizer from searching
        the non-executable 0-5 uL transfer region for reagents that are ON.

        params:
            np.ndarray mask:
                Binary ON/OFF mask with shape:
                    n_dimensions

        returns:
            list[tuple]:
                Bounds for scipy minimize over active/ON dimensions only.
        '''
        if self.min_conc is None or self.max_conc is None or self.total_volume is None:
            raise ValueError(
                "Masked optimization requires min_conc, max_conc, and "
                "total_volume to be provided to OptimizationModel."
            )

        n_dimensions = self._get_dimension()
        mask = np.asarray(mask, dtype=int).reshape(n_dimensions)
        active_indices = self._get_active_mask_indices(mask)

        min_conc = np.asarray(self.min_conc, dtype=float).reshape(n_dimensions)
        max_conc = np.asarray(self.max_conc, dtype=float).reshape(n_dimensions)
        total_volume = float(self.total_volume)

        bounds = []

        for reagent_i in active_indices:
            reagent_name = self.variable_reagents[reagent_i]
            stock_conc = self._get_variable_reagent_stock_conc(reagent_name)

            if math.isclose(stock_conc, 0.0, rel_tol=0, abs_tol=1e-12):
                raise ValueError(
                    f"Cannot calculate masked bounds for {reagent_name}: "
                    "stock concentration is 0."
                )

            five_ul_conc = stock_conc * 5.0 / total_volume

            if math.isclose(
                max_conc[reagent_i],
                min_conc[reagent_i],
                rel_tol=0,
                abs_tol=1e-12
            ):
                raise ValueError(
                    f"Cannot calculate masked bounds for {reagent_name}: "
                    "min_conc and max_conc are equal."
                )

            lower_bound = (
                (five_ul_conc - min_conc[reagent_i])
                / (max_conc[reagent_i] - min_conc[reagent_i])
            )

            if lower_bound > 1.0:
                raise ValueError(
                    f"Masked lower bound for {reagent_name} is above 1.0. "
                    "The reagent cannot reach a 5 uL executable transfer "
                    "within the configured concentration range."
                )

            lower_bound = float(np.clip(lower_bound, 0.0, 1.0))

            bounds.append((lower_bound, 1.0))

        return bounds
    
    def _validate_predictive_standard_deviation_nm(
        self,
        predicted_lambda_std_nm,
        acquisition_mode
    ):
        '''
        Validates GP predictive standard deviation for acquisition scoring.

        GPModel.predict() supplies predictive standard deviation rather than
        variance. The caller converts that value directly to nanometers before
        this validation; no additional square root belongs here.

        params:
            float predicted_lambda_std_nm:
                GP predictive standard deviation in nanometers.

            str acquisition_mode:
                Canonical mode requesting uncertainty. Used to produce a clear
                mode-specific validation error.

        returns:
            float:
                Finite, nonnegative predictive standard deviation in nm.
        '''
        if predicted_lambda_std_nm is None:
            raise ValueError(
                f"{acquisition_mode} acquisition requires GP predictive "
                "standard deviation in nanometers."
            )

        predicted_lambda_std_nm = float(
            predicted_lambda_std_nm
        )

        if (
            not math.isfinite(predicted_lambda_std_nm)
            or predicted_lambda_std_nm < 0.0
        ):
            raise ValueError(
                f"{acquisition_mode} acquisition requires a finite, "
                "nonnegative GP predictive standard deviation in "
                "nanometers. Received: "
                f"{predicted_lambda_std_nm!r}."
            )

        return predicted_lambda_std_nm

    def set_incumbent_target_error_nm(self, incumbent_target_error_nm):
        '''
        Stores the QC-approved condition-level incumbent used by target EI.

        The controller owns the scientific decision about which conditions are
        eligible for model training. This optimizer setter only validates and
        stores the resulting best absolute target error.

        params:
            float incumbent_target_error_nm:
                Best absolute condition-level target error in nanometers.
                Values must be finite and nonnegative.

        returns:
            None
        '''
        try:
            incumbent_target_error_nm = float(
                incumbent_target_error_nm
            )
        except (TypeError, ValueError):
            raise ValueError(
                "incumbent_target_error_nm must be a finite, nonnegative "
                "condition-level target error in nanometers. Received: "
                f"{incumbent_target_error_nm!r}."
            )

        if (
            not math.isfinite(incumbent_target_error_nm)
            or incumbent_target_error_nm < 0.0
        ):
            raise ValueError(
                "incumbent_target_error_nm must be a finite, nonnegative "
                "condition-level target error in nanometers. Received: "
                f"{incumbent_target_error_nm!r}."
            )

        self.incumbent_target_error_nm = incumbent_target_error_nm

    def _standard_normal_pdf(self, value):
        '''Returns the standard-normal probability density at value.'''
        value = float(value)

        return float(
            math.exp(-0.5 * value ** 2)
            / math.sqrt(2.0 * math.pi)
        )

    def _standard_normal_cdf(self, value):
        '''Returns the standard-normal cumulative probability at value.'''
        value = float(value)

        return float(
            0.5 * (
                1.0
                + math.erf(value / math.sqrt(2.0))
            )
        )

    def _calculate_target_error_expected_improvement_nm(
        self,
        predicted_lambda_mean_nm,
        predicted_lambda_std_nm,
        incumbent_target_error_nm
    ):
        '''
        Calculates expected improvement in absolute target error.

        Let the GP posterior response be:

            Y ~ Normal(mu, sigma)

        with target t and incumbent absolute target error d_best. Target-error
        improvement is:

            max(0, d_best - abs(Y - t))

        This method evaluates its exact expectation by integrating the normal
        density over the improvement interval:

            t - d_best <= Y <= t + d_best

        The interval is split at the target because absolute error is linear on
        each side. The implementation uses the standard-normal CDF and PDF and
        introduces no numerical quadrature or additional dependency.

        When sigma is effectively zero, the normal posterior is deterministic
        and the exact limiting value is used directly:

            max(0, d_best - abs(mu - t))

        params:
            float predicted_lambda_mean_nm:
                GP-predicted lambda-max mean in nanometers.

            float predicted_lambda_std_nm:
                GP predictive standard deviation in nanometers.

            float incumbent_target_error_nm:
                Best QC-approved condition-level absolute target error in nm.

        returns:
            float:
                Finite expected target-error improvement in nanometers, bounded
                between zero and incumbent_target_error_nm.
        '''
        try:
            predicted_lambda_mean_nm = float(
                predicted_lambda_mean_nm
            )
            target_value_nm = float(self.target_value)
            incumbent_target_error_nm = float(
                incumbent_target_error_nm
            )
        except (TypeError, ValueError):
            raise ValueError(
                "Target-EI requires finite numeric mean, target, standard "
                "deviation, and incumbent values in nanometers."
            )

        if (
            not math.isfinite(predicted_lambda_mean_nm)
            or not math.isfinite(target_value_nm)
        ):
            raise ValueError(
                "Target-EI requires finite GP-predicted mean and target "
                "values in nanometers."
            )

        predicted_lambda_std_nm = (
            self._validate_predictive_standard_deviation_nm(
                predicted_lambda_std_nm,
                acquisition_mode='Target-EI'
            )
        )

        if (
            not math.isfinite(incumbent_target_error_nm)
            or incumbent_target_error_nm < 0.0
        ):
            raise ValueError(
                "Target-EI incumbent target error must be finite and "
                "nonnegative in nanometers. Received: "
                f"{incumbent_target_error_nm!r}."
            )

        centered_mean_nm = (
            predicted_lambda_mean_nm
            - target_value_nm
        )

        # Deterministic posterior limit. The tolerance is far below any
        # physically meaningful wavelength resolution and prevents unstable
        # division by a numerically zero standard deviation.
        if predicted_lambda_std_nm <= 1e-12:
            return float(
                max(
                    0.0,
                    incumbent_target_error_nm
                    - abs(centered_mean_nm)
                )
            )

        if incumbent_target_error_nm == 0.0:
            return 0.0

        lower_z = (
            -incumbent_target_error_nm
            - centered_mean_nm
        ) / predicted_lambda_std_nm
        target_z = -centered_mean_nm / predicted_lambda_std_nm
        upper_z = (
            incumbent_target_error_nm
            - centered_mean_nm
        ) / predicted_lambda_std_nm

        lower_probability = (
            self._standard_normal_cdf(target_z)
            - self._standard_normal_cdf(lower_z)
        )
        upper_probability = (
            self._standard_normal_cdf(upper_z)
            - self._standard_normal_cdf(target_z)
        )

        lower_improvement = (
            (incumbent_target_error_nm + centered_mean_nm)
            * lower_probability
            + predicted_lambda_std_nm
            * (
                self._standard_normal_pdf(lower_z)
                - self._standard_normal_pdf(target_z)
            )
        )
        upper_improvement = (
            (incumbent_target_error_nm - centered_mean_nm)
            * upper_probability
            + predicted_lambda_std_nm
            * (
                self._standard_normal_pdf(upper_z)
                - self._standard_normal_pdf(target_z)
            )
        )

        expected_improvement_nm = (
            lower_improvement
            + upper_improvement
        )

        # Cancellation in extreme normal tails can produce tiny values just
        # outside the mathematical [0, d_best] interval. Clamp only to those
        # exact theoretical bounds.
        expected_improvement_nm = min(
            incumbent_target_error_nm,
            max(0.0, expected_improvement_nm)
        )

        return float(expected_improvement_nm)

    def _calculate_acquisition_score(
        self,
        predicted_lambda_mean_nm,
        predicted_lambda_std_nm=None,
        incumbent_target_error_nm=None
    ):
        '''
        Calculates the statistical acquisition score for one GP prediction.

        The surrounding optimizer minimizes this score. Statistical scoring is
        intentionally kept separate from reagent masks and physical-feasibility
        penalties so every acquisition mode must continue through the same
        executable-recipe pathway.

        Exploit preserves the legacy squared target-distance formula. Explore
        minimizes negative GP predictive standard deviation. Balanced uses a
        target-aware straddle score in nanometers:

            absolute target error
            - balanced_exploration_weight * predictive standard deviation

        Target EI minimizes the negative expected reduction in the best
        QC-approved condition-level absolute target error achieved so far.

        params:
            float predicted_lambda_mean_nm:
                GP-predicted lambda-max mean in nanometers.

            float predicted_lambda_std_nm:
                GP predictive standard deviation in nanometers. Required by
                explore and balanced; exploit does not use this value.

            float incumbent_target_error_nm:
                Best QC-approved condition-level target error in nanometers.
                Required by target EI and unused by the other modes.

        returns:
            float:
                Acquisition score to minimize.
        '''
        if self.acquisition_mode not in self._SUPPORTED_ACQUISITION_MODES:
            raise ValueError(
                "OptimizationModel acquisition_mode must be one of: "
                "exploit, explore, balanced, or target_ei. "
                f"Received: {self.acquisition_mode!r}."
            )

        if self.acquisition_mode == 'exploit':
            target_error = (
                float(predicted_lambda_mean_nm)
                - float(self.target_value)
            )

            return float(target_error ** 2)

        if self.acquisition_mode == 'explore':
            predicted_lambda_std_nm = (
                self._validate_predictive_standard_deviation_nm(
                    predicted_lambda_std_nm,
                    acquisition_mode='Explore'
                )
            )

            # The surrounding optimizer minimizes. Negating standard
            # deviation therefore selects maximum predictive uncertainty.
            return float(-1.0 * predicted_lambda_std_nm)

        if self.acquisition_mode == 'balanced':
            try:
                predicted_lambda_mean_nm = float(
                    predicted_lambda_mean_nm
                )
            except (TypeError, ValueError):
                raise ValueError(
                    "Balanced acquisition requires a finite GP-predicted "
                    "lambda-max mean in nanometers. Received: "
                    f"{predicted_lambda_mean_nm!r}."
                )

            if not math.isfinite(predicted_lambda_mean_nm):
                raise ValueError(
                    "Balanced acquisition requires a finite GP-predicted "
                    "lambda-max mean in nanometers. Received: "
                    f"{predicted_lambda_mean_nm!r}."
                )

            try:
                target_value_nm = float(self.target_value)
            except (TypeError, ValueError):
                raise ValueError(
                    "Balanced acquisition requires a finite target lambda "
                    "max in nanometers. Received: "
                    f"{self.target_value!r}."
                )

            if not math.isfinite(target_value_nm):
                raise ValueError(
                    "Balanced acquisition requires a finite target lambda "
                    "max in nanometers. Received: "
                    f"{target_value_nm!r}."
                )

            predicted_lambda_std_nm = (
                self._validate_predictive_standard_deviation_nm(
                    predicted_lambda_std_nm,
                    acquisition_mode='Balanced'
                )
            )

            predicted_target_error_nm = abs(
                predicted_lambda_mean_nm
                - target_value_nm
            )

            # Both terms are expressed in nanometers. The dimensionless weight
            # therefore has an interpretable one-for-one default scale.
            return float(
                predicted_target_error_nm
                - self.balanced_exploration_weight
                * predicted_lambda_std_nm
            )

        if self.acquisition_mode == 'target_ei':
            if incumbent_target_error_nm is None:
                incumbent_target_error_nm = getattr(
                    self,
                    'incumbent_target_error_nm',
                    None
                )

            if incumbent_target_error_nm is None:
                raise ValueError(
                    "Target-EI requires a QC-approved condition-level "
                    "incumbent target error before recipe selection."
                )

            expected_improvement_nm = (
                self._calculate_target_error_expected_improvement_nm(
                    predicted_lambda_mean_nm=predicted_lambda_mean_nm,
                    predicted_lambda_std_nm=predicted_lambda_std_nm,
                    incumbent_target_error_nm=incumbent_target_error_nm
                )
            )

            # The surrounding optimizer minimizes, so negate expected
            # improvement to select the candidate with the largest value.
            return float(-1.0 * expected_improvement_nm)

        raise NotImplementedError(
            "Acquisition score for mode "
            f"{self.acquisition_mode!r} is not implemented yet."
        )

    def _masked_acquisition_objective(self, x_active, mask):
        '''
        Computes the acquisition objective for one masked candidate.

        The mask controls which reagents are OFF and which reagents are ON:
            OFF reagents are forced to exactly zero.
            ON reagents are optimized continuously within executable bounds.

        The active candidate is expanded into a full normalized recipe, checked
        for volume feasibility, then evaluated with the Gaussian process model.

        params:
            np.ndarray x_active:
                Normalized candidate values for the ON reagents only.

            np.ndarray mask:
                Binary ON/OFF mask with shape:
                    n_dimensions

        returns:
            float:
                Statistical acquisition score for a feasible candidate, or a
                large finite penalty for a volume-infeasible candidate.
        '''
        full_x = self._expand_masked_candidate_to_full_recipe(
            x_active,
            mask
        )

        volume_balance = self._get_candidate_volume_balance(full_x)

        if not volume_balance['volume_feasible']:
            overflow_volume = max(
                0.0,
                -1.0 * volume_balance['water_volume']
            )

            bad_water_penalty = 0.0

            if (
                volume_balance['volume_does_not_overflow']
                and not volume_balance['water_transfer_executable']
            ):
                bad_water_penalty = 1.0

            # Large finite penalty. The overflow term distinguishes overflow
            # severity, and the bad_water_penalty distinguishes non-executable
            # 0-5 uL water top-off cases from otherwise valid candidates.
            return float(1e12 + overflow_volume ** 2 + bad_water_penalty)

        # Legacy single-mode callers and isolated tests do not need to carry
        # portfolio state. Avoid invoking the portfolio helper unless a
        # portfolio has already selected at least one prior recipe.
        if getattr(self, '_portfolio_selected_normalized_recipes', []):
            portfolio_distance = self._get_portfolio_nearest_distance(full_x)
        else:
            portfolio_distance = None

        if (
            portfolio_distance is not None
            and (
                portfolio_distance <= 1e-12
                or portfolio_distance < self.portfolio_min_distance
            )
        ):
            # This candidate is physically executable but would duplicate an
            # earlier member of the same unmeasured portfolio batch. Use a
            # penalty larger than every normal acquisition score so SLSQP
            # seeks its best distinct alternative. Physical infeasibility
            # remains independently represented by the existing penalty.
            return float(
                1e13
                + self.portfolio_min_distance
                - portfolio_distance
            )

        predicted_lambda_std = None

        if self.acquisition_mode == 'exploit':
            # Preserve the exact stable exploit prediction pathway. Explore
            # and later uncertainty-aware modes require the full distribution.
            predicted_lambda_max = self._predict_lambda_max_nm(full_x)

        else:
            (
                predicted_lambda_max,
                predicted_lambda_std
            ) = self.predict_lambda_distribution_nm(full_x)

        return self._calculate_acquisition_score(
            predicted_lambda_mean_nm=predicted_lambda_max,
            predicted_lambda_std_nm=predicted_lambda_std,
            incumbent_target_error_nm=getattr(
                self,
                'incumbent_target_error_nm',
                None
            )
        )

    def _get_portfolio_nearest_distance(self, normalized_recipe):
        '''
        Returns the nearest prior portfolio recipe in normalized RMS distance.

        RMS distance makes the 0--1 normalized concentration scale comparable
        across two- and three-variable runs, while exact-zero mask choices
        remain represented by literal zero coordinates.
        '''
        selected_recipes = getattr(
            self,
            '_portfolio_selected_normalized_recipes',
            []
        )

        if len(selected_recipes) == 0:
            return None

        normalized_recipe = np.asarray(
            normalized_recipe,
            dtype=float
        ).reshape(-1)

        distances = []
        for selected_recipe in selected_recipes:
            selected_recipe = np.asarray(
                selected_recipe,
                dtype=float
            ).reshape(-1)

            if selected_recipe.shape != normalized_recipe.shape:
                raise ValueError(
                    "Portfolio diversity comparison received incompatible "
                    "normalized recipe shapes."
                )

            distances.append(
                float(
                    np.sqrt(
                        np.mean(
                            (normalized_recipe - selected_recipe) ** 2
                        )
                    )
                )
            )

        return min(distances)

    def _masked_target_distance_objective(self, x_active, mask):
        '''
        Backward-compatible wrapper for the masked acquisition objective.

        The active mixed-mask optimizer now uses
        _masked_acquisition_objective() directly. This wrapper preserves the
        prior internal method name for isolated callers and stable regression
        comparisons during the staged acquisition-mode implementation.
        '''
        return self._masked_acquisition_objective(x_active, mask)
    
    def _generate_feasible_masked_starting_points(self, mask, n_restarts):
        '''
        Generates feasible starting points for optimization within one mask.

        The returned points only contain values for ON reagents. OFF reagents
        are handled by the mask and are forced to exactly zero when the active
        candidate is expanded into full recipe space.

        Starting points are sampled within the executable ON-reagent bounds,
        expanded to full recipe space, and checked against the full volume
        feasibility rules:
            - no overflow
            - water top-off is 0 uL or >= 5 uL

        params:
            np.ndarray mask:
                Binary ON/OFF mask with shape:
                    n_dimensions

            int n_restarts:
                Desired number of feasible starting points.

        returns:
            list[np.ndarray]:
                Feasible active-dimension starting points.
        '''
        bounds = self._get_masked_bounds(mask)
        active_indices = self._get_active_mask_indices(mask)

        if len(bounds) != len(active_indices):
            raise ValueError(
                "Masked bounds length does not match number of active reagents."
            )

        feasible_points = []
        max_attempts = max(100, n_restarts * 100)

        attempts = 0

        while len(feasible_points) < n_restarts and attempts < max_attempts:
            attempts += 1

            x_active = np.array(
                [
                    np.random.uniform(low, high)
                    for low, high in bounds
                ],
                dtype=float
            )

            full_x = self._expand_masked_candidate_to_full_recipe(
                x_active,
                mask
            )

            volume_balance = self._get_candidate_volume_balance(full_x)

            portfolio_distance = (
                self._get_portfolio_nearest_distance(full_x)
                if getattr(
                    self,
                    '_portfolio_selected_normalized_recipes',
                    []
                )
                else None
            )

            if (
                volume_balance['volume_feasible']
                and (
                    portfolio_distance is None
                    or (
                        portfolio_distance > 1e-12
                        and portfolio_distance >= self.portfolio_min_distance
                    )
                )
            ):
                feasible_points.append(x_active)

        if len(feasible_points) < n_restarts:
            print(
                f"<<optimizer>> warning: generated only "
                f"{len(feasible_points)} feasible starts for mask "
                f"{mask.tolist()} out of {n_restarts} requested"
            )

        return feasible_points
    
    def _optimize_single_mask(self, mask, n_restarts=25):
        '''
        Optimizes the acquisition objective within one ON/OFF reagent mask.

        OFF reagents are fixed at exactly zero by the mask. ON reagents are
        optimized continuously within executable transfer bounds.

        params:
            np.ndarray mask:
                Binary ON/OFF mask with shape:
                    n_dimensions

            int n_restarts:
                Number of feasible starting points to try for this mask.

        returns:
            dict:
                Optimization result information for this mask.
        '''
        mask = np.asarray(mask, dtype=int).reshape(self._get_dimension())
        bounds = self._get_masked_bounds(mask)
        active_indices = self._get_active_mask_indices(mask)

        if len(active_indices) == 0:
            raise ValueError(
                "Cannot optimize an all-OFF mask with no active reagents."
            )

        starting_points = self._generate_feasible_masked_starting_points(
            mask,
            n_restarts
        )

        # If no feasible random starts were found, fall back to the midpoint of
        # the active bounds. The objective penalty will still reject it if it is
        # physically infeasible.
        if not starting_points:
            midpoint = np.array(
                [
                    (low + high) / 2.0
                    for low, high in bounds
                ],
                dtype=float
            )
            starting_points = [midpoint]

        best_x_active = None
        best_full_x = None
        best_objective = np.inf
        best_result_success = False
        best_result_message = None
        best_result_method = None
        best_result_status = None

        for x0 in starting_points:
            result = minimize(
                fun=lambda x_active: self._masked_acquisition_objective(
                    x_active,
                    mask
                ),
                x0=x0,
                bounds=bounds,
                method='SLSQP'
            )

            # SLSQP can return values infinitesimally outside a declared bound.
            # Clip first, then evaluate and rank the exact candidate that will
            # be returned and reported. Trusting result.fun from the unclipped
            # point would let cross-mask ordering disagree with selection-time
            # metadata and, in an edge case, with physical feasibility.
            candidate_x_active = np.asarray(
                result.x,
                dtype=float
            ).copy()

            for i, (low, high) in enumerate(bounds):
                candidate_x_active[i] = np.clip(
                    candidate_x_active[i],
                    low,
                    high
                )

            candidate_objective = self._masked_acquisition_objective(
                candidate_x_active,
                mask
            )

            if (
                np.isfinite(candidate_objective)
                and candidate_objective < best_objective
            ):
                best_objective = float(candidate_objective)
                best_x_active = candidate_x_active
                best_full_x = self._expand_masked_candidate_to_full_recipe(
                    best_x_active,
                    mask
                )
                best_result_success = bool(result.success)
                best_result_message = result.message
                best_result_method = 'SLSQP'
                best_result_status = getattr(result, 'status', None)

        # A feasible optimum can lie exactly on an executable bound. SLSQP
        # occasionally reports a failed status at such points even though the
        # clipped candidate has a finite, deterministically re-evaluated
        # objective. Retry that candidate once with a bound-native method so a
        # spurious SLSQP status does not remain the sole optimizer verdict.
        if best_x_active is not None and not best_result_success:
            recovery_result = minimize(
                fun=lambda x_active: self._masked_acquisition_objective(
                    x_active,
                    mask
                ),
                x0=best_x_active,
                bounds=bounds,
                method='L-BFGS-B'
            )

            recovery_x_active = np.asarray(
                recovery_result.x,
                dtype=float
            ).copy()

            for i, (low, high) in enumerate(bounds):
                recovery_x_active[i] = np.clip(
                    recovery_x_active[i],
                    low,
                    high
                )

            recovery_objective = self._masked_acquisition_objective(
                recovery_x_active,
                mask
            )
            recovery_full_x = self._expand_masked_candidate_to_full_recipe(
                recovery_x_active,
                mask
            )
            recovery_volume_balance = self._get_candidate_volume_balance(
                recovery_full_x
            )

            recovery_is_equivalent_or_better = (
                recovery_objective < best_objective
                or np.isclose(
                    recovery_objective,
                    best_objective,
                    rtol=1e-9,
                    atol=1e-9
                )
            )

            if (
                bool(recovery_result.success)
                and np.isfinite(recovery_objective)
                and recovery_volume_balance['volume_feasible']
                and recovery_is_equivalent_or_better
            ):
                best_objective = float(recovery_objective)
                best_x_active = recovery_x_active
                best_full_x = recovery_full_x
                best_result_success = True
                best_result_message = (
                    "Recovered after failed SLSQP status: "
                    f"{recovery_result.message}"
                )
                best_result_method = 'L-BFGS-B recovery'
                best_result_status = getattr(
                    recovery_result,
                    'status',
                    None
                )

        if best_full_x is None:
            return {
                'mask': mask,
                'is_selected': False,
                'success': False,
                'message': 'No finite optimizer result found for mask.',
                'optimizer_method': None,
                'optimizer_status': None,
                'objective': np.inf,
                'acquisition_mode': self.acquisition_mode,
                'acquisition_score': None,
                'balanced_exploration_weight': (
                    self.balanced_exploration_weight
                    if self.acquisition_mode == 'balanced'
                    else None
                ),
                'incumbent_target_error_nm': getattr(
                    self,
                    'incumbent_target_error_nm',
                    None
                ),
                'x_active': None,
                'x_full': None,
                'normalized_recipe': None,
                'physical_concentrations': None,
                'volume_balance': None,
                'predicted_lambda_max': None,
                'predicted_lambda_mean_nm': None,
                'predicted_lambda_std_nm': None,
                'predicted_target_error_nm': None
            }

        volume_balance = self._get_candidate_volume_balance(best_full_x)

        predicted_lambda_max = None
        predicted_lambda_std_nm = None
        predicted_target_error_nm = None
        acquisition_score = None

        min_conc = np.asarray(self.min_conc, dtype=float).reshape(-1)
        max_conc = np.asarray(self.max_conc, dtype=float).reshape(-1)
        physical_values = (
            np.asarray(best_full_x, dtype=float)
            * (max_conc - min_conc)
            + min_conc
        )
        physical_concentrations = {
            str(reagent_name): float(physical_values[reagent_i])
            for reagent_i, reagent_name
            in enumerate(self.variable_reagents)
        }

        if volume_balance['volume_feasible']:
            (
                predicted_lambda_max,
                predicted_lambda_std_nm
            ) = self.predict_lambda_distribution_nm(best_full_x)
            predicted_target_error_nm = float(
                abs(predicted_lambda_max - float(self.target_value))
            )
            acquisition_score = self._calculate_acquisition_score(
                predicted_lambda_mean_nm=predicted_lambda_max,
                predicted_lambda_std_nm=predicted_lambda_std_nm,
                incumbent_target_error_nm=getattr(
                    self,
                    'incumbent_target_error_nm',
                    None
                )
            )

        return {
            'mask': mask,
            'is_selected': False,
            'success': best_result_success,
            'message': best_result_message,
            'optimizer_method': best_result_method,
            'optimizer_status': best_result_status,
            'objective': best_objective,
            'acquisition_mode': self.acquisition_mode,
            'acquisition_score': acquisition_score,
            'balanced_exploration_weight': (
                self.balanced_exploration_weight
                if self.acquisition_mode == 'balanced'
                else None
            ),
            'incumbent_target_error_nm': getattr(
                self,
                'incumbent_target_error_nm',
                None
            ),
            'x_active': best_x_active,
            'x_full': best_full_x,
            'normalized_recipe': np.asarray(
                best_full_x,
                dtype=float
            ).copy(),
            'physical_concentrations': physical_concentrations,
            'volume_balance': volume_balance,
            'predicted_lambda_max': predicted_lambda_max,
            'predicted_lambda_mean_nm': predicted_lambda_max,
            'predicted_lambda_std_nm': predicted_lambda_std_nm,
            'predicted_target_error_nm': predicted_target_error_nm
        }
    
    def _optimize_acquisition_with_masks(self, n_restarts_per_mask=25):
        '''
        Finds the best normalized recipe for the configured acquisition mode
        using mixed discrete/continuous mask optimization.

        Exploit minimizes squared distance from the requested target. Explore
        minimizes negative GP predictive standard deviation. Balanced trades
        absolute target error against weighted predictive uncertainty, with
        both statistical quantities expressed in nanometers. Target EI
        minimizes negative expected improvement in the best QC-approved
        condition-level absolute target error.

        Discrete part:
            Each binary mask decides which variable reagents are OFF or ON.

        Continuous part:
            For each mask, ON reagents are optimized continuously within their
            executable transfer bounds. OFF reagents are forced to exactly zero.

        The best feasible candidate across all allowed masks is returned.

        params:
            int n_restarts_per_mask:
                Number of feasible starting points to try for each mask.

        returns:
            np.ndarray:
                Best full normalized recipe point found, with shape:
                    n_dimensions
        '''
        masks = self._get_reagent_masks_for_current_settings()

        mask_results = []

        for mask in masks:
            result = self._optimize_single_mask(
                mask,
                n_restarts=n_restarts_per_mask
            )
            mask_results.append(result)

        feasible_results = [
            result for result in mask_results
            if (
                result['x_full'] is not None
                and np.isfinite(result['objective'])
                and result.get('volume_balance') is not None
                and result['volume_balance'].get(
                    'volume_feasible',
                    False
                )
            )
        ]

        if not feasible_results:
            raise RuntimeError(
                "Mixed mask optimization failed: no finite, physically "
                "feasible optimizer result was found for any allowed reagent "
                "mask."
            )

        best_result = min(
            feasible_results,
            key=lambda result: result['objective']
        )

        for result in mask_results:
            result['is_selected'] = result is best_result

        if (
            best_result['volume_balance'] is None
            or not best_result['volume_balance']['volume_feasible']
        ):
            raise RuntimeError(
                "Mixed mask optimization failed: best finite result was not "
                "volume-feasible. The controller would reject this recipe."
            )

        if not best_result['success']:
            print(
                "<<optimizer>> warning: selected best finite mask result "
                f"despite scipy status for mask "
                f"{best_result['mask'].tolist()}: {best_result['message']}"
            )

        best_x = np.asarray(best_result['x_full'], dtype=float)

        # Store detailed debug information for getNextReaction() and terminal
        # reporting.
        self.last_mask_results = mask_results
        self.last_selected_mask = best_result['mask']
        self.last_raw_optimizer_candidate = best_x
        self.last_repaired_optimizer_candidate = best_x
        self.last_optimizer_objective = float(best_result['objective'])
        self.last_optimizer_predicted_lambda_max = best_result['predicted_lambda_max']
        self.last_optimizer_volume_balance = best_result['volume_balance']
        # Preserve the SciPy outcome for the selected feasible mask as
        # selection-time provenance.  This is intentionally separate from the
        # acquisition score: a finite feasible candidate can be selected after
        # a non-success SciPy status, while a successful recovery is recorded
        # as ``L-BFGS-B recovery`` for later audit.
        self.last_optimizer_method = best_result.get('optimizer_method')
        self.last_optimizer_success = best_result.get('success')
        self.last_optimizer_status = best_result.get('optimizer_status')
        self.last_optimizer_message = best_result.get('message')

        if getattr(self, 'terminal_verbosity', 'standard') == 'diagnostic':
            for result in mask_results:
                result_mask = result.get('mask')
                result_mask_for_audit = (
                    result_mask.tolist()
                    if hasattr(result_mask, 'tolist')
                    else result_mask
                )
                result_volume_balance = result.get('volume_balance')
                print(
                    "<<optimizer diagnostic>> mask acquisition audit: "
                    f"mask={result_mask_for_audit}, "
                    f"selected={result.get('is_selected', False)}, "
                    f"optimizer_method={result.get('optimizer_method')}, "
                    f"optimizer_success={result.get('success')}, "
                    f"optimizer_status={result.get('optimizer_status')}, "
                    f"optimizer_message={result.get('message')}, "
                    f"mode={result.get('acquisition_mode', getattr(self, 'acquisition_mode', None))}, "
                    f"objective={result.get('objective')}, "
                    f"score={result.get('acquisition_score')}, "
                    f"predicted_mean_nm="
                    f"{result.get('predicted_lambda_mean_nm')}, "
                    f"predicted_std_nm="
                    f"{result.get('predicted_lambda_std_nm')}, "
                    f"volume_feasible="
                    f"{bool(result_volume_balance and result_volume_balance.get('volume_feasible', False))}"
                )

        return best_x

    def _optimize_target_distance_with_masks(self, n_restarts_per_mask=25):
        '''
        Backward-compatible wrapper for mixed-mask acquisition optimization.

        Exploit remains target-distance minimization, so existing isolated
        callers using the prior internal method name retain equivalent behavior.
        '''
        return self._optimize_acquisition_with_masks(
            n_restarts_per_mask=n_restarts_per_mask
        )
    
    def _get_variable_reagent_stock_conc(self, reagent_name):
        '''
        Gets the stock concentration currently available on the deck for a
        variable reagent.

        Reagent containers are indexed by names like:
            silver_nitrateC0.375
            potassium_bromideC0.01

        This helper matches the base reagent name before the concentration
        marker and returns the deck concentration from ``reagent_info``.
        Multiple same-name source containers are supported when they contain
        the same stock concentration. The robot represents those containers as
        a MultiContainer and switches between them during aspiration when the
        current tube becomes insufficient. Optimizer concentration-to-volume
        calculations therefore use their shared stock concentration.

        Multiple containers with different stock concentrations are rejected
        deliberately. One Auto variable represents one concentration
        dimension, so silently choosing one concentration would make transfer
        volume, feasibility, and model-coordinate calculations inconsistent.

        params:
            str reagent_name:
                Base reagent name, such as 'silver_nitrate'.

        returns:
            float:
                Stock concentration of the reagent on the deck.
        '''
        matching_sources = []

        # Iterate over rows rather than looking each name up with ``.loc``.
        # ``.loc[name, 'conc']`` returns a Series when same-name backup tubes
        # create a duplicate reagent_info index. That is a supported
        # robot-side MultiContainer configuration, not an invalid worksheet.
        for reagent_container_name, reagent_row in self.reagent_info.iterrows():
            reagent_container_name = str(reagent_container_name)

            if 'C' in reagent_container_name:
                base_name = reagent_container_name.split('C')[0]
            else:
                base_name = reagent_container_name

            if base_name == reagent_name:
                try:
                    stock_conc = float(reagent_row['conc'])
                except (TypeError, ValueError):
                    raise ValueError(
                        "Could not parse stock concentration for variable "
                        f"reagent {reagent_name!r} from source container "
                        f"{reagent_container_name!r}."
                    )

                if not math.isfinite(stock_conc) or stock_conc <= 0.0:
                    raise ValueError(
                        "Stock concentration for variable reagent "
                        f"{reagent_name!r} in source container "
                        f"{reagent_container_name!r} must be finite and "
                        f"greater than zero. Received {stock_conc!r}."
                    )

                matching_sources.append((
                    reagent_container_name,
                    reagent_row.get('deck_pos', None),
                    reagent_row.get('loc', None),
                    stock_conc
                ))

        if len(matching_sources) == 0:
            raise ValueError(
                f"Could not find stock concentration for variable reagent "
                f"{reagent_name} in reagent_info."
            )

        reference_conc = matching_sources[0][3]
        inconsistent_sources = [
            source
            for source in matching_sources[1:]
            if not math.isclose(
                source[3],
                reference_conc,
                rel_tol=1e-9,
                abs_tol=1e-12
            )
        ]

        if inconsistent_sources:
            source_details = '; '.join(
                f"{name} at deck {deck_pos}, {loc}: {conc:g}"
                for name, deck_pos, loc, conc in matching_sources
            )
            raise ValueError(
                "Variable reagent "
                f"{reagent_name!r} has source containers with different "
                "stock concentrations. A single Auto variable reagent must "
                "use one stock concentration so concentration-to-volume and "
                f"feasibility calculations remain valid. Sources: {source_details}."
            )

        return reference_conc
    
    def _apply_true_zero_transfer_rule_to_volume(self, volume):
        '''
        Applies the Auto true-zero transfer rule to one transfer volume.

        Rule:
            0 uL stays 0
            0 < volume < 2.5 uL maps to 0 uL
            2.5 <= volume < 5.0 uL maps to 5.0 uL
            volume >= 5.0 uL is unchanged

        A small tolerance is used at the 0, 2.5, and 5.0 uL boundaries so
        floating-point artifacts do not send mathematically equivalent values
        to the wrong side of a threshold.

        params:
            float volume:
                Transfer volume in uL.

        returns:
            float:
                Repaired transfer volume in uL.
        '''
        volume = float(volume)
        boundary_tol = 1e-9

        if math.isclose(volume, 0.0, rel_tol=0, abs_tol=boundary_tol):
            return 0.0

        # Values clearly below the midpoint round down to true zero.
        if volume < 2.5 - boundary_tol:
            return 0.0

        # Values from the midpoint up to just below the minimum transfer round
        # up to 5 uL. Values effectively equal to 5 uL are left unchanged below.
        if volume < 5.0 - boundary_tol:
            return 5.0

        return volume
    
    def _repair_normalized_candidate_for_true_zero(self, x):
        '''
        Repairs one normalized optimizer candidate according to the Auto
        true-zero transfer rule.

        The optimizer works in normalized 0-1 model space, but the robot
        executes transfer volumes. This helper converts a normalized candidate
        into target concentrations, converts those concentrations into transfer
        volumes, applies the true-zero transfer rule, and then converts the
        repaired values back into normalized model space.

        This allows the optimizer to evaluate candidates as the robot/controller
        would actually execute them, instead of treating the forbidden 0-5 uL
        transfer region as a meaningful continuous search space.

        params:
            np.ndarray x:
                One normalized recipe candidate with shape:
                    n_dimensions
                or:
                    1 x n_dimensions

        returns:
            np.ndarray:
                Repaired normalized recipe candidate with shape:
                    n_dimensions
        '''
        if self.min_conc is None or self.max_conc is None or self.total_volume is None:
            raise ValueError(
                "True-zero-aware optimization requires min_conc, max_conc, "
                "and total_volume to be provided to OptimizationModel."
            )

        n_dimensions = self._get_dimension()
        x = np.asarray(x, dtype=float).reshape(n_dimensions)

        min_conc = np.asarray(self.min_conc, dtype=float).reshape(n_dimensions)
        max_conc = np.asarray(self.max_conc, dtype=float).reshape(n_dimensions)
        total_volume = float(self.total_volume)

        repaired_x = np.array(x, dtype=float, copy=True)
        true_zero_reagent_indices = set(
            getattr(
                self,
                'true_zero_reagent_indices',
                list(range(n_dimensions))
                if getattr(self, 'allow_true_zero', False)
                else []
            )
        )

        for reagent_i, reagent_name in enumerate(self.variable_reagents):
            # Required-ON reagents already have a 5 uL-equivalent lower bound.
            # Do not apply a repair that could convert one of them to zero.
            if reagent_i not in true_zero_reagent_indices:
                continue

            stock_conc = self._get_variable_reagent_stock_conc(reagent_name)

            if math.isclose(stock_conc, 0.0, rel_tol=0, abs_tol=1e-12):
                raise ValueError(
                    f"Cannot repair true-zero candidate for {reagent_name}: "
                    "stock concentration is 0."
                )

            # Convert normalized model coordinate to target concentration.
            target_conc = (
                x[reagent_i] * (max_conc[reagent_i] - min_conc[reagent_i])
                + min_conc[reagent_i]
            )

            # Convert target concentration to the transfer volume that would be
            # required to make that concentration in the final reaction volume.
            transfer_volume = target_conc * total_volume / stock_conc

            repaired_volume = self._apply_true_zero_transfer_rule_to_volume(
                transfer_volume
            )

            # Convert repaired transfer volume back to target concentration.
            repaired_conc = repaired_volume * stock_conc / total_volume

            # Convert repaired concentration back to normalized model space.
            # Clip defensively to keep numerical artifacts inside the GP domain.
            repaired_x[reagent_i] = (
                (repaired_conc - min_conc[reagent_i])
                / (max_conc[reagent_i] - min_conc[reagent_i])
            )

        return np.clip(repaired_x, 0.0, 1.0)
    
    def _get_variable_transfer_volumes_for_normalized_candidate(
        self,
        x,
        apply_true_zero_repair=True
    ):
        '''
        Converts one normalized optimizer candidate into variable reagent
        transfer volumes.

        The optimizer works in normalized 0-1 model space. This helper converts
        that candidate into target concentrations and then into the physical
        transfer volumes required to make those concentrations in the final
        reaction volume.

        By default the candidate is repaired with the true-zero rule before
        transfer volumes are calculated, so optimization receives executable
        robot behavior. Read-only diagnostic plots may disable that repair to
        reveal the physically non-executable interval between exact zero and
        the 5 uL minimum transfer.

        params:
            np.ndarray x:
                One normalized recipe candidate with shape:
                    n_dimensions
                or:
                    1 x n_dimensions

        returns:
            dict:
                Variable reagent names as keys and repaired transfer volumes
                in uL as values.
        '''
        if self.min_conc is None or self.max_conc is None or self.total_volume is None:
            raise ValueError(
                "Volume-aware optimization requires min_conc, max_conc, "
                "and total_volume to be provided to OptimizationModel."
            )

        n_dimensions = self._get_dimension()
        x = np.asarray(x, dtype=float).reshape(n_dimensions)

        if apply_true_zero_repair:
            repaired_x = self._repair_normalized_candidate_for_true_zero(x)
        else:
            repaired_x = np.array(x, dtype=float, copy=True)

        min_conc = np.asarray(self.min_conc, dtype=float).reshape(n_dimensions)
        max_conc = np.asarray(self.max_conc, dtype=float).reshape(n_dimensions)
        total_volume = float(self.total_volume)

        variable_transfer_volumes = {}

        for reagent_i, reagent_name in enumerate(self.variable_reagents):
            stock_conc = self._get_variable_reagent_stock_conc(reagent_name)

            if math.isclose(stock_conc, 0.0, rel_tol=0, abs_tol=1e-12):
                raise ValueError(
                    f"Cannot calculate transfer volume for {reagent_name}: "
                    "stock concentration is 0."
                )

            repaired_conc = (
                repaired_x[reagent_i] * (max_conc[reagent_i] - min_conc[reagent_i])
                + min_conc[reagent_i]
            )

            transfer_volume = repaired_conc * total_volume / stock_conc
            variable_transfer_volumes[reagent_name] = float(transfer_volume)

        return variable_transfer_volumes

    def _get_candidate_volume_balance(
        self,
        x,
        apply_true_zero_repair=True
    ):
        '''
        Calculates the well-volume balance for one normalized optimizer
        candidate.

        This helper is dimension-general and mirrors the controller-side
        volume-balance logic. It does not rescale variable reagents as ratios.
        Variable reagent volumes are treated independently, and water fills the
        remaining space.

        A candidate is volume-feasible only if:
            1. each variable transfer is exactly 0 uL or at least 5 uL
            2. fixed + variable volumes do not exceed the final reaction volume
            3. water top-off is either exactly 0 uL or at least 5 uL

        This prevents the optimizer from selecting candidates that would require
        non-executable 0-5 uL water transfers or would run underfilled.

        params:
            np.ndarray x:
                One normalized recipe candidate with shape:
                    n_dimensions
                or:
                    1 x n_dimensions

        returns:
            dict:
                Volume-balance information for the candidate.
        '''
        if self.total_volume is None or self.fixed_reagent_volumes is None:
            raise ValueError(
                "Volume-aware optimization requires total_volume and "
                "fixed_reagent_volumes to be provided to OptimizationModel."
            )

        total_volume = float(self.total_volume)

        fixed_volume_total = float(
            sum(float(volume) for volume in self.fixed_reagent_volumes.values())
        )

        variable_transfer_volumes = (
            self._get_variable_transfer_volumes_for_normalized_candidate(
                x,
                apply_true_zero_repair=apply_true_zero_repair
            )
        )
        variable_volume_total = float(sum(variable_transfer_volumes.values()))
        variable_transfer_executable_by_reagent = {
            reagent_name: bool(
                math.isclose(
                    transfer_volume,
                    0.0,
                    rel_tol=0,
                    abs_tol=1e-9
                )
                or transfer_volume >= 5.0 - 1e-9
            )
            for reagent_name, transfer_volume
            in variable_transfer_volumes.items()
        }
        variable_transfers_executable = all(
            variable_transfer_executable_by_reagent.values()
        )

        volume_before_water = fixed_volume_total + variable_volume_total
        water_volume = total_volume - volume_before_water

        volume_tol = 1e-9

        # Treat tiny floating-point artifacts around zero as exactly zero water.
        if math.isclose(water_volume, 0.0, rel_tol=0, abs_tol=volume_tol):
            water_volume = 0.0

        volume_does_not_overflow = water_volume >= -volume_tol

        # Water top-off must be executable. If water is needed, it must be at
        # least 5 uL. Otherwise, the recipe would require a non-executable
        # 0-5 uL water transfer or would run underfilled.
        water_transfer_executable = (
            math.isclose(water_volume, 0.0, rel_tol=0, abs_tol=volume_tol)
            or water_volume >= 5.0 - volume_tol
        )

        volume_feasible = (
            variable_transfers_executable
            and volume_does_not_overflow
            and water_transfer_executable
        )

        return {
            'total_volume': total_volume,
            'fixed_transfer_volumes': {
                str(reagent_name): float(volume)
                for reagent_name, volume
                in self.fixed_reagent_volumes.items()
            },
            'fixed_volume_total': fixed_volume_total,
            'variable_transfer_volumes': variable_transfer_volumes,
            'variable_transfer_executable_by_reagent': (
                variable_transfer_executable_by_reagent
            ),
            'variable_transfers_executable': bool(
                variable_transfers_executable
            ),
            'variable_volume_total': variable_volume_total,
            'volume_before_water': volume_before_water,
            'water_volume': float(water_volume),
            'volume_does_not_overflow': bool(volume_does_not_overflow),
            'water_transfer_executable': bool(water_transfer_executable),
            'volume_feasible': bool(volume_feasible)
        }

    def get_candidate_volume_balance_for_plotting(self, x):
        '''
        Returns read-only physical-feasibility data for one plotting point.

        Higher-dimensional GP visualizations need to display the executable
        recipe region used by acquisition optimization, including the
        non-executable interval strictly between 0 and 5 uL. This intentionally
        small public wrapper prevents controller plotting code from
        reimplementing or drifting from the authoritative volume rules. Unlike
        optimization evaluation, it deliberately does not repair near-zero
        values to exact zero before assessing the displayed point. It does not
        alter the candidate, fitted GP, optimizer history, or controller state.

        params:
            np.ndarray x:
                One normalized recipe with one entry per variable reagent.

        returns:
            dict:
                The volume-balance dictionary returned by the internal,
                dimension-general feasibility evaluator.
        '''
        return self._get_candidate_volume_balance(
            x,
            apply_true_zero_repair=False
        )

    def get_candidate_feasibility_for_plotting(self, x):
        '''
        Returns a read-only, mask-aware feasibility classification for one
        plotted Auto candidate.

        A physical volume balance alone cannot express the complete Auto
        search policy: an exact 0 uL transfer is physically executable, but
        it is selectable only when that reagent is configured in
        ``true_zero_reagents``.  Likewise, the all-off recipe is physically
        possible but intentionally excluded from mixed-mask optimization.

        This diagnostic helper preserves the raw ``0 < transfer < 5 uL``
        interval for feasibility-overlay plots while adding those discrete
        mask rules.  It is observational only and never repairs a candidate,
        modifies GP history, or changes recipe selection.

        params:
            np.ndarray x:
                One normalized recipe with one entry per variable reagent.

        returns:
            dict:
                The physical volume-balance fields plus mask-aware flags used
                by diagnostic plots. ``mask_feasible`` is the same executable
                domain used by Auto selection for the displayed recipe.
        '''
        balance = self.get_candidate_volume_balance_for_plotting(x)

        if hasattr(self, 'true_zero_reagents'):
            true_zero_reagents = {
                str(reagent_name)
                for reagent_name in self.true_zero_reagents
            }
        elif getattr(self, 'allow_true_zero', False):
            # Preserve the documented legacy all-variable interpretation for
            # lightweight test doubles and older restored model objects.
            true_zero_reagents = {
                str(reagent_name)
                for reagent_name in self.variable_reagents
            }
        else:
            true_zero_reagents = set()

        zero_transfer_by_reagent = {}
        zero_transfer_permitted_by_reagent = {}
        zero_transfer_not_permitted_by_reagent = {}

        for reagent_name, transfer_volume in balance[
            'variable_transfer_volumes'
        ].items():
            is_exact_zero = math.isclose(
                float(transfer_volume),
                0.0,
                rel_tol=0,
                abs_tol=1e-9
            )
            zero_is_permitted = (
                str(reagent_name) in true_zero_reagents
            )
            zero_transfer_by_reagent[str(reagent_name)] = is_exact_zero
            zero_transfer_permitted_by_reagent[str(reagent_name)] = (
                is_exact_zero and zero_is_permitted
            )
            zero_transfer_not_permitted_by_reagent[str(reagent_name)] = (
                is_exact_zero and not zero_is_permitted
            )

        all_variable_transfers_zero = all(
            zero_transfer_by_reagent.values()
        )
        all_off_mask_excluded = bool(all_variable_transfers_zero)
        disallowed_zero_transfer = any(
            zero_transfer_not_permitted_by_reagent.values()
        )

        balance.update({
            'true_zero_eligible_reagents': sorted(true_zero_reagents),
            'zero_transfer_by_reagent': zero_transfer_by_reagent,
            'zero_transfer_permitted_by_reagent': (
                zero_transfer_permitted_by_reagent
            ),
            'zero_transfer_not_permitted_by_reagent': (
                zero_transfer_not_permitted_by_reagent
            ),
            'all_variable_transfers_zero': bool(all_variable_transfers_zero),
            'all_off_mask_excluded': all_off_mask_excluded,
            'mask_feasible': bool(
                balance['volume_feasible']
                and not disallowed_zero_transfer
                and not all_off_mask_excluded
            )
        })

        return balance
    
    def _generate_feasible_starting_points(self, n_restarts):
        '''
        Generates feasible normalized starting points for optimizer restarts.

        Random starts are repaired with the true-zero rule and checked against
        the well-volume constraint. Feasible starts are preferred so the
        optimizer spends less time searching over physically impossible
        overfilled recipes.

        If the feasible region is small and not enough feasible points are
        found, this helper returns the feasible points it did find. The caller
        can still add fallback points if needed.

        params:
            int n_restarts:
                Desired number of feasible random starting points.

        returns:
            list[np.ndarray]:
                Feasible normalized starting points.
        '''
        n_dimensions = self._get_dimension()
        feasible_points = []
        max_attempts = max(100, n_restarts * 50)

        attempts = 0

        while len(feasible_points) < n_restarts and attempts < max_attempts:
            attempts += 1

            candidate = np.random.random(n_dimensions)
            repaired_candidate = self._repair_normalized_candidate_for_true_zero(
                candidate
            )
            volume_balance = self._get_candidate_volume_balance(
                repaired_candidate
            )

            if volume_balance['volume_feasible']:
                feasible_points.append(repaired_candidate)

        if len(feasible_points) < n_restarts:
            print(
                f"<<optimizer>> warning: generated only "
                f"{len(feasible_points)} feasible optimizer starts out of "
                f"{n_restarts} requested"
            )

        return feasible_points
    
    def _predict_lambda_max_nm(self, x):
        '''
        Predicts lambda max in nanometers for one normalized recipe point.

        The Gaussian process model is trained on normalized Y values where:
            0 corresponds to 300 nm
            1 corresponds to 900 nm

        This helper converts the model prediction back into nanometers so the
        optimizer objective can compare predictions directly to target_value.

        params:
            np.ndarray x:
                One normalized recipe point with shape:
                    n_dimensions
                or:
                    1 x n_dimensions

        returns:
            float:
                Predicted lambda max in nanometers.
        '''
        x = np.asarray(x, dtype=float).reshape(1, self._get_dimension())

        normalized_prediction = self.gp_model.predict(x)[0]
        normalized_prediction = float(normalized_prediction.flatten()[0])

        return normalized_prediction * 600.0 + 300.0

    def predict_lambda_distribution_nm(self, x):
        '''
        Predicts lambda max mean and GP predictive standard deviation in
        nanometers for one normalized recipe point.

        The Gaussian process model is trained on normalized Y values where:
            0 corresponds to 300 nm
            1 corresponds to 900 nm

        This helper converts both the model-predicted mean and predictive
        standard deviation back into nanometers. It is intended for Auto
        reporting and should be called at recipe-selection time, before the
        selected recipe is experimentally run and before the result is added
        back into the training data.

        params:
            np.ndarray x:
                One normalized recipe point with shape:
                    n_dimensions
                or:
                    1 x n_dimensions

        returns:
            tuple(float, float):
                Predicted lambda max mean in nanometers and GP predictive
                standard deviation in nanometers.
        '''
        if self.gp_model is None:
            return None, None

        x = np.asarray(x, dtype=float).reshape(1, self._get_dimension())

        normalized_mean, normalized_std = self.gp_model.predict(x)

        normalized_mean = float(normalized_mean.flatten()[0])
        normalized_std = float(normalized_std.flatten()[0])

        if not math.isfinite(normalized_mean):
            raise ValueError(
                "GP prediction returned a non-finite normalized mean: "
                f"{normalized_mean!r}."
            )

        if not math.isfinite(normalized_std):
            raise ValueError(
                "GP prediction returned a non-finite normalized predictive "
                f"standard deviation: {normalized_std!r}."
            )

        # Predictive standard deviation is mathematically nonnegative. Permit
        # only negligible floating-point roundoff; a material negative value
        # indicates a broken model/wrapper contract and must fail closed rather
        # than silently turning an invalid uncertainty into zero.
        negative_std_roundoff_tolerance = 1e-12
        if normalized_std < -negative_std_roundoff_tolerance:
            raise ValueError(
                "GP prediction returned a materially negative normalized "
                "predictive standard deviation: "
                f"{normalized_std!r}."
            )

        normalized_std = max(normalized_std, 0.0)

        predicted_lambda_mean_nm = normalized_mean * 600.0 + 300.0
        predicted_lambda_std_nm = normalized_std * 600.0

        return predicted_lambda_mean_nm, predicted_lambda_std_nm

    def predict_lambda_distribution_nm_batch(
        self,
        x_values,
        chunk_size=4096
    ):
        '''
        Predicts GP mean and standard deviation in nm for normalized recipes.

        This read-only batch counterpart to predict_lambda_distribution_nm()
        is used by visualization only.  It keeps the established 300--900 nm
        output conversion and validates the GP standard-deviation contract,
        while avoiding per-pixel GP calls for conditional slice atlases.

        params:
            np.ndarray x_values:
                Two-dimensional array of normalized recipes.  Rows are
                candidates and columns follow variable_reagents order.

            int chunk_size:
                Maximum candidates submitted to one GP prediction call.  A
                bounded default keeps three-variable plots responsive without
                changing model state.

        returns:
            tuple(np.ndarray, np.ndarray):
                Mean lambda max and predictive standard deviation in nm, one
                value per supplied candidate.
        '''
        if self.gp_model is None:
            raise ValueError(
                "Cannot predict a GP distribution batch before the GP model "
                "has been initialized."
            )

        try:
            chunk_size = int(chunk_size)
        except (TypeError, ValueError, OverflowError):
            raise ValueError(
                "GP prediction batch chunk_size must be a positive integer."
            )

        if chunk_size < 1:
            raise ValueError(
                "GP prediction batch chunk_size must be at least 1."
            )

        n_dimensions = self._get_dimension()
        x_values = np.asarray(x_values, dtype=float)

        if x_values.ndim == 1:
            x_values = x_values.reshape(1, -1)

        if (
            x_values.ndim != 2
            or x_values.shape[1] != n_dimensions
            or x_values.shape[0] == 0
        ):
            raise ValueError(
                "GP prediction batch requires a nonempty N x D normalized "
                f"recipe array with D={n_dimensions}."
            )

        if not np.all(np.isfinite(x_values)):
            raise ValueError(
                "GP prediction batch received non-finite normalized recipes."
            )

        normalized_means = []
        normalized_stds = []
        negative_std_roundoff_tolerance = 1e-12

        for start_index in range(0, x_values.shape[0], chunk_size):
            x_chunk = x_values[start_index:start_index + chunk_size]
            chunk_mean, chunk_std = self.gp_model.predict(x_chunk)

            chunk_mean = np.asarray(chunk_mean, dtype=float).reshape(-1)
            chunk_std = np.asarray(chunk_std, dtype=float).reshape(-1)

            if (
                chunk_mean.shape[0] != x_chunk.shape[0]
                or chunk_std.shape[0] != x_chunk.shape[0]
            ):
                raise ValueError(
                    "GP prediction batch returned an unexpected output "
                    "shape."
                )

            if (
                not np.all(np.isfinite(chunk_mean))
                or not np.all(np.isfinite(chunk_std))
            ):
                raise ValueError(
                    "GP prediction batch returned non-finite mean or "
                    "predictive standard-deviation values."
                )

            if np.any(chunk_std < -negative_std_roundoff_tolerance):
                raise ValueError(
                    "GP prediction batch returned a materially negative "
                    "predictive standard deviation."
                )

            normalized_means.append(chunk_mean)
            normalized_stds.append(np.maximum(chunk_std, 0.0))

        normalized_means = np.concatenate(normalized_means)
        normalized_stds = np.concatenate(normalized_stds)

        return (
            normalized_means * 600.0 + 300.0,
            normalized_stds * 600.0
        )
    
    def _target_distance_objective(self, x):
        '''
        Computes the target-distance objective for one normalized recipe point.

        The raw optimizer candidate is first repaired according to the Auto
        true-zero transfer rule. The repaired candidate is then checked for
        volume feasibility before being evaluated with the Gaussian process
        model.

        This prevents the optimizer from treating either the forbidden 0-5 uL
        transfer region or overfilled well-volume recipes as meaningful
        executable search space.

        The objective is the squared distance between the model-predicted lambda
        max and the target lambda max. Lower values are better. Overfilled
        recipes receive a large finite penalty so they are not selected.

        params:
            np.ndarray x:
                One normalized recipe point with shape:
                    n_dimensions
                or:
                    1 x n_dimensions

        returns:
            float:
                Squared error between predicted lambda max and target_value, or
                a large penalty for volume-infeasible candidates.
        '''
        repaired_x = self._repair_normalized_candidate_for_true_zero(x)
        volume_balance = self._get_candidate_volume_balance(repaired_x)

        if not volume_balance['volume_feasible']:
            overflow_volume = -1.0 * volume_balance['water_volume']

            # Use a large finite penalty instead of inf so scipy can keep
            # searching without running into nan/inf optimizer behavior.
            return float(1e12 + overflow_volume ** 2)

        predicted_lambda_max = self._predict_lambda_max_nm(repaired_x)

        return self._calculate_acquisition_score(
            predicted_lambda_mean_nm=predicted_lambda_max
        )
    
    def _optimize_target_distance(self, n_restarts=50):
        '''
        Finds the normalized recipe point predicted to be closest to the target
        lambda max.

        This replaces the old brute-force 2D grid search with a continuous,
        dimension-general optimization. Multiple random restarts are used to
        reduce the chance of getting stuck in a poor local optimum.

        The objective evaluates true-zero-repaired candidates, and this method
        returns the repaired candidate so the controller receives the same
        physically executable recipe that was scored by the optimizer.

        Because true-zero repair creates flat or discontinuous regions around
        the 0-5 uL transfer boundary, SciPy may occasionally return a finite
        useful result even when result.success is False. In that case, the best
        finite result is accepted with a warning instead of failing the run.

        Debug attributes are stored so getNextReaction() can report whether the
        optimizer's raw candidate was changed by true-zero repair.

        params:
            int n_restarts:
                Number of random starting points to try.

        returns:
            np.ndarray:
                Best repaired normalized recipe point found, with shape:
                    n_dimensions
        '''
        n_dimensions = self._get_dimension()
        bounds = [(0.0, 1.0)] * n_dimensions

        best_x = None
        best_objective = np.inf
        best_result_success = False
        best_result_message = None

        # Include the center point as a deterministic restart so every run has
        # at least one stable starting location. If it is volume-infeasible, the
        # objective penalty will handle it.
        starting_points = [np.full(n_dimensions, 0.5)]

        # Prefer feasible random restarts so the optimizer spends less time
        # searching over physically impossible overfilled recipes.
        starting_points.extend(
            self._generate_feasible_starting_points(n_restarts)
        )

        # If the feasible-region sampler could not find enough starts, add
        # ordinary random starts as a fallback. The objective function still
        # penalizes any overfilled candidates.
        while len(starting_points) < n_restarts + 1:
            starting_points.append(np.random.random(n_dimensions))
        
        for x0 in starting_points:
            result = minimize(
                fun=self._target_distance_objective,
                x0=x0,
                bounds=bounds,
                method='L-BFGS-B'
            )

            # Accept the best finite result, even if SciPy reports non-success.
            # This is intentional because the true-zero repair rule can make the
            # objective less smooth near transfer-volume thresholds.
            if np.isfinite(result.fun) and result.fun < best_objective:
                best_objective = float(result.fun)
                best_x = np.clip(result.x, 0.0, 1.0)
                best_result_success = bool(result.success)
                best_result_message = result.message

        if best_x is None:
            raise RuntimeError(
                "Target-distance optimization failed from all restart points."
            )

        if not best_result_success:
            print(
                "<<optimizer>> warning: using best finite optimizer result "
                f"despite scipy status: {best_result_message}"
            )

        repaired_best_x = self._repair_normalized_candidate_for_true_zero(best_x)

        # Store both values for transparent debugging/reporting.
        self.last_raw_optimizer_candidate = best_x
        self.last_repaired_optimizer_candidate = repaired_best_x
        self.last_optimizer_objective = best_objective

        return repaired_best_x
    
    def _update_prediction_grid_for_plotting(self, grid_size=100):
        '''
        Updates the two-dimensional GP prediction grids used by the controller
        heatmap plots.

        The first variable reagent is the horizontal x-axis and the second
        variable reagent is the vertical y-axis. The saved arrays therefore use
        the standard Matplotlib heatmap convention:

            columns:
                first variable reagent / x-axis

            rows:
                second variable reagent / y-axis

        NumPy meshgrid with indexing='xy' already produces points in this
        orientation when its flattened predictions are reshaped in C order.
        The arrays must not be transposed after reshaping.

        For experiments with any dimensionality other than two, both plotting
        grids are cleared so the controller can skip 2D heatmaps cleanly.

        params:
            int grid_size:
                Number of model-evaluation points along each reagent axis.

        returns:
            None
        '''
        try:
            grid_size = int(
                grid_size
            )

        except (TypeError, ValueError, OverflowError):
            raise ValueError(
                "Prediction-grid size must be an integer of at least 2. "
                f"Received: {grid_size!r}."
            )

        if grid_size < 2:
            raise ValueError(
                "Prediction-grid size must be at least 2. "
                f"Received: {grid_size}."
            )

        if self._get_dimension() != 2:
            self.predictions = None
            self.prediction_uncertainty = None
            return

        if self.gp_model is None:
            self.predictions = None
            self.prediction_uncertainty = None

            raise RuntimeError(
                "Cannot generate GP prediction grids before the GP model has "
                "been initialized."
            )

        reagent_0_axis = np.linspace(
            0.0,
            1.0,
            grid_size
        )

        reagent_1_axis = np.linspace(
            0.0,
            1.0,
            grid_size
        )

        reagent_0_grid, reagent_1_grid = np.meshgrid(
            reagent_0_axis,
            reagent_1_axis,
            indexing='xy'
        )

        grid_points = np.column_stack(
            (
                reagent_0_grid.ravel(order='C'),
                reagent_1_grid.ravel(order='C')
            )
        )

        (
            normalized_predictions,
            normalized_prediction_stds
        ) = self.gp_model.predict(
            grid_points
        )

        normalized_predictions = np.asarray(
            normalized_predictions,
            dtype=float
        ).reshape(-1)

        normalized_prediction_stds = np.asarray(
            normalized_prediction_stds,
            dtype=float
        ).reshape(-1)

        expected_point_count = grid_size * grid_size

        if normalized_predictions.size != expected_point_count:
            raise ValueError(
                "GP prediction output size does not match the requested "
                f"{grid_size} x {grid_size} plotting grid. Expected "
                f"{expected_point_count} values, received "
                f"{normalized_predictions.size}."
            )

        if normalized_prediction_stds.size != expected_point_count:
            raise ValueError(
                "GP uncertainty output size does not match the requested "
                f"{grid_size} x {grid_size} plotting grid. Expected "
                f"{expected_point_count} values, received "
                f"{normalized_prediction_stds.size}."
            )

        if not np.all(np.isfinite(normalized_predictions)):
            raise ValueError(
                "GP prediction grid contains non-finite values."
            )

        if not np.all(np.isfinite(normalized_prediction_stds)):
            raise ValueError(
                "GP uncertainty grid contains non-finite values."
            )

        heatmap_shape = (
            reagent_1_axis.size,
            reagent_0_axis.size
        )

        # Rows correspond to reagent 1 (the plotted y-axis), and columns
        # correspond to reagent 0 (the plotted x-axis). Do not transpose.
        self.predictions = (
            normalized_predictions
            .reshape(
                heatmap_shape,
                order='C'
            )
            * 600.0
            + 300.0
        )

        self.prediction_uncertainty = (
            normalized_prediction_stds
            .reshape(
                heatmap_shape,
                order='C'
            )
            * 600.0
        )
    
    def refresh_prediction_grid_for_plotting(self, grid_size=100):
        '''
        Rebuilds the controller-facing GP prediction and uncertainty grids from
        the optimizer's current fitted model state.

        This public lifecycle method is called by the controller after a
        completed experimental batch has been incorporated into the GP model.
        It ensures that heatmaps labeled "After Batch N" are based on a model
        that actually includes Batch N.

        For exactly two variable reagents, this refreshes:

            self.predictions:
                Predicted lambda-max values in nm. Array columns correspond to
                the first variable reagent / x-axis, and rows correspond to the
                second variable reagent / y-axis.

            self.prediction_uncertainty:
                GP predictive standard deviations in nm, using the same axis
                orientation as self.predictions.

        For any dimensionality other than two, the internal grid helper clears
        both plotting arrays to None so the controller can skip 2D heatmaps.

        This method does not refit the GP, select a recipe, alter experimental
        data, or change optimizer iteration state.

        params:
            int grid_size:
                Number of GP evaluation points along each reagent axis.

        returns:
            tuple:
                (
                    self.predictions,
                    self.prediction_uncertainty
                )
        '''
        if self.gp_model is None:
            self.predictions = None
            self.prediction_uncertainty = None

            if getattr(self, 'terminal_verbosity', 'standard') != 'essential':
                print(
                    "<<optimizer>> skipping prediction-grid refresh because "
                    "the GP model has not been initialized"
                )

            return (
                self.predictions,
                self.prediction_uncertainty
            )

        self._update_prediction_grid_for_plotting(
            grid_size=grid_size
        )

        if (
            self._get_dimension() == 2
            and getattr(self, 'terminal_verbosity', 'standard') != 'essential'
        ):
            print(
                "<<optimizer>> refreshed 2D GP prediction and uncertainty "
                "grids from the current fitted model using a "
                f"{int(grid_size)} x {int(grid_size)} grid"
            )

        return (
            self.predictions,
            self.prediction_uncertainty
        )
    
    def initialize_optimizer(self, X_init, Y_init):
        '''
        Initializes the Gaussian Process model and other components for
        Bayesian Optimization with initial experimental data.

        params:
            np.ndarray X_init:
                Initial recipe points in normalized model space. Shape:
                    n_observations x n_variable_reagents

            np.ndarray Y_init:
                Initial normalized objective values corresponding to X_init.
                Shape:
                    n_observations x 1
        '''
        def f(x):
            return abs(sum(x) - (self.target_value * 3))

        input_dim = self._get_dimension()

        kernel = GPy.kern.sde_Matern32(
            input_dim=input_dim,
            variance=1.0,
            lengthscale=1,
            ARD=False,
            active_dims=None,
            name='Mat32'
        )

        self.gp_model = GPyOpt.models.GPModel(
            kernel,
            noise_var=1e-4,
            optimize_restarts=0,
            verbose=False
        )

        self.gp_model.updateModel(X_init, Y_init, None, None)

        self.acq_optimizer = GPyOpt.optimization.acquisition_optimizer.AcquisitionOptimizer(
            self.space,
            optimizer='lbfgs'
        )

        self.acquisition = GPyOpt.acquisitions.AcquisitionEI(
            self.gp_model,
            self.space,
            self.acq_optimizer
        )

        self.evaluator = GPyOpt.core.evaluators.Sequential(self.acquisition)
        objective = GPyOpt.core.task.objective.SingleObjective(f)

        self.optimizer = GPyOpt.methods.ModularBayesianOptimization(
            self.gp_model,
            self.space,
            objective,
            self.acquisition,
            self.evaluator,
            X_init,
            Y_init
        )
        
    def getNextReaction(self):
        '''
        Suggests the next normalized recipe point for experimentation.

        This method uses mixed discrete/continuous mask optimization.

        Discrete part:
            A binary reagent mask decides which variable reagents are OFF or ON.
            OFF reagents are forced to exact true zero.

        Continuous part:
            ON reagents are optimized continuously within executable transfer
            bounds, so the optimizer does not search the non-executable 0-5 uL
            transfer region for active reagents.

        The selected recipe's predicted lambda-max mean and GP predictive
        standard deviation, target error, acquisition score, acquisition mode,
        incumbent, and reagent mask are recorded before the recipe is
        experimentally run. These values form an immutable selection-time
        audit record that the controller can compare with the subsequently
        measured result.

        Prediction and uncertainty heatmap grids are intentionally not generated
        here. The controller refreshes those grids only after a completed batch
        has been incorporated into the fitted GP model, ensuring that a plot
        labeled "After Batch N" actually includes Batch N.

        returns:
            list[np.ndarray]:
                A single suggested normalized recipe point wrapped in a list.
                This preserves the controller-facing return format:

                    [array([...])]
        '''
        if (
            self.acquisition_mode
            not in self.IMPLEMENTED_ACQUISITION_MODES
        ):
            raise NotImplementedError(
                "Acquisition mode "
                f"{self.acquisition_mode!r} is configured, but its recipe "
                "selection behavior is not implemented yet. Only 'exploit', "
                "'explore', 'balanced', and 'target_ei' may select recipes at "
                "the current implementation stage."
            )

        best_x = self._optimize_acquisition_with_masks()

        self.last_optimizer_selected_normalized_recipe = np.array(
            best_x,
            dtype=float,
            copy=True
        )
        self.last_optimizer_balanced_exploration_weight = (
            getattr(self, 'balanced_exploration_weight', 1.0)
            if self.acquisition_mode == 'balanced'
            else None
        )

        self.last_optimizer_incumbent_target_error_nm = getattr(
            self,
            'incumbent_target_error_nm',
            None
        )

        self.last_optimizer_acquisition_mode = self.acquisition_mode

        (
            predicted_lambda_max,
            predicted_lambda_std
        ) = self.predict_lambda_distribution_nm(
            best_x
        )

        self.last_optimizer_predicted_lambda_mean_nm = (
            predicted_lambda_max
        )

        self.last_optimizer_predicted_lambda_std_nm = (
            predicted_lambda_std
        )

        self.last_optimizer_predicted_target_error_nm = float(
            abs(
                predicted_lambda_max
                - float(self.target_value)
            )
        )

        # Re-evaluate the statistical score for the final returned candidate.
        # This avoids treating an optimizer-internal objective value as audit
        # metadata if a bounded candidate was clipped by a negligible amount.
        # Every acquisition function is expressed as a minimization score, so
        # a lower recorded value always means the candidate was preferred.
        self.last_optimizer_acquisition_score = (
            self._calculate_acquisition_score(
                predicted_lambda_mean_nm=predicted_lambda_max,
                predicted_lambda_std_nm=predicted_lambda_std,
                incumbent_target_error_nm=(
                    self.last_optimizer_incumbent_target_error_nm
                )
            )
        )

        selected_mask = getattr(
            self,
            'last_selected_mask',
            None
        )

        if (
            getattr(self, 'terminal_verbosity', 'standard') != 'essential'
            and selected_mask is not None
        ):
            print(
                f"<<optimizer>> selected reagent mask "
                f"{selected_mask.tolist()} for suggested recipe"
            )

        selected_mask_for_audit = (
            selected_mask.tolist()
            if selected_mask is not None
            else None
        )
        incumbent_for_audit = (
            f"{self.last_optimizer_incumbent_target_error_nm:.4f} nm"
            if self.last_optimizer_incumbent_target_error_nm is not None
            else 'not used'
        )
        balanced_weight_for_audit = (
            f"{self.last_optimizer_balanced_exploration_weight:.4f}"
            if self.last_optimizer_balanced_exploration_weight is not None
            else 'not used'
        )

        if getattr(self, 'terminal_verbosity', 'standard') != 'essential':
            print(
                "<<optimizer>> acquisition audit: "
                f"mode={self.last_optimizer_acquisition_mode}, "
                f"score={self.last_optimizer_acquisition_score:.6f}, "
                f"predicted_target_error="
                f"{self.last_optimizer_predicted_target_error_nm:.4f} nm, "
                f"predicted_lambda_mean={predicted_lambda_max:.4f} nm, "
                f"predicted_lambda_std={predicted_lambda_std:.4f} nm, "
                f"incumbent_target_error={incumbent_for_audit}, "
                f"balanced_exploration_weight="
                f"{balanced_weight_for_audit}, "
                f"selected_mask={selected_mask_for_audit}"
            )

        volume_balance = getattr(
            self,
            'last_optimizer_volume_balance',
            None
        )

        if (
            getattr(self, 'terminal_verbosity', 'standard') != 'essential'
            and volume_balance is not None
        ):
            print(
                "<<optimizer>> suggested recipe volume balance: "
                f"fixed={volume_balance['fixed_volume_total']:.4f} uL, "
                f"variable={volume_balance['variable_volume_total']:.4f} uL, "
                f"water={volume_balance['water_volume']:.4f} uL, "
                f"feasible={volume_balance['volume_feasible']}"
            )

        if getattr(self, 'terminal_verbosity', 'standard') == 'diagnostic':
            print(
                f"<<optimizer diagnostic>> suggested normalized recipe "
                f"{best_x} with predicted lambda max "
                f"{predicted_lambda_max:.4f} nm and GP predictive std "
                f"{predicted_lambda_std:.4f} nm"
            )

        return [
            best_x
        ]

    def getNextPortfolio(self, acquisition_modes, portfolio_min_distance):
        '''
        Selects one distinct recipe for each ordered acquisition mode.

        Every mode sees the same already-fitted GP and the same target-EI
        incumbent. No model update occurs within this method. The written
        portfolio order matters only when independent mode optima collide:
        earlier modes retain their candidate and later modes re-optimize
        outside the normalized-RMS diversity radius.

        params:
            list[str] acquisition_modes:
                Ordered canonical modes, one unique condition per member.

            float portfolio_min_distance:
                Minimum normalized RMS distance between distinct portfolio
                conditions. Zero disables near-duplicate exclusion but exact
                duplicate detection still remains fail-closed.

        returns:
            list[dict]:
                Immutable selection records in requested mode order.
        '''
        acquisition_modes = list(acquisition_modes)

        if len(acquisition_modes) == 0:
            raise ValueError(
                "Acquisition portfolio must contain at least one mode."
            )

        if len(acquisition_modes) != len(set(acquisition_modes)):
            raise ValueError(
                "Acquisition portfolio must not repeat a canonical mode."
            )

        unsupported_modes = [
            mode for mode in acquisition_modes
            if mode not in self.IMPLEMENTED_ACQUISITION_MODES
        ]

        if unsupported_modes:
            raise NotImplementedError(
                "Acquisition portfolio contains unsupported mode(s): "
                + ', '.join(unsupported_modes)
            )

        try:
            portfolio_min_distance = float(portfolio_min_distance)
        except (TypeError, ValueError):
            raise ValueError(
                "portfolio_min_distance must be a finite value between 0 "
                "and 1 in normalized RMS recipe space."
            )

        if (
            not math.isfinite(portfolio_min_distance)
            or portfolio_min_distance < 0.0
            or portfolio_min_distance > 1.0
        ):
            raise ValueError(
                "portfolio_min_distance must be a finite value between 0 "
                "and 1 in normalized RMS recipe space."
            )

        if (
            'target_ei' in acquisition_modes
            and getattr(self, 'incumbent_target_error_nm', None) is None
        ):
            raise ValueError(
                "Target-EI is explicitly requested in acquisition_modes, "
                "but no QC-approved condition-level incumbent target error "
                "is available. The portfolio batch will not be substituted "
                "or partially executed."
            )

        original_mode = self.acquisition_mode
        original_selected_recipes = getattr(
            self,
            '_portfolio_selected_normalized_recipes',
            []
        )
        original_min_distance = getattr(
            self,
            'portfolio_min_distance',
            0.05
        )

        self.portfolio_min_distance = portfolio_min_distance
        self._portfolio_selected_normalized_recipes = []
        selection_records = []

        try:
            for selection_index, acquisition_mode in enumerate(
                acquisition_modes
            ):
                self.acquisition_mode = acquisition_mode
                selected_recipe = np.asarray(
                    self.getNextReaction()[0],
                    dtype=float
                ).reshape(-1)

                nearest_distance = self._get_portfolio_nearest_distance(
                    selected_recipe
                )

                # A penalty can be returned only when every searched point is
                # excluded. Fail before controller preparation rather than
                # silently executing an ineffective duplicate condition.
                if (
                    nearest_distance is not None
                    and (
                        nearest_distance <= 1e-12
                        or nearest_distance < portfolio_min_distance
                    )
                ):
                    raise RuntimeError(
                        "Acquisition portfolio could not find a distinct "
                        f"feasible recipe for {acquisition_mode!r}. Nearest "
                        "selected normalized RMS distance was "
                        f"{nearest_distance:.6f}, below required "
                        f"{portfolio_min_distance:.6f}."
                    )

                selected_mask = getattr(self, 'last_selected_mask', None)
                selected_mask = (
                    None
                    if selected_mask is None
                    else np.asarray(selected_mask).astype(int).tolist()
                )
                mask_results = copy.deepcopy(
                    list(getattr(self, 'last_mask_results', []) or [])
                )
                feasible_mask_result_count = sum(
                    1
                    for result in mask_results
                    if (
                        result.get('x_full') is not None
                        and result.get('volume_balance') is not None
                        and result['volume_balance'].get(
                            'volume_feasible',
                            False
                        )
                    )
                )

                selection_records.append({
                    'acquisition_mode': acquisition_mode,
                    'normalized_recipe': selected_recipe.copy(),
                    'acquisition_score': getattr(
                        self,
                        'last_optimizer_acquisition_score',
                        None
                    ),
                    'balanced_exploration_weight': getattr(
                        self,
                        'last_optimizer_balanced_exploration_weight',
                        None
                    ),
                    'selected_mask': selected_mask,
                    'optimizer_method': getattr(
                        self,
                        'last_optimizer_method',
                        None
                    ),
                    'optimizer_success': getattr(
                        self,
                        'last_optimizer_success',
                        None
                    ),
                    'optimizer_status': getattr(
                        self,
                        'last_optimizer_status',
                        None
                    ),
                    'optimizer_message': getattr(
                        self,
                        'last_optimizer_message',
                        None
                    ),
                    'predicted_target_error_nm': getattr(
                        self,
                        'last_optimizer_predicted_target_error_nm',
                        None
                    ),
                    'predicted_lambda_mean_nm': getattr(
                        self,
                        'last_optimizer_predicted_lambda_mean_nm',
                        None
                    ),
                    'predicted_lambda_std_nm': getattr(
                        self,
                        'last_optimizer_predicted_lambda_std_nm',
                        None
                    ),
                    'incumbent_target_error_nm': getattr(
                        self,
                        'last_optimizer_incumbent_target_error_nm',
                        None
                    ),
                    'optimizer_volume_balance': copy.deepcopy(
                        getattr(
                            self,
                            'last_optimizer_volume_balance',
                            {}
                        ) or {}
                    ),
                    'mask_results': mask_results,
                    'mask_result_count': len(mask_results),
                    'feasible_mask_result_count': (
                        feasible_mask_result_count
                    ),
                    'portfolio_selection_index': selection_index,
                    'portfolio_acquisition_modes': list(
                        acquisition_modes
                    ),
                    'portfolio_min_distance': portfolio_min_distance,
                    'portfolio_nearest_distance': nearest_distance
                })
                self._portfolio_selected_normalized_recipes.append(
                    selected_recipe.copy()
                )

        finally:
            self.acquisition_mode = original_mode
            self._portfolio_selected_normalized_recipes = (
                original_selected_recipes
            )
            self.portfolio_min_distance = original_min_distance

        self.last_portfolio_selection_records = copy.deepcopy(
            selection_records
        )

        return selection_records

    def update_experiment_data(
        self,
        X_all,
        Y_all,
        X_new,
        Y_new
    ):
        """
        Refits the GP with the complete cumulative training dataset and keeps
        the GPyOpt optimizer object's stored X/Y history synchronized.

        The controller constructs X_all and Y_all by appending the newly
        completed, QC-approved batch to model.optimizer.X and
        model.optimizer.Y. Therefore, after every successful model update,
        those optimizer-side arrays must be replaced with the same cumulative
        arrays. Otherwise, the next controller update would append to stale
        seed-only data and silently discard earlier optimizer batches.

        params:
            np.ndarray X_all:
                Complete cumulative normalized recipe matrix, including the
                newly completed batch.

            np.ndarray Y_all:
                Complete cumulative normalized response column, including the
                newly completed batch.

            np.ndarray X_new:
                Normalized recipe matrix for only the newly completed batch.

            np.ndarray Y_new:
                Normalized response column for only the newly completed batch.

        returns:
            None
        """
        if self.gp_model is None:
            raise RuntimeError(
                "Cannot update Auto experiment data before the GP model has "
                "been initialized."
            )

        if self.optimizer is None:
            raise RuntimeError(
                "Cannot synchronize Auto experiment history before the "
                "GPyOpt optimizer has been initialized."
            )

        X_all_array = np.asarray(
            X_all,
            dtype=float
        )

        Y_all_array = np.asarray(
            Y_all,
            dtype=float
        )

        X_new_array = np.asarray(
            X_new,
            dtype=float
        )

        Y_new_array = np.asarray(
            Y_new,
            dtype=float
        )

        if X_all_array.ndim == 1:
            X_all_array = X_all_array.reshape(
                1,
                -1
            )

        if X_new_array.ndim == 1:
            X_new_array = X_new_array.reshape(
                1,
                -1
            )

        if Y_all_array.ndim == 1:
            Y_all_array = Y_all_array.reshape(
                -1,
                1
            )

        if Y_new_array.ndim == 1:
            Y_new_array = Y_new_array.reshape(
                -1,
                1
            )

        expected_dimension = self._get_dimension()

        if (
            X_all_array.ndim != 2
            or X_all_array.shape[1] != expected_dimension
        ):
            raise ValueError(
                "X_all must be a two-dimensional array with one column per "
                f"variable reagent. Expected {expected_dimension} columns, "
                f"received shape {X_all_array.shape}."
            )

        if (
            X_new_array.ndim != 2
            or X_new_array.shape[1] != expected_dimension
        ):
            raise ValueError(
                "X_new must be a two-dimensional array with one column per "
                f"variable reagent. Expected {expected_dimension} columns, "
                f"received shape {X_new_array.shape}."
            )

        if (
            Y_all_array.ndim != 2
            or Y_all_array.shape[1] != 1
        ):
            raise ValueError(
                "Y_all must be a two-dimensional single-column array. "
                f"Received shape {Y_all_array.shape}."
            )

        if (
            Y_new_array.ndim != 2
            or Y_new_array.shape[1] != 1
        ):
            raise ValueError(
                "Y_new must be a two-dimensional single-column array. "
                f"Received shape {Y_new_array.shape}."
            )

        if X_all_array.shape[0] != Y_all_array.shape[0]:
            raise ValueError(
                "X_all and Y_all must contain the same number of cumulative "
                f"observations. Received {X_all_array.shape[0]} and "
                f"{Y_all_array.shape[0]} rows."
            )

        if X_new_array.shape[0] != Y_new_array.shape[0]:
            raise ValueError(
                "X_new and Y_new must contain the same number of new "
                f"observations. Received {X_new_array.shape[0]} and "
                f"{Y_new_array.shape[0]} rows."
            )

        if X_new_array.shape[0] == 0:
            raise ValueError(
                "X_new and Y_new must contain at least one new observation."
            )

        if X_all_array.shape[0] < X_new_array.shape[0]:
            raise ValueError(
                "The cumulative dataset cannot contain fewer rows than the "
                "new batch."
            )

        if not np.all(np.isfinite(X_all_array)):
            raise ValueError(
                "X_all contains non-finite values."
            )

        if not np.all(np.isfinite(Y_all_array)):
            raise ValueError(
                "Y_all contains non-finite values."
            )

        if not np.all(np.isfinite(X_new_array)):
            raise ValueError(
                "X_new contains non-finite values."
            )

        if not np.all(np.isfinite(Y_new_array)):
            raise ValueError(
                "Y_new contains non-finite values."
            )

        self.gp_model.updateModel(
            X_all=X_all_array,
            Y_all=Y_all_array,
            X_new=X_new_array,
            Y_new=Y_new_array
        )

        # GPyOpt's model wrapper and ModularBayesianOptimization object store
        # their training arrays separately. Synchronize the optimizer object
        # only after the GP update succeeds so the next controller iteration
        # starts from the complete cumulative history.
        self.optimizer.X = np.array(
            X_all_array,
            dtype=float,
            copy=True
        )

        self.optimizer.Y = np.array(
            Y_all_array,
            dtype=float,
            copy=True
        )

        self.curr_iter += 1

        self.update_quit(
            X_new_array,
            Y_new_array
        )

        if getattr(self, 'terminal_verbosity', 'standard') != 'essential':
            print(
                "<<optimizer>> updated GP with "
                f"{X_all_array.shape[0]} cumulative observations "
                f"({X_new_array.shape[0]} new)"
            )
    

    def update_quit(self, X_new, Y_new):
        '''
        Checks whether the optimization process should terminate because the
        maximum number of Auto iterations has been reached.

        Target-based stopping is intentionally not handled here because Y_new
        contains physical replicate-well results. In duplicate-based Auto mode,
        stopping on any individual replicate is too permissive. The controller
        applies the final stop rule later using condition-level duplicate
        summary statistics.

        This method intentionally does not print an exit message. The controller
        owns user-facing stop messages so the terminal output does not contain
        duplicate max-iteration notices.
        '''
        if self.curr_iter >= self.max_iters:
            self.quit = True
        else:
            self.quit = False
        
