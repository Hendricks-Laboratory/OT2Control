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
        allow_true_zero=False
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

        self.gp_model = None
        self.acquisition = None
        self.optimizer = None
        self.prediction = None
        self.predictions = None
        
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
            if self.allow_true_zero and np.allclose(
                repaired_candidate,
                0.0,
                rtol=0,
                atol=1e-9
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

        print("<<optimizer>> generated volume-feasible maximin initial design")
        print(f"<<optimizer>> initial design points: {n_points}")
        print(f"<<optimizer>> initial design dimensions: {n_dimensions}")
        print(f"<<optimizer>> feasible candidate points generated: {len(feasible_points)}")
        print(f"<<optimizer>> candidate generation attempts: {attempts}")
        print(f"<<optimizer>> minimum pairwise distance: {self._minimum_pairwise_distance(initial_design)}")

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
    
    def _generate_reagent_masks(self, include_all_off_mask=False):
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

        returns:
            list[np.ndarray]:
                List of binary masks, each with shape:
                    n_dimensions
        '''
        n_dimensions = self._get_dimension()
        masks = []

        for mask_int in range(2 ** n_dimensions):
            mask = np.array(
                [
                    (mask_int >> reagent_i) & 1
                    for reagent_i in range(n_dimensions)
                ],
                dtype=int
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
                include_all_off_mask=False
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
    
    def _masked_target_distance_objective(self, x_active, mask):
        '''
        Computes the target-distance objective for one masked candidate.

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
                Squared error between predicted lambda max and target_value, or
                a large finite penalty for volume-infeasible candidates.
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

        predicted_lambda_max = self._predict_lambda_max_nm(full_x)
        target_error = predicted_lambda_max - self.target_value

        return float(target_error ** 2)
    
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

            if volume_balance['volume_feasible']:
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
        Optimizes the target-distance objective within one ON/OFF reagent mask.

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

        for x0 in starting_points:
            result = minimize(
                fun=lambda x_active: self._masked_target_distance_objective(
                    x_active,
                    mask
                ),
                x0=x0,
                bounds=bounds,
                method='SLSQP'
            )

            if np.isfinite(result.fun) and result.fun < best_objective:
                best_objective = float(result.fun)
                best_x_active = np.asarray(result.x, dtype=float)

                # Keep the active result inside the executable active bounds.
                for i, (low, high) in enumerate(bounds):
                    best_x_active[i] = np.clip(best_x_active[i], low, high)

                best_full_x = self._expand_masked_candidate_to_full_recipe(
                    best_x_active,
                    mask
                )
                best_result_success = bool(result.success)
                best_result_message = result.message

        if best_full_x is None:
            return {
                'mask': mask,
                'success': False,
                'message': 'No finite optimizer result found for mask.',
                'objective': np.inf,
                'x_active': None,
                'x_full': None,
                'volume_balance': None,
                'predicted_lambda_max': None
            }

        volume_balance = self._get_candidate_volume_balance(best_full_x)

        predicted_lambda_max = None
        if volume_balance['volume_feasible']:
            predicted_lambda_max = self._predict_lambda_max_nm(best_full_x)

        return {
            'mask': mask,
            'success': best_result_success,
            'message': best_result_message,
            'objective': best_objective,
            'x_active': best_x_active,
            'x_full': best_full_x,
            'volume_balance': volume_balance,
            'predicted_lambda_max': predicted_lambda_max
        }
    
    def _optimize_target_distance_with_masks(self, n_restarts_per_mask=25):
        '''
        Finds the normalized recipe point predicted to be closest to the target
        lambda max using mixed discrete/continuous mask optimization.

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

        finite_results = [
            result for result in mask_results
            if result['x_full'] is not None and np.isfinite(result['objective'])
        ]

        if not finite_results:
            raise RuntimeError(
                "Mixed mask optimization failed: no finite optimizer result "
                "was found for any allowed reagent mask."
            )

        best_result = min(
            finite_results,
            key=lambda result: result['objective']
        )

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

        return best_x
    
    def _get_variable_reagent_stock_conc(self, reagent_name):
        '''
        Gets the stock concentration currently available on the deck for a
        variable reagent.

        Reagent containers are indexed by names like:
            silver_nitrateC0.375
            potassium_bromideC0.01

        This helper matches the base reagent name before the concentration
        marker and returns the deck concentration from reagent_info.

        params:
            str reagent_name:
                Base reagent name, such as 'silver_nitrate'.

        returns:
            float:
                Stock concentration of the reagent on the deck.
        '''
        matching_concs = []

        for reagent_container_name in self.reagent_info.index:
            reagent_container_name = str(reagent_container_name)

            if 'C' in reagent_container_name:
                base_name = reagent_container_name.split('C')[0]
            else:
                base_name = reagent_container_name

            if base_name == reagent_name:
                matching_concs.append(
                    float(self.reagent_info.loc[reagent_container_name, 'conc'])
                )

        if len(matching_concs) == 0:
            raise ValueError(
                f"Could not find stock concentration for variable reagent "
                f"{reagent_name} in reagent_info."
            )

        if len(matching_concs) > 1:
            raise ValueError(
                f"Found multiple stock concentrations for variable reagent "
                f"{reagent_name}: {matching_concs}. True-zero-aware optimization "
                f"currently expects one stock concentration per variable reagent."
            )

        return matching_concs[0]
    
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

        for reagent_i, reagent_name in enumerate(self.variable_reagents):
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
    
    def _get_variable_transfer_volumes_for_normalized_candidate(self, x):
        '''
        Converts one normalized optimizer candidate into variable reagent
        transfer volumes.

        The optimizer works in normalized 0-1 model space. This helper converts
        that candidate into target concentrations and then into the physical
        transfer volumes required to make those concentrations in the final
        reaction volume.

        The candidate is repaired with the true-zero rule before transfer
        volumes are calculated, so the returned volumes reflect executable
        robot behavior.

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

        repaired_x = self._repair_normalized_candidate_for_true_zero(x)

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

    def _get_candidate_volume_balance(self, x):
        '''
        Calculates the well-volume balance for one normalized optimizer
        candidate.

        This helper is dimension-general and mirrors the controller-side
        volume-balance logic. It does not rescale variable reagents as ratios.
        Variable reagent volumes are treated independently, and water fills the
        remaining space.

        A candidate is volume-feasible only if:
            1. fixed + variable volumes do not exceed the final reaction volume
            2. water top-off is either exactly 0 uL or at least 5 uL

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

        variable_transfer_volumes = self._get_variable_transfer_volumes_for_normalized_candidate(
            x
        )
        variable_volume_total = float(sum(variable_transfer_volumes.values()))

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
            volume_does_not_overflow
            and water_transfer_executable
        )

        return {
            'total_volume': total_volume,
            'fixed_volume_total': fixed_volume_total,
            'variable_transfer_volumes': variable_transfer_volumes,
            'variable_volume_total': variable_volume_total,
            'volume_before_water': volume_before_water,
            'water_volume': float(water_volume),
            'volume_does_not_overflow': bool(volume_does_not_overflow),
            'water_transfer_executable': bool(water_transfer_executable),
            'volume_feasible': bool(volume_feasible)
        }
    
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

        # Numerical safety: predictive standard deviation should not be
        # negative, but tiny negative values can appear from floating-point
        # artifacts or model-wrapper behavior.
        normalized_std = max(normalized_std, 0.0)

        predicted_lambda_mean_nm = normalized_mean * 600.0 + 300.0
        predicted_lambda_std_nm = normalized_std * 600.0

        return predicted_lambda_mean_nm, predicted_lambda_std_nm
    
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
        target_error = predicted_lambda_max - self.target_value

        return float(target_error ** 2)
    
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
        Updates 2D GP prediction grids for the controller heatmap plots.

        The controller's plot_2D_GPR() function expects self.predictions to be
        a grid_size x grid_size array of predicted lambda max values in nm.

        When available, self.prediction_uncertainty is also populated as a
        grid_size x grid_size array of GP predictive standard deviations in nm.

        These visualizations only make sense for exactly two variable reagents.

        For experiments with more than two variable reagents, this method sets
        both plotting grids to None so the controller can skip the 2D heatmaps
        cleanly.

        params:
            int grid_size:
                Number of grid points per axis for the 2D prediction and
                uncertainty heatmaps.
        '''
        if self._get_dimension() != 2:
            self.predictions = None
            self.prediction_uncertainty = None
            return

        grid_x, grid_y = np.meshgrid(
            np.linspace(0.0, 1.0, grid_size),
            np.linspace(0.0, 1.0, grid_size)
        )

        grid_points = np.stack(
            [grid_x.ravel(), grid_y.ravel()],
            axis=-1
        )

        normalized_predictions, normalized_prediction_stds = (
            self.gp_model.predict(grid_points)
        )

        self.predictions = (
            normalized_predictions
            .flatten()
            .reshape(grid_size, grid_size)
            .T * 600.0 + 300.0
        )

        self.prediction_uncertainty = (
            normalized_prediction_stds
            .flatten()
            .reshape(grid_size, grid_size)
            .T * 600.0
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

        For 2D experiments, this also updates self.predictions and
        self.prediction_uncertainty so the controller can generate 2D GP
        prediction and uncertainty heatmaps. For higher-dimensional
        experiments, both plotting grids are set to None because the heatmaps
        are only valid for exactly two variable reagents.

        The final selected recipe prediction is saved on the optimizer object
        so the controller can record pre-experiment model performance before
        the recipe is physically run.

        returns:
            list[np.ndarray]:
                A single suggested normalized recipe point wrapped in a list.
                This preserves the controller-facing return format:
                    [array([...])]
        '''
        best_x = self._optimize_target_distance_with_masks()
        predicted_lambda_max, predicted_lambda_std = (
            self.predict_lambda_distribution_nm(best_x)
        )

        self.last_optimizer_predicted_lambda_mean_nm = predicted_lambda_max
        self.last_optimizer_predicted_lambda_std_nm = predicted_lambda_std

        self._update_prediction_grid_for_plotting()

        selected_mask = getattr(self, 'last_selected_mask', None)

        if selected_mask is not None:
            print(
                f"<<optimizer>> selected reagent mask {selected_mask.tolist()} "
                f"for suggested recipe"
            )

        volume_balance = getattr(self, 'last_optimizer_volume_balance', None)

        if volume_balance is not None:
            print(
                f"<<optimizer>> suggested recipe volume balance: "
                f"fixed={volume_balance['fixed_volume_total']:.4f} uL, "
                f"variable={volume_balance['variable_volume_total']:.4f} uL, "
                f"water={volume_balance['water_volume']:.4f} uL, "
                f"feasible={volume_balance['volume_feasible']}"
            )

        print(
            f"<<optimizer>> suggested normalized recipe {best_x} "
            f"with predicted lambda max {predicted_lambda_max:.4f} nm "
            f"and GP predictive std {predicted_lambda_std:.4f} nm"
        )

        return [best_x]

    def update_experiment_data(self, X_all, Y_all, X_new, Y_new):
        '''
        Updates the optimizer with new experimental data, extending the historical dataset.
        params:
        np.ndarray X_new: The new parameter values from the experiments.
        np.ndarray Y_new: The new objective function values corresponding to X_new.
        '''
        
        self.gp_model.updateModel(X_all=X_all, Y_all=Y_all, X_new=X_new, Y_new=Y_new)
        
        self.curr_iter += 1
        self.update_quit(X_new, Y_new)
    

    def update_quit(self, X_new, Y_new):
        '''
        Checks whether the optimization process should terminate because the
        maximum number of Auto iterations has been reached.

        Target-based stopping is intentionally not handled here because Y_new
        contains physical replicate-well results. In duplicate-based Auto mode,
        stopping on any individual replicate is too permissive. The controller
        applies the target stop rule later using condition-level duplicate
        summary statistics.
        '''
        if self.curr_iter >= self.max_iters:
            self.quit = True
            print("Exit due to max_iters")
        else:
            self.quit = False
        
