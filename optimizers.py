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
        total_volume=None
    ):
        '''
        Initializes the Auto optimization model.

        In addition to the normalized optimization bounds, this stores physical
        recipe context needed for true-zero-aware optimization. The controller
        remains the final source of truth for robot execution, but the optimizer
        needs the same concentration/volume context so it can avoid treating the
        forbidden 0-5 uL region as a normal continuous search space.

        params:
            list[dict] bounds:
                GPyOpt bounds for normalized model-space variables.

            float target_value:
                Target lambda max in nm.

            pd.DataFrame reagent_info:
                Reagent dataframe from the controller. Used to look up deck
                stock concentrations for variable reagents.

            list fixed_reagents:
                Fixed reagent names/volumes used by the controller.

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

        # Physical recipe context used by true-zero-aware optimization. These
        # are optional for backwards compatibility, but should be provided by
        # Auto mode before true-zero candidate repair is enabled.
        self.min_conc = min_conc
        self.max_conc = max_conc
        self.total_volume = total_volume

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

        The design is generated in normalized 0-1 model space using maximin
        Latin hypercube sampling. This keeps each reagent dimension stratified
        across its range while selecting the candidate design with the best
        space-filling behavior.

        returns:
            np.ndarray:
                Initial design matrix with shape:
                    initial_design_numdata x number_of_variable_reagents
        '''
        n_points = int(self.initial_design_numdata)
        n_dimensions = len(self.variable_reagents)

        initial_design = self._generate_maximin_lhs_design(
            n_points=n_points,
            n_dimensions=n_dimensions,
            n_candidates=500
        )

        print("<<optimizer>> generated maximin Latin hypercube initial design")
        print(f"<<optimizer>> initial design points: {n_points}")
        print(f"<<optimizer>> initial design dimensions: {n_dimensions}")
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

    def _target_distance_objective(self, x):
        '''
        Computes the target-distance objective for one normalized recipe point.

        The raw optimizer candidate is first repaired according to the Auto
        true-zero transfer rule, then evaluated with the Gaussian process model.
        This prevents the optimizer from treating the physically forbidden
        0-5 uL transfer region as a meaningful continuous search space.

        The objective is the squared distance between the model-predicted lambda
        max and the target lambda max. Lower values are better.

        params:
            np.ndarray x:
                One normalized recipe point with shape:
                    n_dimensions
                or:
                    1 x n_dimensions

        returns:
            float:
                Squared error between predicted lambda max and target_value.
        '''
        repaired_x = self._repair_normalized_candidate_for_true_zero(x)
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

        # Include the center point as a deterministic restart so every run has
        # at least one stable starting location.
        starting_points = [np.full(n_dimensions, 0.5)]

        # Add random restarts to improve global search behavior.
        starting_points.extend(
            np.random.random((n_restarts, n_dimensions))
        )

        for x0 in starting_points:
            result = minimize(
                fun=self._target_distance_objective,
                x0=x0,
                bounds=bounds,
                method='L-BFGS-B'
            )

            if result.success and result.fun < best_objective:
                best_objective = float(result.fun)
                best_x = np.clip(result.x, 0.0, 1.0)

        if best_x is None:
            raise RuntimeError(
                "Target-distance optimization failed from all restart points."
            )

        repaired_best_x = self._repair_normalized_candidate_for_true_zero(best_x)

        # Store both values for transparent debugging/reporting.
        self.last_raw_optimizer_candidate = best_x
        self.last_repaired_optimizer_candidate = repaired_best_x
        self.last_optimizer_objective = best_objective

        return repaired_best_x
    
    def _update_prediction_grid_for_plotting(self, grid_size=100):
        '''
        Updates self.predictions for the existing 2D GPR heatmap.

        The controller's plot_2D_GPR() function expects self.predictions to be
        a grid_size x grid_size array of predicted lambda max values in nm.
        That visualization only makes sense for exactly two variable reagents.

        For experiments with more than two variable reagents, this method sets
        self.predictions to None so the controller can skip the 2D heatmap
        cleanly.

        params:
            int grid_size:
                Number of grid points per axis for the 2D prediction heatmap.
        '''
        if self._get_dimension() != 2:
            self.predictions = None
            return

        grid_x, grid_y = np.meshgrid(
            np.linspace(0.0, 1.0, grid_size),
            np.linspace(0.0, 1.0, grid_size)
        )

        grid_points = np.stack(
            [grid_x.ravel(), grid_y.ravel()],
            axis=-1
        )

        normalized_predictions = self.gp_model.predict(grid_points)[0]

        self.predictions = (
            normalized_predictions
            .flatten()
            .reshape(grid_size, grid_size)
            .T * 600.0 + 300.0
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

        This method replaces the old brute-force 2D grid search with a
        dimension-general continuous optimizer. The optimizer searches normalized
        0-1 reagent space for the recipe whose predicted lambda max is closest
        to the target value.

        Optimizer candidates are repaired according to the Auto true-zero
        transfer rule before being scored and returned. This prevents the
        optimizer from treating the physically forbidden 0-5 uL transfer region
        as a meaningful continuous search space.

        For 2D experiments, this also updates self.predictions so the existing
        2D GPR heatmap can still be generated by the controller. For higher
        dimensional experiments, self.predictions is set to None because the
        existing heatmap is only valid for two variable reagents.

        returns:
            list[np.ndarray]:
                A single suggested normalized recipe point wrapped in a list.
                This preserves the controller-facing return format:
                    [array([...])]
        '''
        best_x = self._optimize_target_distance()
        predicted_lambda_max = self._predict_lambda_max_nm(best_x)

        self._update_prediction_grid_for_plotting()

        raw_x = getattr(self, 'last_raw_optimizer_candidate', None)
        repaired_x = getattr(self, 'last_repaired_optimizer_candidate', best_x)

        if raw_x is not None and not np.allclose(raw_x, repaired_x, rtol=0, atol=1e-9):
            print(
                f"<<optimizer>> true-zero repaired optimizer candidate "
                f"from {raw_x} to {repaired_x}"
            )

        print(
            f"<<optimizer>> suggested normalized recipe {best_x} "
            f"with predicted lambda max {predicted_lambda_max:.4f} nm"
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
        Checks if the optimization process should be terminated based on the current iteration, maximum iterations, and the results.
        params:
        np.ndarray X_new: The latest parameter values from the experiments.
        np.ndarray Y_new: The latest objective function values corresponding to X_new.
        '''
        if self.curr_iter >= self.max_iters:
            self.quit = True
            print("Exit due to max_iters")
        else:
            # Check if any of the new results meet the target threshold condition.
            self.quit = any(y < self.threshold for y in Y_new)
            print("Exit due to meeting target value")
        
