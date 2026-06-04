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
    def __init__(self, bounds, target_value, reagent_info, fixed_reagents, variable_reagents, initial_design_numdata, batch_size, max_iters):
        #super().__init__(None, max_iters)  # don't have a base model
        self.bounds = bounds
        self.target_value = target_value
        self.reagent_info = reagent_info
        self.fixed_reagents = fixed_reagents  # Additional info, if needed for constraints
        self.variable_reagents = variable_reagents
        self.space = GPyOpt.Design_space(bounds)
        #self.constraints = self.define_constraints()
        self.initial_design_numdata = initial_design_numdata
        self.batch_size = batch_size
        self.experiment_data = {'X': [], 'Y': []}
        self.threshold = 1
        self.quit=False
        self.acquisition_functions = ['EI', 'MPI', 'LCB']
        self.current_acquisition_index = 0
        self.curr_iter = 0
        self.max_iters = max_iters
        self.gp_model = None
        self.acquisition = None
        self.optimizer = None
        self.prediction = None
        
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

        normalized_prediction = self.optimizer.model.predict(x)[0]
        normalized_prediction = float(normalized_prediction.flatten()[0])

        return normalized_prediction * 600.0 + 300.0

    def _target_distance_objective(self, x):
        '''
        Computes the target-distance objective for one normalized recipe point.

        The objective is the squared distance between the model-predicted lambda
        max and the target lambda max. Lower values are better.

        This preserves the current exploitation behavior of choosing the recipe
        whose predicted lambda max is closest to the target.

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
        predicted_lambda_max = self._predict_lambda_max_nm(x)
        target_error = predicted_lambda_max - self.target_value

        return float(target_error ** 2)
    
    def _optimize_target_distance(self, n_restarts=50):
        '''
        Finds the normalized recipe point predicted to be closest to the target
        lambda max.

        This replaces the old brute-force 2D grid search with a continuous,
        dimension-general optimization. Multiple random restarts are used to
        reduce the chance of getting stuck in a poor local optimum.

        params:
            int n_restarts:
                Number of random starting points to try.

        returns:
            np.ndarray:
                Best normalized recipe point found, with shape:
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

        return best_x
    
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

        normalized_predictions = self.optimizer.model.predict(grid_points)[0]

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
        
