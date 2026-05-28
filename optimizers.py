import math
import GPyOpt
from pyDOE import lhs
from GPyOpt import Design_space
import GPy

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
        
    def _generate_maximin_lhs_design(self, num_candidates=1000, random_seed=None):
        '''
        Generates a maximin Latin hypercube initial design.

        Parameters:
            num_candidates:
                Number of Latin hypercube candidate designs to generate and compare.

            random_seed:
                Optional random seed for reproducibility. If None, the design will vary
                between runs.

        Returns:
            np.ndarray:
                The Latin hypercube design with the largest minimum pairwise distance
                among the generated candidates.

        Postconditions:
            - Returns an initial design with shape:
              (self.initial_design_numdata, number_of_variables).
            - Preserves Latin hypercube-style sampling through GPyOpt's latin design.
            - Selects the candidate whose closest pair of points is as far apart as possible.
        '''
        if random_seed is not None:
            np.random.seed(random_seed)

        best_design = None
        best_min_distance = -np.inf

        for _ in range(num_candidates):
            candidate_design = GPyOpt.experiment_design.initial_design(
                'latin',
                self.space,
                self.initial_design_numdata
            )

            min_distance = np.inf

            for i in range(len(candidate_design)):
                for j in range(i + 1, len(candidate_design)):
                    distance = np.linalg.norm(candidate_design[i] - candidate_design[j])

                    if distance < min_distance:
                        min_distance = distance

            if min_distance > best_min_distance:
                best_min_distance = min_distance
                best_design = candidate_design

        print("Using maximin Latin hypercube initial design")
        print(f"Best minimum pairwise distance: {best_min_distance}")

        return best_design
    
    def generate_initial_design(self):
        '''
        Generates an initial design of experiments using a maximin Latin
        hypercube design within the bounds.

        Returns:
            np.ndarray:
                The initial set of parameters for the experiments.
        '''
        initial_design = self._generate_maximin_lhs_design(
            num_candidates=1000,
            random_seed=None
        )

        return initial_design

    def initialize_optimizer(self, X_init, Y_init):
        '''
        Initializes the Gaussian Process model and other components for Bayesian Optimization with initial experimental data.
        params:
        np.ndarray X_init: The initial parameter values for the experiments.
        np.ndarray Y_init: The initial objective function values corresponding to X_init.
        '''
        def f(x):
            return abs(sum(x)-(self.target_value*3))

        
        kernel = GPy.kern.sde_Matern32(input_dim=2, variance=1.0, lengthscale=1, ARD=False, active_dims=None, name='Mat32')
        self.gp_model = GPyOpt.models.GPModel(kernel, noise_var=1e-4, optimize_restarts=0,verbose=False)
        self.gp_model.updateModel(X_init, Y_init, None, None)
        self.acq_optimizer = GPyOpt.optimization.acquisition_optimizer.AcquisitionOptimizer(self.space, optimizer='lbfgs')
        self.acquisition = GPyOpt.acquisitions.AcquisitionEI(self.gp_model, self.space, self.acq_optimizer)
        self.evaluator = GPyOpt.core.evaluators.Sequential(self.acquisition)
        objective = GPyOpt.core.task.objective.SingleObjective(f)

        self.optimizer = GPyOpt.methods.ModularBayesianOptimization(
            self.gp_model, self.space, objective, self.acquisition, self.evaluator, X_init, Y_init)
        
    def getNextReaction(self, num_points=100):
        '''
        Suggests the next locations (parameter sets) for experimentation based on the current acquisition function.
        returns:
        np.ndarray: The suggested parameters for the next experiments.
        max_conc: The maximum concentration of the variable reagents.
        '''

        predictions = []
        stdev = []
        reagent_combinations = 10000
        num_steps = math.floor(reagent_combinations ** (1 / len(self.variable_reagents)))
        
        def denormalize(x_normalized, min_val, max_val):
            return x_normalized * (max_val - min_val) + min_val
   
        #calculates the cartesian product that creates all possible recipes of two reagents (normalized)
        #TODO for more dimensions these two lines must be redone
        grid_x, grid_y = np.meshgrid(np.linspace(0,1,num_points),np.linspace(0,1,num_points))
        concentrations = np.stack([grid_x.ravel(), grid_y.ravel()], axis=-1)

        #for every possible recipes stored in concentration a prediction and a standard devation is calculated by gpr
        for i in range(len(concentrations)):
            pred, std = self.gp_model.predict(np.array([concentrations[i]]))
            predictions.append(pred)
            stdev.append(std)

        #this variable is made for accessability during plotting (Controller: plot_2d_gpr)
        self.predictions = (np.concatenate(predictions).flatten().reshape(100,100).T*600 + 300) 

        #the gpr predictions and standard devations are normalized, here we denormalize them to find the next reaction to run
        predictions = denormalize(np.array(predictions).flatten(),300, 900)
        stdev = np.array(stdev).flatten()*(600)
        
        #for exploration we find the maximum error and choose the reaction that corresponds to it 
        maximum = np.argmax(stdev)
        explore = concentrations[maximum]
        print(f"The maximum uncertainty {maximum} occurs at low")
        
        #for exploitation we find the closest lambda max and the reaction that corresponds to it
        closest = np.argmin(np.abs(predictions - self.target_value))
        exploit = concentrations[closest]
        print(f"{exploit} results in a predicted lambda max of {predictions[closest]} nm")
        
        #if the uncertainty everywhere is bellow a threashold we let the robot exploit, otherwise we explore
        return [exploit]
    
        # TODO: Maximum number of rounds of exploration is a variable that can be put in header parameters
        if all(unc < 100 for unc in stdev):
            print("Exploiting!!!")
            return [exploit]
        else:
            print("Exploring!!!")
            return[explore]


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
        
