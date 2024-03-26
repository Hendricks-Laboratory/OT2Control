import GPyOpt
from pyDOE import lhs
from GPyOpt import Design_space


from abc import abstractmethod
import time  #TODO delete this debugging only
from abc import ABC
from abc import abstractmethod
import threading

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


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
    def __init__(self, bounds, target_value, reagent_info, fixed_reagents, variable_reagents, initial_design_numdata=15, batch_size=3, max_iters=10):
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
        self.model = None
        self.acquisition = None
        self.optimizer = None

    def check_bounds(self, suggestion):
        '''
        Checks if a suggested set of parameters is within the pre-defined bounds and constraints.
        params:
        list suggestion: The suggested parameters to be checked.
        returns:
        bool: True if the suggestion is within bounds, False otherwise.
        '''
        print("------checking bounds-------")
        print(f'reagent info: {self.reagent_info}')
        print(self.reagent_info.columns)
        print(f'fixed_reagents: {self.fixed_reagents}')
        print(f'variable_reagents: {self.variable_reagents}')
        print(f'suggestion is: {suggestion}')






        #  M1V1 = M1V2
        deck = self.reagent_info
        recipe = suggestion
        water_volume = 0

        total_volume = 200
        for x, y in recipe:
            total_volume -= (y*(200/deck[x]))
        # invalid case: total volume is greater than 200mL
        if total_volume < 0:
            return False
        
        # valid case: add water to get the 200mL concentration
        water_volume = total_volume
        print(f"Water volume is {water_volume}")
        return True
    
        
    def generate_initial_design(self):
        '''
        Generates an initial design of experiments using Latin Hypercube Sampling within the bounds.
        returns:
        np.ndarray: The initial set of parameters for the experiments.
        '''
        initial_design = GPyOpt.experiment_design.initial_design('latin', self.space, self.initial_design_numdata)
        # Here, you might want to filter or adjust initial_design based on constraints
        # This is a placeholder; actual implementation may require validating each point

        for recipe in initial_design:
            if not self.check_bounds(recipe):
                initial_design = np.delete(initial_design, np.where(initial_design == recipe))
        return initial_design

    def initialize_optimizer(self, X_init, Y_init):
        '''
        Initializes the Gaussian Process model and other components for Bayesian Optimization with initial experimental data.
        params:
        np.ndarray X_init: The initial parameter values for the experiments.
        np.ndarray Y_init: The initial objective function values corresponding to X_init.
        '''
        self.model = GPyOpt.models.GPModel(optimize_restarts=5, verbose=False)
        self.acq_optimizer = GPyOpt.optimization.acquisition_optimizer.AcquisitionOptimizer(self.space, optimizer='lbfgs')
        self.acquisition = GPyOpt.acquisitions.AcquisitionEI(self.model, self.space, self.acq_optimizer)
        self.evaluator = GPyOpt.core.evaluators.Sequential(self.acquisition)

        self.optimizer = GPyOpt.methods.ModularBayesianOptimization(
            self.model, self.space, None, self.acquisition, self.evaluator, X_init, Y_init)
    
    def _update_acquisition(self):
        '''
        Updates the acquisition function based on the current index and reinitializes the evaluator with the new acquisition function.
        '''
        acquisition_type = self.acquisition_functions[self.current_acquisition_index] 
        if acquisition_type == 'EI':
            self.acquisition = GPyOpt.acquisitions.AcquisitionEI(self.model, self.space, self.acq_optimizer)
        elif acquisition_type == 'MPI':
            self.acquisition = GPyOpt.acquisitions.AcquisitionMPI(self.model, self.space, self.acq_optimizer)
        elif acquisition_type == 'LCB':
            self.acquisition = GPyOpt.acquisitions.AcquisitionLCB(self.model, self.space, self.acq_optimizer)
        self.evaluator = GPyOpt.core.evaluators.Sequential(self.acquisition)
        #print('last,', self.acquisition)
        #print(self.evaluator)
    
    def get_target_wavelength(self):
        # TODO: something with evaluate_objective() maybe.
        pass

        
    def suggest_next_locations(self):
        '''
        Suggests the next locations (parameter sets) for experimentation based on the current acquisition function.
        returns:
        np.ndarray: The suggested parameters for the next experiments.
        '''
        self.current_acquisition_index = (self.current_acquisition_index + 1) % len(self.acquisition_functions)
        print(self.current_acquisition_index)
        print(self.acquisition_functions[self.current_acquisition_index])
        self._update_acquisition()
        suggestions = self.optimizer.suggest_next_locations()
        
        # loop to check suggestion and get new if out of bounds

        #has a call to self.check_bounds()

        return suggestions

    def update_experiment_data(self, X_new, Y_new):
        '''
        Updates the optimizer with new experimental data, extending the historical dataset.
        params:
        np.ndarray X_new: The new parameter values from the experiments.
        np.ndarray Y_new: The new objective function values corresponding to X_new.
        '''
        print(type(X_new))
        print(type(Y_new))
        self.experiment_data['X'].extend(X_new)
        self.experiment_data['Y'].extend(Y_new)
        self.initialize_optimizer(np.array(self.experiment_data['X']), np.array(self.experiment_data['Y']).reshape(-1, 1))
        self.curr_iter += 1
        self.update_quit(X_new, Y_new)
    
    def calc_obj(self, x):
        '''
        Calculates the objective function based on the difference from the target value.
        params:
        np.ndarray x: The experimental results to evaluate.
        returns:
        np.ndarray: The calculated objective function values.
        '''
        return abs(self.target_value - x).reshape(-1, 1)

    def update_quit(self, X_new, Y_new):
        '''
        Checks if the optimization process should be terminated based on the current iteration, maximum iterations, and the results.
        params:
        np.ndarray X_new: The latest parameter values from the experiments.
        np.ndarray Y_new: The latest objective function values corresponding to X_new.
        '''
        if self.curr_iter >= self.max_iters:
            self.quit = True
        else:
            # Check if any of the new results meet the target threshold condition.
            self.quit = any(y < self.threshold for y in Y_new)
        
