import math
import GPyOpt
from pyDOE import lhs
from GPyOpt import Design_space
import GPy


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


        
        # Initialize an empty dictionary
        deck = {}

        # Iterate through the DataFrame index
        for index_value in self.reagent_info.index:
            # Use regex to remove "C" and the floating-point number that follows
            modified_index = re.sub(r'C\d+(\.\d+)?', '', index_value)
            # Assign the conc value to the corresponding key in the dictionary
            deck[modified_index] = self.reagent_info.loc[index_value, 'conc']

        print(f'deck: {deck}')
        # create deck variable holding dict of reagents available and their concs on deck. take out the suffix



        #  M1V1 = M1V2 
        water_volume = 0

        total_volume = 200
        i = 0
        for conc in suggestion:
            total_volume -= (float(conc)*(200/float(deck[self.variable_reagents[i]])))
            i += 1
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

        """for recipe in initial_design:
            if not self.check_bounds(recipe):
                initial_design = np.delete(initial_design, np.where(initial_design == recipe))"""
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

        
        kernel = GPy.kern.sde_Matern32(input_dim=1, variance=1.0, lengthscale=1, ARD=False, active_dims=None, name='Mat32')
        self.gp_model = GPyOpt.models.GPModel(kernel, noise_var=1e-4, optimize_restarts=0,verbose=False)
        self.gp_model.updateModel(X_init, Y_init, None, None)
        self.acq_optimizer = GPyOpt.optimization.acquisition_optimizer.AcquisitionOptimizer(self.space, optimizer='lbfgs')
        self.acquisition = GPyOpt.acquisitions.AcquisitionEI(self.gp_model, self.space, self.acq_optimizer)
        self.evaluator = GPyOpt.core.evaluators.Sequential(self.acquisition)
        objective = GPyOpt.core.task.objective.SingleObjective(f)

        self.optimizer = GPyOpt.methods.ModularBayesianOptimization(
            self.gp_model, self.space, objective, self.acquisition, self.evaluator, X_init, Y_init)
    
    def _update_acquisition(self):
        '''
        Updates the acquisition function based on the current index and reinitializes the evaluator with the new acquisition function.
        '''
        """if self.curr_iter < 6:
            self.acquisition = GPyOpt.acquisitions.AcquisitionLCB(self.model, self.space, self.acq_optimizer,exploration_weight=8)
        else:
            self.acquisition = GPyOpt.acquisitions.AcquisitionLCB(self.model, self.space, self.acq_optimizer, exploration_weight=0.5)
        self.evaluator = GPyOpt.core.evaluators.Sequential(self.acquisition)
        """
        acquisition_type = self.acquisition_functions[1] 
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

        
    def suggest(self):
        '''
        Suggests the next locations (parameter sets) for experimentation based on the current acquisition function.
        returns:
        np.ndarray: The suggested parameters for the next experiments.
        '''
        #self.current_acquisition_index = (self.current_acquisition_index + 1) % len(self.acquisition_functions)
        #print(self.current_acquisition_index)
        #print(self.acquisition_functions[self.current_acquisition_index])
        #self._update_acquisition()

        reagent_combinations = 10000
        num_steps = math.floor(reagent_combinations ** (1 / len(self.variable_reagents)))
        
        def normalize(x, min_val, max_val):
           return (x - min_val) / (max_val - min_val)
        def denormalize(x_normalized, min_val, max_val):
            return x_normalized * (max_val - min_val) + min_val

        temp = 0
        predictions = []
        stdev = []

        for i in range(200):
            pred, std = self.gp_model.predict(np.array([[temp]]))
            predictions.append(pred)
            stdev.append(std)
            temp += 0.005
        
        predictions = denormalize(np.array(predictions).flatten(),300, 900)
        stdev = np.array(stdev).flatten()*(600)
        linespace = np.linspace(0,0.002,200)

        closest = np.argmin(np.abs(predictions - self.target_value))
        best_lambda = predictions[closest]
        best = linespace[closest]
        print(f"{best} uM KBr results in a lambda max of {best_lambda} nm")
        #suggestions = self.optimizer.suggest_next_locations()
        
        # loop to check suggestion and get new if out of bounds

        #has a call to self.check_bounds()

        #TODO plot the model and save it after every iteration here 

        return best

    def update_experiment_data(self, X_new, Y_new):
        '''
        Updates the optimizer with new experimental data, extending the historical dataset.
        params:
        np.ndarray X_new: The new parameter values from the experiments.
        np.ndarray Y_new: The new objective function values corresponding to X_new.
        '''
        print(f"X_new: {X_new}")
        print(f"Y_new: {Y_new}")

        print(f"X_all Before Update: {self.optimizer.X}")
        print(f"Y_all Before Update: {self.optimizer.Y}")

        #print(f"Experiment Data X: {self.experiment_data["X"]}")
        self.gp_model.updateModel(X_all=np.array(self.experiment_data['X']), Y_all=np.array(self.experiment_data['Y']),X_new=X_new, Y_new=Y_new)
        self.experiment_data['X'].extend(X_new)
        self.experiment_data['Y'].extend(Y_new)
        #self.initialize_optimizer(np.array(self.experiment_data['X']), np.array(self.experiment_data['Y']).reshape(-1, 1))
        
        print(f"X_all After Update: {self.optimizer.X}")
        print(f"Y_all After Update: {self.optimizer.Y}")
        
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
        return abs(x - self.target_value).reshape(-1, 1)

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
        
