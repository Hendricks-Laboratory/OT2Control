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
    def __init__(self, bounds, reagent_info, fixed_reagents, initial_design_numdata=15, batch_size=3, max_iters=10):
        #super().__init__(None, max_iters)  # don't have a model
        self.bounds = bounds
        self.target_value = self.get_target_wavelength()
        self.reagent_info = reagent_info  # This should include 'a', 'b', 'c' values
        self.fixed_reagents = fixed_reagents  # Additional info, if needed for constraints
        self.space = GPyOpt.Design_space(bounds)
        self.constraints = self.define_constraints()
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

    def define_constraints(self):
        # Example constraint: x_1*a + x_2*b + x_3*c + d <= 200
        # 'a', 'b', 'c' values should be defined in reagent_info
        a, b, c = self.reagent_info['reagent2'], self.reagent_info['reagent3'], self.reagent_info['reagent4']
        constraint_expr = f'x[:,0]*{a} + x[:,1]*{b} + x[:,2]*{c} + {sum([fr[1] for fr in self.fixed_reagents])} <= 200'
        return [{'name': 'volume_constraint', 'constraint': constraint_expr}]

    def generate_initial_design(self):
        """Generates initial design using Latin Hypercube Sampling and checks constraints."""
        initial_design = GPyOpt.experiment_design.initial_design('latin', self.space, self.initial_design_numdata)
        # Here, you might want to filter or adjust initial_design based on constraints
        # This is a placeholder; actual implementation may require validating each point
        return initial_design

    def initialize_optimizer(self, X_init, Y_init):
        """Initializes the BO components with the initial experimental data."""
        self.model = GPyOpt.models.GPModel(optimize_restarts=20, verbose=False)
        self.acq_optimizer = GPyOpt.optimization.acquisition_optimizer.AcquisitionOptimizer(self.space, optimizer='lbfgs')
        self.acquisition = GPyOpt.acquisitions.AcquisitionEI(self.model, self.space, self.acq_optimizer)
        self.evaluator = GPyOpt.core.evaluators.Sequential(self.acquisition)

        self.optimizer = GPyOpt.methods.ModularBayesianOptimization(
            self.model, self.space, None, self.acquisition, self.evaluator, X_init, Y_init)
    
    def _update_acquisition(self):
        """Updates the acquisition function based on the current index and reinitializes the evaluator."""
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
        """Suggests the next locations for experimentation, updating the acquisition function as needed."""
        self.current_acquisition_index = (self.current_acquisition_index + 1) % len(self.acquisition_functions)
        print(self.current_acquisition_index)
        print(self.acquisition_functions[self.current_acquisition_index])
        self._update_acquisition()
        #self.optimizer = GPyOpt.methods.ModularBayesianOptimization(
          #self.model, self.space, None, self.acquisition, self.evaluator, np.array(self.experiment_data['X']), self.experiment_data['Y'])

        return self.optimizer.suggest_next_locations()

    def update_experiment_data(self, X_new, Y_new):
        """Updates the model with new experimental data."""
        print(type(X_new))
        print(type(Y_new))
        self.experiment_data['X'].extend(X_new)
        self.experiment_data['Y'].extend(Y_new)
        self.initialize_optimizer(np.array(self.experiment_data['X']), np.array(self.experiment_data['Y']).reshape(-1, 1))
        self.curr_iter += 1
        self.update_quit(X_new, Y_new)

    def update_quit(self, X_new, Y_new):
        '''
        used to update the quit parameter of this model  
        This method will just check that you have not exceded max_iters, but should be
        extended by children to check if you've reached the target.  
        '''
        if self.curr_iter >= self.max_iters:
            self.quit = True
        else:
            # Check if any of the new results meet the target threshold condition.
            self.quit = any(y < self.threshold for y in Y_new)
        
