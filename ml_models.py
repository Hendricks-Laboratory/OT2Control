import GPyOpt
from pyDOE import lhs
from GPyOpt import Design_space


from abc import abstractmethod
import time  #TODO delete this debugging only
from abc import ABC
from abc import abstractmethod
import threading

from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C


class MLModel():
    """
    This is the base class of any machine learning model.
    It provides the basic interface that is required in order to use the
    ML model with the controller
    ATTRIBUTES:
        list<int> tids: a list of thread ids that this model has (or potentially some form of a
          thread executor object
        bool quit: True indicates MLModel is ready to quit, False indicates MLModel would like
          to keep going
        y_shape is the number of reagents you're guessing at in recipes
    """

    def __init__(self, model, max_iters=np.inf):
        self.curr_iter = 0
        self.max_iters = max_iters
        self.quit = False
        self.model = model
        self.model_lock = threading.Lock()
        self.X_lock = threading.Lock()
        self.X = None
        self.quit = self.update_quit()

    def train(self, X, y):
        """
        This is function is used to train the ML model.
        Internally, launches a private thread
        This call should wait on any current training threads to complete
        This call should launch a training thread to retrain the model on the new data
        params:
            np.array X: shape (num_pts, num_features) the recieved data for each new well
            np.array y: shape(num_pts, n_classes) the labels to predict
        """
        train_thread = threading.Thread(target=self._train, name='train thread', args=(X, y))
        train_thread.start()
        self.curr_iter += 1
        self.update_quit()

    @abstractmethod
    def _train(self, X, y):
        '''
        This method should take care of all training. It is expected that it will update
        the model in whatever way is fitting for your model. It will be called when the user
        calls train.  
        Use extreme caution when implementing this method, and note that self is NOT threadsafe.
        i.e. if you plan on using any of the attributes of this class, make sure you lock them
        appropriately, or only use them in this method (WITH A HUGE COMMENT SOMEWHERE)  
        The line of code in this method should almost always be, with self.model_lock: ...  
        Strictly speaking, the model_lock is overkill. Since we always join this method before
        calling predict which uses the model, but it's good practice if other methods ever use
        the model_lock.  
        As long as we freeze the ml_model while training, things are simple, and this allows
        the controller to run other commands while we're training, but I've implemented the
        architecture to have the ml_model do other work while it's training.  
        params:  
            np.array X: shape (num_pts, num_features) the recieved data for each new well  
            np.array y: shape(num_pts, n_classes) the labels to predict  
        Postconditions:  
            The model has been trained on the new data
        '''
        pass

    def predict(self):
        '''
        This is the starter code for any predict method. It must be overriden, but every override
        should first call super().predict(n_predictions)  
        This call should wait on the training thread to complete if it is has not been collected
        yet.  
        '''
        train_thread = [thread for thread in threading.enumerate() \
                        if thread.name == 'train thread']
        if train_thread:
            train_thread = train_thread[0]
            train_thread.join()  # wait till you're done training

    def update_quit(self):
        '''
        used to update the quit parameter of this model  
        This method will just check that you have not exceded max_iters, but should be
        extended by children to check if you've reached the target.  
        '''
        self.quit = self.curr_iter >= self.max_iters

    @abstractmethod
    def generate_seed_rxns(self):
        '''
        This method is called before the model is trained to generate a batch of training
        points  
        returns:  
            np.array: (batch_size,y.shape) 
        '''
        pass


class DummyMLModel(MLModel):
    '''
    This is the base class of any machine learning model.  
    It provides the basic interface that is required in order to use the
    ML model with the controller  
    ATTRIBUTES:  
        list<int> tids: a list of thread ids that this model has (or potentially some form of a
          thread executor object  
        bool quit: True indicates MLModel is ready to quit, False indicates MLModel would like
          to keep going  
        int curr_iter: formally, this is the number of times the train method has been called
        int max_iters: the number of iters to execute before quiting  
    '''

    def __init__(self, y_shape, max_iters=np.inf, batch_size=5):
        super().__init__(None, max_iters)  # don't have a model
        self.y_shape = y_shape
        self.batch_size = batch_size

    def _train(self, X, y):
        '''
        This call should wait on any current training threads to complete  
        This call should launch a training thread to retrain the model on the new data
        training is also where current iteration is updated  
        params:  
            np.array X: shape (num_pts, num_features) the recieved data for each new well  
            np.array y: shape(num_pts, n_classes) the labels to predict  
        Postconditions:  
            The model has been trained on the new data
        '''
        with self.model_lock:  # note for dummy this is not necessary, just an example
            print('<<ML>> training!')

    def predict(self):
        '''
        This call should wait on the training thread to complete if it is has not been collected
        yet.  
        params:  
            int n_predictions: the number of instances to predict  
        returns:  
            np.array: shape is n_predictions, y.shape. Features are pi e-2  
        '''
        with self.model_lock:
            print('<<ML>> generating preditions')
        
        # y_shape is currently set to 1: need to restructure somehow.
        return np.ones((self.batch_size, self.y_shape)) * 3.1415e-2

    def generate_seed_rxns(self):
        '''
        This method is called before the model is trained to generate a batch of training
        points  
        returns:  
            np.array: (batch_size,n_features) 
        '''
        if self.generate_seed_rxns.n_calls == 0:
            return np.ones((self.batch_size, self.y_shape)) * 3.1415e-2
        else:
            return np.ones((self.batch_size, self.y_shape)) * 2 * 3.1415e-2

    generate_seed_rxns.n_calls = 0

    def search_over_reaction_space(self, desiredAbsorption, tolerance, guess1, guess2):
        '''
        This method executes a binary search algorithm on reaction space by generating seed values
        and using the current model to predict the final max absorbence value of a reaction given
        these concentrations. It generates such seeds until it finds a set of values with predicted
        max absorbences on either side of the desired value. It then executes a prediction on the
        midpoint of these sets of values.
        '''
        lowAcceptable = desiredAbsorption - tolerance
        highAcceptable = desiredAbsorption + tolerance 
        if not guess1:
            guess1 = self.generate_seed_rxns()
        if not guess2:
            guess2 = self.generate_seed_rxns()
        if  self.predict(guess1) < highAcceptable and self.predict(guess1) > lowAcceptable:
            return guess1
        
        if self.predict(guess1) > highAcceptable:
            while (self.predict(guess2) > lowAcceptable):
                guess2 = self.generate_seed_rxns()
            #find midpoint
            guess1 = [(e1 + e2) / 2 for e1, e2 in zip(guess1, guess2)]
            self.search_over_reaction_space(self, desiredAbsorption, tolerance, guess1, guess2)


        if self.predict(guess1) < lowAcceptable:
            while (self.predict(guess2) < highAcceptable):
                guess2 = self.generate_seed_rxns()
            guess2 = [(e1 + e2) / 2 for e1, e2 in zip(guess1, guess2)]
            self.search_over_reaction_space(self, desiredAbsorption, tolerance, guess2, guess1)


class OptimizationModel(MLModel):
    def __init__(self, bounds, target_value, reagent_info, fixed_reagents, initial_design_numdata=15, batch_size=3, max_iters=10):
        super().__init__(None, max_iters)  # don't have a model
        self.bounds = bounds
        self.target_value = target_value
        self.reagent_info = reagent_info  # This should include 'a', 'b', 'c' values
        self.fixed_reagents = fixed_reagents  # Additional info, if needed for constraints
        self.space = GPyOpt.Design_space(bounds)
        self.constraints = self.define_constraints()
        self.initial_design_numdata = initial_design_numdata
        self.batch_size = batch_size
        self.experiment_data = {'X': [], 'Y': []}

        self.acquisition_functions = ['EI', 'MPI', 'LCB']
        self.current_acquisition_index = 0

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
        self.model = GPyOpt.models.GPModel(optimize_restarts=10, verbose=False)
        self.acquisition = GPyOpt.acquisitions.AcquisitionEI(self.model, self.space, optimizer='lbfgs')
        self.evaluator = GPyOpt.core.evaluators.Sequential(self.acquisition)
        
        self.optimizer = GPyOpt.methods.ModularBayesianOptimization(
            self.model, self.space, None, self.acquisition, self.evaluator, X_init, Y_init)
    
    def _update_acquisition(self):
        """Updates the acquisition function based on the current index and reinitializes the evaluator."""
        acquisition_type = self.acquisition_functions[self.current_acquisition_index]
        if acquisition_type == 'EI':
            self.acquisition = GPyOpt.acquisitions.AcquisitionEI(self.model, self.space, optimizer='lbfgs')
        elif acquisition_type == 'MPI':
            self.acquisition = GPyOpt.acquisitions.AcquisitionMPI(self.model, self.space, optimizer='lbfgs')
        elif acquisition_type == 'LCB':
            self.acquisition = GPyOpt.acquisitions.AcquisitionLCB(self.model, self.space, optimizer='lbfgs')
        self.evaluator = GPyOpt.core.evaluators.Sequential(self.acquisition)
        print('last,', self.acquisition)
        print(self.evaluator)
        
    def suggest_next_locations(self):
        """Suggests the next locations for experimentation, updating the acquisition function as needed."""
        self.current_acquisition_index = (self.current_acquisition_index + 1) % len(self.acquisition_functions)
        print(self.current_acquisition_index)
        print(self.acquisition_functions[self.current_acquisition_index])
        self._update_acquisition()
        self.optimizer = GPyOpt.methods.ModularBayesianOptimization(
          self.model, self.space, None, self.acquisition, self.evaluator, np.array(self.experiment_data['X']), self.experiment_data['Y'])

        return self.optimizer.suggest_next_locations()

    def update_experiment_data(self, X_new, Y_new):
        """Updates the model with new experimental data."""
        self.experiment_data['X'].extend(X_new)
        self.experiment_data['Y'].extend(Y_new)
        self.initialize_optimizer(np.array(self.experiment_data['X']), np.array(self.experiment_data['Y']).reshape(-1, 1))



