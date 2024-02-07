import GPyOpt
from pyDOE import lhs

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


class LinReg(MLModel):
    '''
    params:
        tuple<int> scan_bounds: size 2. If you wish to ignore aspects of the
          scan, and only focus
          on a single peak for learning, you may specify manually the start
          and stop index of the data you are interested in. Only this data
          will be used for training.
        int duplication: This is used to copy the reactions you're running if
          you are worried about redundancy. the number is the number of times
          you duplicate each reaction.
    Model to use Linear Regression algorithm
    model_lock also locks X
    UNIMPLEMENTED:  
      only runs for batch size of 1  
    '''

    def __init__(self, model, final_spectra, y_shape, max_iters, batch_size=1,
                 scan_bounds=None, duplication=1):
        super().__init__(model, max_iters)  # don't have a model
        self.scan_bounds = scan_bounds
        if scan_bounds:
            # if you only want to pay attention in bounds, predict on those vals
            self.FINAL_SPECTRA = final_spectra[:, scan_bounds[0]:scan_bounds[1]]
        else:
            self.FINAL_SPECTRA = final_spectra
        self.y_shape = y_shape
        self.batch_size = batch_size
        self.duplication = duplication

    def generate_seed_rxns(self):
        """
        This method is called before the model is trained to generate a batch of training
        points
        returns:
            np.array: (batch_size,n_features)
        """
        upper_bound = 2.5
        lower_bound = 0.25
        recipes = np.random.rand(self.batch_size, self.y_shape) \
                  * (upper_bound - lower_bound) + lower_bound
        print("seed,", recipes)
        recipes = np.repeat(recipes, self.duplication, axis=0)
        return recipes

        def predict(self):
            '''
            This call should wait on the training thread to complete if it is has not been collected
            yet.  
            params:  
                int n_predictions: the number of instances to predict  
            returns:  
                np.array: shape is n_predictions, y.shape. Features are pi e-2  
            '''
            super().predict()
            
            with self.model_lock:
                self.FINAL_SPECTRA = self.FINAL_SPECTRA.reshape(1, -1)  # Reshape if FINAL_SPECTRA is a single sample
                y_pred = self.model.predict(self.FINAL_SPECTRA)
            print("predicted", y_pred)
        
            # Repeat y_pred self.duplication times along both axes
            y_pred_repeated = np.repeat(y_pred, self.duplication, axis=0)
            y_pred_repeated = np.repeat(y_pred_repeated, self.duplication, axis=1)
        
            return y_pred_repeated



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
        # NOTE we may get fancier in the future here and do more preprocessing
        if self.scan_bounds:
            # if you only want to pay attention in bounds, train on those vals
            processedX = X[:, self.scan_bounds[0]:self.scan_bounds[1]]
        else:
            processedX = X
        # update the data with the new scans
        time.sleep(40)
        print('<<ML>> training')
        with self.model_lock:
            if isinstance(self.X, np.ndarray):
                self.X = np.concatenate((self.X, processedX))
                self.y = np.concatenate((self.y, y))
            else:
                self.X = processedX
                self.y = y
            print("model fitting on X", self.X)
            print("model fitting on y", self.y)
            self.model.fit(self.X, self.y)
        print('<<ML>> done training')


# This class is not being used
class GradientDescent(MLModel):
    def __init__(self, model, final_spectra, y_shape, max_iters, learning_rate, batch_size=1,
                 scan_bounds=None, duplication=1):
        super().__init__(model, max_iters)  # don't have a model
        self.scan_bounds = scan_bounds

        if scan_bounds:
            # if you only want to pay attention in bounds, predict on those vals
            self.FINAL_SPECTRA = final_spectra[:, scan_bounds[0]:scan_bounds[1]]
        else:
            self.FINAL_SPECTRA = final_spectra

        self.learning_rate = 0.01
        self.y_shape = y_shape
        self.batch_size = batch_size
        self.duplication = duplication
        self.parameters = {"W": 0.0, "b": 0.0}

    def predict(self):
        pass

    def _train(self, X, y):
        """
        This call should wait on any current training threads to complete
        This call should launch a training thread to retrain the model on the new data
        training is also where current iteration is updated
        params:
            np.array X: shape (num_pts, num_features) the recieved data for each new well
            np.array y: shape(num_pts, n_classes) the labels to predict
        Postconditions:
            The model has been trained on the new data
        Returns
        """

        costs = []
        """
        Loop for max_iters:
            get Y_pred
            calcualte and record cost at step (for debugging)
            update coefficients            
        """

        # Choose whether to trim data to within certain bounds
        if self.scan_bounds:
            proccessedX = X[:, self.scan_bounds[0]:self.scan_bounds[1]]
        else:
            processedX = X

        m = self.y_shape
        alpha = self.learning_rate
        for _ in range(self.max_iters):
            Y_pred = self.predict()

            # Compute and record cost
            cost = (1 / 2 * m) * (np.sum(Y_pred - y) ** 2)
            costs.append(cost)

            # update coefficients
            self.parameters["W"] = self.parameters["W"] - (alpha * ((1 / m) * np.sum(Y_pred - y)))
            self.parameters["b"] = self.parameters["b"] - (alpha * ((1 / m) * np.sum((Y_pred - y) * X)))

            #


class SegmentedLinReg(MLModel):
    '''
    params:
        tuple<int> scan_bounds: size 2. If you wish to ignore aspects of the
          scan, and only focus
          on a single peak for learning, you may specify manually the start
          and stop index of the data you are interested in. Only this data
          will be used for training.
        int duplication: This is used to copy the reactions you're running if
          you are worried about redundancy. the number is the number of times
          you duplicate each reaction.
    Model to use Linear Regression algorithm
    model_lock also locks X
    UNIMPLEMENTED:
      only runs for batch size of 1
    '''

    def __init__(self, model, final_spectra, y_shape, max_iters, batch_size=1,
                 scan_bounds=None, duplication=1):
        super().__init__(model, max_iters)  # don't have a model
        self.scan_bounds = scan_bounds
        if scan_bounds:
            # if you only want to pay attention in bounds, predict on those vals
            self.FINAL_SPECTRA = final_spectra[:, scan_bounds[0]:scan_bounds[1]]
        else:
            self.FINAL_SPECTRA = final_spectra
        self.y_shape = y_shape
        self.batch_size = batch_size
        self.duplication = duplication

    def generate_seed_rxns(self):
        """
        This method is called before the model is trained to generate a batch of training
        points
        returns:
            np.array: (batch_size,n_features)
        """
        upper_bound = 2.5
        lower_bound = 0.25
        recipes = np.random.rand(self.batch_size, self.y_shape) \
                  * (upper_bound - lower_bound) + lower_bound
        print("seed,", recipes)
        recipes = np.repeat(recipes, self.duplication, axis=0)
        return recipes

    def predict(self):
        '''
        This call should wait on the training thread to complete if it is has not been collected
        yet.
        params:
            int n_predictions: the number of instances to predict
        returns:
            np.array: shape is n_predictions, y.shape. Features are pi e-2
        '''
        super().predict()
        with self.model_lock:
            y_pred = self.model.predict(self.FINAL_SPECTRA)
        print("predicted", y_pred)
        breakpoint()
        return np.repeat(y_pred, self.duplication, axis=0)

    def training(self, df, target, r_val):
        '''
            Method to train the model.
            params:
                df                 : dataset created on each experiement
                target             : asked peak wavelength (input)
                r_val              : number of training to produce the correspoing image
            returns:
                input_user         : same or recomputed asked peak wavelength (input)
                user_concentration : predicted concentration for the asked peak wavelength (input)
                train_prediction   : predicted peak wavelengths of all data
                W                  : parameter W value
                b                  : parameter b value
            '''

        def getting_params(concentration, wavelength):
            W = sum(wavelength * (concentration - np.mean(concentration))) / sum(
                (concentration - np.mean(concentration)) ** 2)
            b = np.mean(wavelength) - W * np.mean(concentration)
            print("--->", W, b)
            return W, b

        print("-----> Model Training")
        print("Computing W and b")
        W, b = getting_params(df['Concentration'], df['Wavelength'])

        initial_predict = df['Concentration'] * W + b
        initial_error = df['Wavelength'] - initial_predict

        print("----")
        print("initial_predict:", initial_predict)
        print("----")
        print("initial_error", initial_error)

        def piecewise_linear(x, x0, y0, k1, k2):
            return np.piecewise(x, [x < x0, x >= x0],
                                [lambda x: k1 * x + y0 - k1 * x0, lambda x: k2 * x + y0 - k2 * x0])

        # optimized_params, covars = optimize.curve_fit(piecewise_linear, df['Concentration'], df['Wavelength'])


class PolynomialRegression(MLModel):
    def __init__(self, model, final_spectra, y_shape, max_iters, max_order, reagents_to_vary,
                 batch_size=1, scan_bounds=None, duplication=1, degree=6):
        super().__init__(model,
                         max_iters)  # don't have a model (this comment was present when the JGB team began work, but its meaning is unclear)
        self.scan_bounds = scan_bounds
        if scan_bounds:
            # if you only want to pay attention in bounds, predict on those vals
            self.FINAL_SPECTRA = final_spectra[:, scan_bounds[0]:scan_bounds[1]]
        else:
            self.FINAL_SPECTRA = final_spectra
        self.y_shape = y_shape
        self.batch_size = batch_size
        self.duplication = duplication
        self.max_order = max_order
        self.regresser = None
        self.reagents_varied = reagents_to_vary
        self.poly_degree = degree
        self.error_record = {}
        self.train_call = 0

    # This code should resemble what's done in the LinearRegression class from Diego's MLA branch
    # The key difference is that we need to use self.regresser instead of a model instance this means
    # that we will need to look at the sklearn docs to see what attributes the sklearn.LinearRegression class
    # has and then add new features while needed (i.e. probably some extra caching capability)
    def predict(self, X):
        if self.regresser is None:
            raise Exception("Cannot make a prediction until training has occured at least once")
        else:
            return self.regressor.predict(PolynomialFeatures(degree=self.poly_degree).fit_transform(X))

    # TODO: Figure out optimization with this approach
    # Resources for algorithm:
    # https://scikit-learn.org/stable/modules/linear_model.html#polynomial-regression-extending-linear-models-with-basis-functions
    def training(self, df):
        """
            Method to train the model.
            params:
                df                 : dataset created on each experiment
                r_val              : number of training to produce the corresponding image
            returns:
                minErrorPred       : wavelength prediction made by the optimal model
                minErrorIndex      : Degree of the polynomial that results in the minimum error
        """
        X = df.iloc[:, :].values
        # Changed this df name for testing purposes
        y = df['Wavelength'].values
        # _Y = df['Wavelength'].values

        poly_regressor = PolynomialFeatures(degree=self.poly_degree)
        X_poly = poly_regressor.fit_transform(X)
        poly_regressor.fit(X_poly, y)

        # first run
        if self.regresser is None:
            linear_regressor = LinearRegression()
            linear_regressor.fit(X_poly, y)

            self.regresser = linear_regressor

        else:
            # second run, simply refit and record error
            self.regresser.fit(X_poly, y)

        error = mean_squared_error(y, self.regresser.predict(X_poly), squared=False)
        self.error_record[self.train_call] = "Iteration {}\nMean Squared Error={}".format(self.train_call, error)
        print(self.error_record[self.train_call])
        self.train_call += 1

    def generate_seed_rxns(self):
        """
        randomly selects some parameter values from which
        exploration/training will begin
        returns:
            np.array: (batch_size, n_features)

        for now, we have to hard-code in bounds for these params
        we want to be able to read in those bounds from elsewhere;
        this functionality will be implemented in a "warm-up" method and this method will be deprecated (hopefully)


        this version of the seed generation method is very similar to the LinReg version;
        the difference is we need to pass the order of the polynomial (?) the number of parameters we want to vary (?)
        """
        upper_bound = 2.5
        lower_bound = 0.25
        recipes = np.random.rand(self.batch_size, self.y_shape) * (upper_bound - lower_bound) + lower_bound
        print("seed, " + np.array_str(recipes))
        recipes = np.repeat(recipes, self.duplication, axis=0)
        return recipes

    def warmup(self):
        """
        this method is an extension of the functionality of generate_seed_rxns(). its purpose is to
        assign variables and boundaries to the experimental run before training the model, as well
        as generating seed reactions
        """

        pass

class BayesianOptMLModel(MLModel):
    """
    Bayesian Optimization Machine Learning Model

    ATTRIBUTES:
        int y_shape: the shape of the y data.
        GPy.kern.src.rbf.RBF kernel: the kernel used for the Bayesian Optimization.
        np.array bounds: the bounds for the Bayesian Optimization.
        np.array X: the input data for the model.
        np.array y: the output data for the model.

    INHERITED ATTRIBUTES:
        int curr_iter
        int max_iters
        bool quit
        GPyOpt.methods.bayesian_optimization.BayesianOptimization model
        threading.Lock model_lock
        threading.Lock X_lock

    METHODS:
        train(X, y) None: see base class
        _train(X, y) None: trains the BayesianOptMLModel on the input data X and output data y.
        predict() np.array: makes predictions using the BayesianOptMLModel.
        mse() float: Calculates the Mean Squared Error (mse) for the current model configuration.
        generate_seed_rxns() np.array: generates seed reactions for the BayesianOptMLModel. 
        
    INHERITED METHODS:
        update_quit()

    """
    def __init__(self, y_shape, kernel, max_iters=np.inf):
        super().__init__(None, max_iters)
        self.y_shape = y_shape
        self.kernel = kernel
        self.bounds = np.array([(0, 1)]*y_shape)
        self.X = np.empty((0, y_shape)) 
        self.y = np.empty((0, 1)) 

    def _train(self, X, y):
        with self.model_lock:
            if self.X is None or self.y is None:
                self.X = X
                self.y = y
            else:
                self.X = np.concatenate([self.X, X])
                self.y = np.concatenate([self.y, y])
            self.model = GPyOpt.methods.BayesianOptimization(
                f=None, domain=self.bounds, X=self.X, Y=self.y, kernel=self.kernel)

    def predict(self):
        super().predict()
        with self.model_lock:
            return self.model.predict()

    def mse(self):
        """
        Calculates the Mean Squared Error (mse) for the current model configuration.
        
        return: float, which represents the mse.
        """
        x_train, x_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=0)
        self.model.fit(x_train, y_train)
        # somehow call the predict function to get our target?
        # predict gives off
        y_pred = predict()
        return  mean_squared_error(y_test, y_pred)
      
    def _mse(self, y_target, y_pred):
        """
        Calculates the Mean Squared Error (mse) for the current model configuration.
        
        return: float, which represents the mse.
        """
        return  mean_squared_error(y_target, y_pred)
        
    def generate_seed_rxns(self):
        x_next = self.model.suggest_next_locations()
        return x_next


class LatinHypercubeMLModel(MLModel):
    """
    Latin Hypercube Sampling Machine Learning Model

    ATTRIBUTES:
        int y_shape: the shape of the y data.
        int batch_size: the size of the batch to generate using Latin Hypercube Sampling.

    INHERITED ATTRIBUTES:
        int curr_iter
        int max_iters
        bool quit
        threading.Lock model_lock
        threading.Lock X_lock

    METHODS:
        train(X, y) None: see base class
        _train(X, y) None: doesn't do anything as LatinHypercubeMLModel doesn't train.
        predict() None: doesn't make predictions as LatinHypercubeMLModel doesn't predict.
        generate_seed_rxns() np.array: generates seed reactions using Latin Hypercube Sampling.
        
    INHERITED METHODS:
        update_quit()

    """
    def __init__(self, y_shape, max_iters, batch_size):
        super().__init__(None, max_iters)
        self.y_shape = y_shape
        self.max_iters = max_iters
        self.batch_size = batch_size
        self.quit = self.update_quit()

    def _train(self, X, y):
        pass

    def predict(self):
        return None

    def generate_seed_rxns(self):
        latinHyperSample = lhs(self.y_shape, samples=self.batch_size)
        return latinHyperSample
    
class BOGP(BaseModel):
    def __init__(self, bounds, kernel=None):
        self.bounds = bounds
        self.kernel = kernel
        self.X = None  # Stores training inputs
        self.Y = None  # Stores training outputs
        self.model = BayesianOptimization(f=None, domain=self.bounds, kernel=self.kernel, model_type='GP')
       
    def train(self, X_new, Y_new):
        if self.X is None or self.Y is None:
            self.X = X_new
            self.Y = Y_new
        else:
            # Append new data to existing
            self.X = np.vstack([self.X, X_new])
            self.Y = np.vstack([self.Y, Y_new])
       
        # Update the model with the aggregated data
        self.model.updateModel(self.X, self.Y)    
               
    def generate_suggested_recipes(self):
        suggestions = []
        for acquisition in ['EI', 'MPI', 'LCB']:
            x_next = self.model.suggest_next_locations(acquisition=acquisition)
            suggestions.append(x_next)
        return suggestions