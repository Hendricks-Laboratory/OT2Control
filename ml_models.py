import time  # TODO delete this debugging only
from abc import ABC
from abc import abstractmethod
import threading

from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


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
            y_pred = self.model.predict(self.FINAL_SPECTRA)
        print("predicted", y_pred)
        breakpoint()
        return np.repeat(y_pred, self.duplication, axis=0)

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
                 batch_size=1, scan_bounds=None, duplication=1):
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
        self.max_order = max_order
        self.regresser = None
        self.reagents_varied = reagents_to_vary

    def generate_seed_rxns(self):
        pass

    def predict(self):
        if self.regresser is None:
            raise Exception("Cannot make a prediction until training has occured at least once")
        pass

    # TODO: Figure out optimization with this approach
    def training(self, df, r_val):
        """
            Method to train the model.
            params:
                df                 : dataset created on each experiment
                r_val              : number of training to produce the corresponding image
            returns:
                minErrorPred       : wavelength prediction made by the optimal model
                minErrorIndex      : Degree of the polynomial that results in the minimum error
        """
        reagentConcentrations = df[[s for s in self.reagents_varied]].values
        # Changed this df name for testing purposes
        _Y = df['price'].values
        # _Y = df['Wavelength'].values

        accuracyRecord = {}
        polyRecord = {}
        for order in range(1, self.max_order+1):
            # Generate fitted model
            poly = PolynomialFeatures(degree=order, include_bias=False)
            _X = poly.fit_transform(reagentConcentrations)

            # Do a single regression to get error
            regresser = LinearRegression()
            regresser.fit(_X, _Y)
            Y_pred = regresser.predict(_X)

            # Record accuracy
            accuracyRecord[order] = mean_squared_error(_Y, Y_pred, squared=False)
            polyRecord[order] = (regresser, Y_pred)

        # Visualize poly degree vs error
        # x_axis = range(1, self.max_order+1)
        # plt.scatter(x_axis, accuracyRecord.values(), color="green")
        # plt.plot(x_axis, accuracyRecord.values(), color="red")
        # plt.xlabel("Polynomial model degree")
        # plt.ylabel("Mean squared error")
        # plt.show()

        minErrorIndex = min(accuracyRecord, key=accuracyRecord.get)
        minErrorModel, minErrorPred = polyRecord[minErrorIndex]

        self.regresser = minErrorModel
        return minErrorPred, minErrorIndex
