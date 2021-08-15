from abc import ABC
from abc import abstractmethod

import numpy as np

class MLModel():
    '''
    This is the base class of any machine learning model.  
    It provides the basic interface that is required in order to use the
    ML model with the controller  
    ATTRIBUTES:
        list<int> tids: a list of thread ids that this model has (or potentially some form of a
          thread executor object
        bool quit: True indicates MLModel is ready to quit, False indicates MLModel would like
          to keep going  
    '''
    quit = None

    @abstractmethod
    def train(self, X, y):
        '''
        This call should wait on any current training threads to complete  
        This call should launch a training thread to retrain the model on the new data
        params:
            np.array X: shape (num_pts, num_features) the recieved data for each new well
            np.array y: shape(num_pts, n_classes) the labels to predict
        '''
        pass

    @abstractmethod
    def predict(self, n_predictions):
        '''
        This call should wait on the training thread to complete if it is has not been collected
        yet.
        '''
        pass
        
class DummyMLModel():
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
    def __init__(self, max_iters, y_shape):
        self.max_iters = max_iters
        self.curr_iter = 0
        self.quit = self.update_quit()
        self.y_shape = y_shape

    def train(self, X, y):
        '''
        This call should wait on any current training threads to complete  
        This call should launch a training thread to retrain the model on the new data
        training is also where current iteration is updated
        params:
            np.array X: shape (num_pts, num_features) the recieved data for each new well
            np.array y: shape(num_pts, n_classes) the labels to predict
        '''
        print('training!')
        self.curr_iter += 1
        self.update_quit()
        pass

    def predict(self, n_predictions):
        '''
        This call should wait on the training thread to complete if it is has not been collected
        yet.
        '''
        print('generating preditions')
        return np.ones((n_predictions, self.y_shape)) / self.y_shape

    def update_quit(self):
        '''
        used to update the quit parameter of this model  
        '''
        self.quit =  self.curr_iter >= self.max_iters
