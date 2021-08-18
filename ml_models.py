from abc import ABC
from abc import abstractmethod
import threading

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
    def __init__(self, model, max_iters=np.inf):
        self.curr_iter = 0
        self.max_iters = max_iters
        self.quit = False
        self.model = model
        self.model_lock = threading.Lock()
        self.quit = self.update_quit()

    def train(self, X, y):
        '''
        This is function is used to train the ML model.  
        Internally, launches a private thread
        This call should wait on any current training threads to complete  
        This call should launch a training thread to retrain the model on the new data
        params:
            np.array X: shape (num_pts, num_features) the recieved data for each new well
            np.array y: shape(num_pts, n_classes) the labels to predict
        '''
        train_thread = threading.Thread(target=self._train, name='train thread', args=(X,y))
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
            train_thread.join() #wait till you're done training

    def update_quit(self):
        '''
        used to update the quit parameter of this model  
        This method will just check that you have not exceded max_iters, but should be
        extended by children to check if you've reached the target.  
        '''
        self.quit =  self.curr_iter >= self.max_iters
        
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
    def __init__(self, y_shape, max_iters=np.inf):
        super().__init__(None, max_iters) #don't have a model
        self.y_shape = y_shape

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
        with self.model_lock: #note for dummy this is not necessary, just an example
            print('training!')

    def predict(self, n_predictions):
        '''
        This call should wait on the training thread to complete if it is has not been collected
        yet.
        params:  
            int n_predictions: the number of instances to predict  
        returns:  
            np.array: shape is n_predictions, y.shape. Features are evenly distributed
        '''
        with self.model_lock:
            print('generating preditions')
        return np.ones((n_predictions, self.y_shape)) / self.y_shape
