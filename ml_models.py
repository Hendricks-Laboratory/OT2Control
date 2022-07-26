from re import M
import time #TODO delete this debugging only
from abc import ABC
from abc import abstractmethod
import threading
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt
import  pandas as pd
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.utils import plot_model
import tensorboard

import keras

from datetime import datetime
import random
import os
import math
import sys

import tensorflow as tf
from tensorflow.keras.utils import plot_model
import tensorboard

import keras

import numpy as np


from datetime import datetime
import random
from tensorflow.keras.layers import Input, Dense, Activation, Concatenate
from tensorflow.keras.models import Model
from keras import backend as K







#i
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
        y_shape is the number of reagents you're guessing at in recipes
    '''
    def __init__(self, model, max_iters=np.inf):
        self.curr_iter = 0
        self.max_iters = max_iters
        self.quit = False
        self.model = model
        self.df = None
        self.r= None
        self.input_user=None
        self.N = 3
        self.recipes = 3
        self.learning_rate = 0.05
        self.model_lock = threading.Lock()
        self.X_lock = threading.Lock()
        self.X = None
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
    def mainModel(self,N,recipes,learning_rate,X=None,Y=None):

        pass

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
        # pass
        train_thread = [thread for thread in threading.enumerate() if thread.name == 'train thread']
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

    @abstractmethod
    def generate_seed_rxns(self):
        '''
        This method is called before the model is trained to generate a batch of training
        points  
        returns:  
            np.array: (batch_size,y.shape) 
        '''
        pass
    @abstractmethod
    def prediction(self):
        '''
        This method is for predict
        '''
        train_thread = [thread for thread in threading.enumerate() if thread.name == 'train thread']
        if train_thread:
            train_thread = train_thread[0]
            train_thread.join() #wait till you're done training



    # @abstractmethod
    # def training(self):
    def Train(self,df, input_user,r):
        
        '''
        This is function is used to train the ML model.  
        Internally, launches a private thread
        This call should wait on any current training threads to complete  
        This call should launch a training thread to retrain the model on the new data
        params:
            np.array X: shape (num_pts, num_features) the recieved data for each new well
            np.array y: shape(num_pts, n_classes) the labels to predict
        '''
        train_thread = threading.Thread(target=self.training, name='train thread', args=(df,input_user,r))
        train_thread.start()
        self.curr_iter += 1
        self.update_quit()

    @abstractmethod
    def training(self, df,input_user,r):
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
        super().__init__(None, max_iters) #don't have a model
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
        with self.model_lock: #note for dummy this is not necessary, just an example
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
            return np.ones((self.batch_size,self.y_shape)) * 3.1415e-2
        else:
            return np.ones((self.batch_size,self.y_shape)) * 2 * 3.1415e-2
    generate_seed_rxns.n_calls = 0


#################################################################



class LinearRegress(MLModel):
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
        super().__init__(model, max_iters) #don't have a model
        self.scan_bounds = scan_bounds
        if scan_bounds:
            #if you only want to pay attention in bounds, predict on those vals
            self.FINAL_SPECTRA = final_spectra[:, scan_bounds[0]:scan_bounds[1]]
        else:
            self.FINAL_SPECTRA = final_spectra
        self.y_shape = y_shape
        self.batch_size = batch_size
        self.duplication = duplication

    def generate_seed_rxns(self,number_recipes):
        '''
        This method is called before the model is trained to generate a batch of training
        points  
        returns:  
            np.array: (batch_size,n_features) 
        '''
        upper_bound = 0.003/2
        lower_bound = 0.00025/2
        recipes =  np.random.rand(number_recipes,1) * (upper_bound - lower_bound) + lower_bound 
        recipes = np.repeat(recipes, self.duplication, axis=0)
        # recipes= recipes.T
        #print("our recipes", recipes)
        return recipes
    
    
    #NEW PREDICTION
    def prediction(self, modelCall):

        '''
        This call should wait on the training thread to complete if it is has not been collected
        yet.  
        params:  
            int n_predictions: the number of instances to predict  
        returns:  
            np.array: shape is n_predictions, y.shape. Features are pi e-2  
        '''
        super().prediction()
        with self.model_lock:
            y_pred = self.FINAL_SPECTRA
        print("predicted NH", y_pred)
        
  
        # def predictLinearModel(X, W, b):
        #     #Overcome Overflow dtype=np.uint32
        #     X= np.array([[X]],dtype=float)
        #     W= np.array([W],dtype=float)
        #     b= np.array([b],dtype=float)
        #     print("predicting",X)
        #     print("predictor W",W)
        #     print("predictor W",b)
            
        #     Y_hat = X * W + b
        #     return Y_hat

        def predictLinearModelInverse(Y_wt, W, b):
            #Overcome Overflow dtype=np.uint32
            Y_wt= np.array([[Y_wt]],dtype=float)
            W= np.array([W],dtype=float)
            b= np.array([b],dtype=float)
            print("predicting",Y_wt)
            print("predictor W",W)
            print("predictor W",b)
            
            X_predicted = (Y_wt -b) /W
            return X_predicted




        def plots_error_avg(model):
            print(len(model["cacheErrorAvg"]))
            min_index=np.min(model["cacheErrorAvg"])
            max_index=np.max(model["cacheErrorAvg"])
            print("NNNNNNNNN",model["cacheErrorAvg"])
            print(min_index)
            errorsSca=[]
            for i in range(len(model["cacheErrorAvg"])):
                errorsSca.append((model["cacheErrorAvg"][i]-min_index)/(max_index- min_index))
            print("AAAAAAAAA",errorsSca)
            plt.plot([i for i in range(1,model["break_epoch"]+1)],errorsSca)
            plt.show()
            
            return plt.show()# for saving
            #plt.savefig('pic.png')

        
        print("b predict",modelCall)
        #print(",",modelCall["cacheErrorAvg"])
        plots_error_avg(modelCall)
        predictQuestion = input("Do you want to make a prediction: [Yes / No ]")
        if predictQuestion == "Yes" or predictQuestion=="y":
            predict = input("Please enter recipe:")
            #prediction = predictLinearModel(predict,modelCall["ParamsToUse"]["Theta"], modelCall["ParamsToUse"]["Bias"])
            prediction = predictLinearModelInverse(predict,modelCall["ParamsToUse"]["Theta"], modelCall["ParamsToUse"]["Bias"])
            print("Predicted concentration [] given a wavelenght", prediction)
            #ADDING DELETE IF NO PROB
            breakpoint()

            return {"inputPredictor":predict, "prediction":prediction , "par_theta":modelCall["ParamsToUse"]["Theta"], "par_bias":modelCall["ParamsToUse"]["Bias"] }
        
        else:
            
            #ADDING DELETE IF NO PROB
            breakpoint()
            
            return {"prediction":0, "par_theta":modelCall["ParamsToUse"]["Theta"], "par_bias":modelCall["ParamsToUse"]["Bias"] }
        
        #return np.repeat(y_pred, self.duplication, axis=0);
    
    
    #NEW TRAIN
    def training(self, df,input_user,r):

            def getting_params(self, concentration,wavelength):
                print("Parasn", concentration)
                W = sum(wavelength*(concentration-np.mean(concentration))) / sum((concentration-np.mean(concentration))**2)
                b = np.mean(wavelength) - W*np.mean(concentration)
                print("--->", W,b)
                return W, b
            
            #Training //Computing the parameters
            print("---->Model training")
            print("Computing W and b")

            W, b = getting_params(self,df['Concentration'],df['Wavelength']) 

            #making predictions based on our current data to see plot and the error
            print("making predictions based on our current data to see plot and the error")
            train_prediction= df['Concentration'] * W + b
            #our error 
            training_error= df['Wavelength']-train_prediction
            print("----")
            print("Train_prediction:",train_prediction)
            print("----")
            print("Error", training_error)
            #ploting
            train_fig = plt.figure(num=None, figsize=(4, 4),dpi=300, facecolor='w', edgecolor='k') 
            plt.plot(df['Concentration'], train_prediction, color='red',label="Predicted Wavelength by Linear Model")
            plt.scatter(df['Concentration'], df['Wavelength'], label="Training Data")
            plt.xlabel("[KBr] Concentration (mM)")
            plt.ylabel("Wavelength (nm)")
            plt.legend(prop={"size":6})
            plt.show()
            train_fig.savefig("training-"+str(r)+"png",dpi=train_fig.dpi)
            #User input for Wavelength wanted


            #conputing the inverse --> from Wavelength to Concentration
            input_user= input_user
            while True:
                user_concentration = (input_user- b) / W
                print("Model predictied concentration given the wavelength: ",user_concentration)
                if user_concentration < 0.00025 or user_concentration >0.003:
                    print("Sorry, the wanted wavelength is not reached given the current concentration of KBr allowed to process")
                    input_user= input("Please ENTER the desire Wavelength: ")
                    input_user= float(input_user)
                    
                    #user_concentration = (input_user- b) / W

                else:

                    break


            print("------------")
            print("Making prediction")
            print("passing ", user_concentration, " to robot")
            return input_user, user_concentration, train_prediction , W, b
    

    ##OLD TRAIN
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

        #NOTE we may get fancier in the future here and do more preprocessing
        # if self.scan_bounds:
        #     #if you only want to pay attention in bounds, train on those vals
        #     processedX = X[:, self.scan_bounds[0]:self.scan_bounds[1]]
        # else:
        #     processedX = X
        #update the data with the new scans
        time.sleep(40)
        # print('<<ML>> training')
        with self.model_lock:
        #     if isinstance(self.X,np.ndarray):
        #         self.X = np.concatenate((self.X, processedX))
        #         self.y = np.concatenate((self.y, y))
        #     else:
        #         self.X = processedX
        #         self.y = y
        #     print("model fitting on X", self.X)
        #     print("model fitting on y", self.y)
        #     self.model.fit(self.X, self.y)
        # print('<<ML>> done training'
            print("X and Y")
            print(X,y) 
            print("--------")
            print('<<ML>> training')

            # ml_model_trained = MainModel(self,10,3,0.05,X,y)

            print('<<ML>> done training')

            return 0 #MainModel(self, X,y)



###########################



class NeuralNet(MLModel):
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
        super().__init__(model, max_iters) #don't have a model
        self.scan_bounds = scan_bounds
        if scan_bounds:
            #if you only want to pay attention in bounds, predict on those vals
            self.FINAL_SPECTRA = final_spectra[:, scan_bounds[0]:scan_bounds[1]]
        else:
            self.FINAL_SPECTRA = final_spectra
        self.y_shape = y_shape
        self.batch_size = batch_size
        self.duplication = duplication

    def generate_seed_rxns(self,number_recipes):
        '''
        This method is called before the model is trained to generate a batch of training
        points  
        returns:  
            np.array: (batch_size,n_features) 
        '''

       
                #Boundary for Cit
        
        upper_bound_Cit = 8.125
        lower_bound_Cit = 0.3125
        
        #Boundary for Ag
        
        upper_bound_Ag = 0.24375
        lower_bound_Ag = 0.0094
        
        #Boundary for KBr
        upper_bound_KBr = 0.0005
        lower_bound_KBr = 0.00025
        
        
        Upper = upper_bound_Ag + upper_bound_Cit + upper_bound_KBr
        Lower = lower_bound_Ag + lower_bound_Cit + lower_bound_KBr
        
        #Generating random 
        
        con_KBr = np.random.random_sample(size= (3,)) * (upper_bound_KBr- lower_bound_KBr) + (lower_bound_KBr) 
        con_Cit = np.random.random_sample(size= (3,)) * (upper_bound_Cit- lower_bound_Cit) + (lower_bound_Cit)
        con_Ag  = np.random.random_sample(size= (3,))  * (upper_bound_Ag- lower_bound_Ag) + (lower_bound_Ag)
        
        
        ##print("Concen (T)",con_Cit.T,con_Ag.T,con_KBr.T,type(con_Ag))
        print("Concen (T)",con_Cit.T,con_Ag.T,con_KBr.T)

        Nextt= False
        Ngen=0
        while True:
            for r in range(3):
                print("rr",r)
                print("Checking creations",(con_KBr[r]*(200)/0.005 + con_Cit[r]*(200)/ 12.5+ con_Ag[r]*(200)/0.375))
                if (con_KBr[r]*(200)/0.005 + con_Cit[r]*(200)/ 12.5+ con_Ag[r]*(200)/0.375) > 130:

                    con_KBr[r] = random.random() * (upper_bound_KBr- lower_bound_KBr) + (lower_bound_KBr) 
                    con_Cit[r] = random.random() * (upper_bound_Cit- lower_bound_Cit) + (lower_bound_Cit)
                    con_Ag[r]  = random.random()  * (upper_bound_Ag- lower_bound_Ag) + (lower_bound_Ag)
                    Nextt = True
                    Ngen +=1
                    break
                else:

                    if con_KBr[r] == upper_bound_KBr:
                        con_Cit[r] = 0
                        con_Ag[r]  = 0

                    if con_Cit[r] == upper_bound_Cit:
                        con_KBr[r] = 0
                        con_Ag[r]  = 0

                    if con_Ag[r] == upper_bound_Ag:
                        con_KBr[r] = 0
                        con_Cit[r]  = 0
                    Nextt= False
            if Nextt== True:
                0
            elif Nextt == False:
                break
                
        print("N",Ngen)
        print("Concen after checking (T)",con_Cit.T,con_Ag.T,con_KBr.T)
        print("")


        print("Creating recipes.....")
        
        print("")
        
        recipes2 = np.concatenate((con_Cit, con_Ag,con_KBr), axis=None)
        print(recipes2)
        print("")
        #for re in range(len(con_Cit)):
        res1=[con_Cit[0],con_Ag[0],con_KBr[0]]
        res2=[con_Cit[1],con_Ag[1],con_KBr[1]]
        res3=[con_Cit[2],con_Ag[2],con_KBr[2]]
        print("SSS",res1)
        
        
        
        
        recipes = np.concatenate((np.array([res1]), np.array([res2]),np.array([res3])), axis=0)
        recipes = np.repeat(recipes, 1, axis=0)
        print("rrecipes",recipes)
        
        res_1 = recipes[0]
        res_1 = np.expand_dims(res_1, axis=1)
        res_1 = res_1.T
        res_1 = np.concatenate((res_1, res_1, res_1), axis=0)


        
        res_2 = recipes[1]
        res_2 = np.expand_dims(res_2, axis=1)
        res_2 = res_2.T
        res_2 = np.concatenate((res_2, res_2, res_2), axis=0)

        
        res_3 = recipes[2]
        res_3 = np.expand_dims(res_3, axis=1)
        res_3 = res_3.T
        res_3 = np.concatenate((res_3, res_3, res_3), axis=0)


        #print("llllll",res_1,res_2,res_3)
        recipe_l= np.concatenate((res_1, res_2,res_3), axis=0)
        print(recipe_l)
        return recipe_l
    
    
    #NEW PREDICTION
    def prediction(self, Test_input):

        '''
        This call should wait on the training thread to complete if it is has not been collected
        yet.  
        params:  
            int n_predictions: the number of instances to predict  
        returns:  
            np.array: shape is n_predictions, y.shape. Features are pi e-2  
        '''
        super().prediction()
        with self.model_lock:
            y_pred = self.FINAL_SPECTRA
        print("predicted NH", y_pred)
        

        # # def predictLinearModelInverse(Y_wt, W, b):
        # #     #Overcome Overflow dtype=np.uint32
        # #     Y_wt= np.array([[Y_wt]],dtype=float)
        # #     W= np.array([W],dtype=float)
        # #     b= np.array([b],dtype=float)
        # #     print("predicting",Y_wt)
        # #     print("predictor W",W)
        # #     print("predictor W",b)
            
        # #     X_predicted = (Y_wt -b) /W
        # #     return X_predicted




        # # def plots_error_avg(model):
        # #     print(len(model["cacheErrorAvg"]))
        # #     min_index=np.min(model["cacheErrorAvg"])
        # #     max_index=np.max(model["cacheErrorAvg"])
        # #     print("NNNNNNNNN",model["cacheErrorAvg"])
        # #     print(min_index)
        # #     errorsSca=[]
        # #     for i in range(len(model["cacheErrorAvg"])):
        # #         errorsSca.append((model["cacheErrorAvg"][i]-min_index)/(max_index- min_index))
        # #     print("AAAAAAAAA",errorsSca)
        # #     plt.plot([i for i in range(1,model["break_epoch"]+1)],errorsSca)
        # #     plt.show()
            
        # #     return plt.show()# for saving
        # #     #plt.savefig('pic.png')

        
        print("b predict",Test_input)
        #print(",",modelCall["cacheErrorAvg"])
        #plots_error_avg(modelCall)
        predictQuestion = input("Do you want to make a prediction: [Yes / No ]")
        if predictQuestion == "Yes" or predictQuestion=="y":
            predict = input("Please enter recipe:")

            print("Generate predictions for 3 samples")
            print("Self mo",self.model)
            y_pred = self.model.predict(np.array(Test_input))#Test_input)
            print("predictions shape:", y_pred.shape)
            print("predictions:",y_pred)
            return y_pred
    
            #prediction = predictLinearModel(predict,modelCall["ParamsToUse"]["Theta"], modelCall["ParamsToUse"]["Bias"])
            prediction = predictLinearModelInverse(predict,modelCall["ParamsToUse"]["Theta"], modelCall["ParamsToUse"]["Bias"])
            print("Predicted concentration [] given a wavelenght", prediction)
            #ADDING DELETE IF NO PROB
            breakpoint()

            return {"inputPredictor":predict, "prediction":prediction , "par_theta":modelCall["ParamsToUse"]["Theta"], "par_bias":modelCall["ParamsToUse"]["Bias"] }
        
        else:
            
            #ADDING DELETE IF NO PROB
            breakpoint()
            
            return {"prediction":0}
            # return {"prediction":0, "par_theta":modelCall["ParamsToUse"]["Theta"], "par_bias":modelCall["ParamsToUse"]["Bias"] }
        
        #return np.repeat(y_pred, self.duplication, axis=0);


    
    #NEW TRAIN
    
    def training(self, df,input_user,r, n_epochs=30, ml_past=False, explo= False):
        

        #type_net = input("Which would you like to use [W-O-->Concent (1)][Concent-->W-O (2)][Concent-->W-->Concent (3)]")
        # while True:
        #     if type_net == "1" or type_net=="2" or type_net=="3":
        #         break
        #     else:
        #         print("Try again")
        #         type_net = input("Which would you like to use [W-O-->Concent (1)][Concent-->W-O (2)][Concent-->W-->Concent (3)]")
        # type_net = input("exploration? yes/y no/n")
        # while True:
        #     if type_net == "y" or "yes" or "no" or "n":
        #         break
        #     else:
        #         print("Try again")
        #         type_net= input("exploration? yes/y no/n")
        # print(type_net)
        if explo==True:
        
            train_size= math.floor(len(df)*(80/100))
            val_size= int((len(df)-train_size)/2)
            test_size= int(len(df)-train_size-val_size)

            train_pre =[]
            for rto in range(train_size):
                train_pre.append((df["Wavelength"][:train_size][rto],df["Observance"][:train_size][rto]))
        
            val_pre =[]
            for ret in range(train_size,train_size+val_size):
                val_pre.append((df["Wavelength"][train_size:train_size+val_size][ret],df["Observance"][train_size:train_size+val_size][ret]))
        
            test_pre =[]
            for retr in range(train_size+val_size,train_size+val_size+test_size):
                test_pre.append((df["Wavelength"][train_size+val_size:train_size+val_size+test_size][retr],df["Observance"][train_size+val_size:train_size+val_size+test_size][retr]))

            train_label_pre =[]
            for ee in range(train_size):
                train_label_pre.append([df["[Cit]"][:train_size][ee],df["[Ag]"][:train_size][ee],df["[KBr]"][:train_size][ee]])

            val_label_pre =[]
            for eer in range(train_size,train_size+val_size):
                val_label_pre.append([df["[Cit]"][train_size:train_size+val_size][eer],df["[Ag]"][train_size:train_size+val_size][eer],df["[KBr]"][train_size:train_size+val_size][eer]])

            test_label_pre =[]
            for eerr in range(train_size+val_size,train_size+val_size+test_size):
                test_label_pre.append([df["[Cit]"][train_size+val_size:train_size+val_size+test_size][eerr],df["[Ag]"][train_size+val_size:train_size+val_size+test_size][eerr],df["[KBr]"][train_size+val_size:train_size+val_size+test_size][eerr]])



            Train_input = np.array(train_pre)
            Train_label = np.array(train_label_pre)

            Val_input   = np.array(val_pre)
            Val_label   = np.array(val_label_pre)
            
            Test_input  = np.array(test_pre)
            Test_label  = np.array(test_label_pre)


            

            upper_bound_Cit = 8.125
            lower_bound_Cit = 0.3125
            
            #Boundary for Ag
            
            upper_bound_Ag = 0.24375
            lower_bound_Ag = 0.0094
            
            #Boundary for KBr
            upper_bound_KBr = 0.0005
            lower_bound_KBr = 0.00025

            def bounded_output_Cit(x):
                scale = upper_bound_Cit - lower_bound_Cit
                return scale * tf.nn.sigmoid(x) + lower_bound_Cit

            def bounded_output_Ag(x):
                scale = upper_bound_Ag - lower_bound_Ag
                return scale * tf.nn.sigmoid(x) + lower_bound_Ag

            def bounded_output_KBr(x):
                scale = upper_bound_KBr - lower_bound_KBr
                return scale * tf.nn.sigmoid(x) + lower_bound_KBr
            

            from keras.utils.generic_utils import get_custom_objects

            get_custom_objects().update({'bounded_output_Cit': Activation(bounded_output_Cit)})
            get_custom_objects().update({'bounded_output_Ag': Activation(bounded_output_Ag)})
            get_custom_objects().update({'bounded_output_KBr': Activation(bounded_output_KBr)})

            #Model
            A1 = Input(shape=(3,),name='A1')
            A2 = Dense(32, activation='relu',name='A2')(A1)
            A3 = Dense(16, activation='relu',name='A3')(A2)
            #Little change adding three outputs
            A4_1 = Dense(1,activation="relu")(A3)
            A4_2 = Dense(1,activation="relu")(A3)
            #A4_3 = Dense(1,activation="bounded_output_KBr")(A3)

            out = Concatenate()([A4_1,A4_2])
            A4 = Dense(2,name='A4')(out)

            #B1 = Input(shape=(1,),name='B1')
            B1 = Dense(2, activation='relu',name='B1')(A4)
            B2 = Dense(16, activation='relu',name='B2')(B1)
            B3 = Dense(32, activation='relu',name='B3')(B2)

            B4_1 = Dense(1,activation="bounded_output_Cit")(B3)
            B4_2 = Dense(1,activation="bounded_output_Ag")(B3)
            B4_3 = Dense(1,activation="bounded_output_KBr")(B3)

            outB = Concatenate()([B4_1,B4_2,B4_3])
            B4 = Dense(3,name='B4')(outB)

            ##MODEL
            ML0 = Model(inputs=[A1], outputs=[A4])
            plot_model(ML0,to_file='ML0.png',show_shapes=True)

            ML2 = Model(inputs=[A1], outputs=[B4])
            plot_model(ML2,to_file='ML2.png',show_shapes=True)

            ##COMPILE

            ML0.compile(
                optimizer='adam',
                loss='mean_squared_error',
                metrics=['mse','mae'])
            #ML1.compile(
            #    optimizer='adam',
            #    loss='mean_squared_error',
            #    metrics=['mse','mae'])
            ML2.compile(
                optimizer='RMSprop',
                loss='mean_squared_error',
                metrics=['mse','mae'])



            ##HISTORY


            logdir0= "logs0/fit/" + datetime.now().strftime("%Y/%m/%d;%H:%M:%S")
            tensorboard_callback0 = keras.callbacks.TensorBoard(log_dir=logdir0)

            #logdir1= "logs1/fit/" + datetime.now().strftime("%Y/%m/%d;%H:%M:%S")
            #tensorboard_callback1 = keras.callbacks.TensorBoard(log_dir=logdir1)

            logdir2= "logs2/fit/" + datetime.now().strftime("%Y/%m/%d;%H:%M:%S")
            tensorboard_callback2 = keras.callbacks.TensorBoard(log_dir=logdir2)

            logdirNew= "logsNew/fit/" + datetime.now().strftime("%Y/%m/%d;%H:%M:%S")
            tensorboard_callbackNew = keras.callbacks.TensorBoard(log_dir=logdirNew)

            logdirNew_2= "logsNew_2/fit/" + datetime.now().strftime("%Y/%m/%d;%H:%M:%S")
            tensorboard_callbackNew_2 = keras.callbacks.TensorBoard(log_dir=logdirNew_2)

            logdir11= "logs11/fit/" + datetime.now().strftime("%Y/%m/%d;%H:%M:%S")
            tensorboard_callback11 = keras.callbacks.TensorBoard(log_dir=logdir11)

            logdir11_2= "logs11_2/fit/" + datetime.now().strftime("%Y/%m/%d;%H:%M:%S")
            tensorboard_callback11_2 = keras.callbacks.TensorBoard(log_dir=logdir11_2)

            historyML0= ML0.fit(
                Train_label,
                Train_input, 
                batch_size=10,
                epochs=200, 
                callbacks=[tensorboard_callback0],
                validation_data=(Val_label, Val_input),
                verbose = 0,
            )

            #historyML1= ML1.fit(
                
            #    Train_input,
            #    Train_label,
            #    batch_size=2,
            #    epochs=200, 
            #    callbacks=[tensorboard_callback1],
            #    validation_data=(Val_input, Val_label),
            #)

            historyML2= ML2.fit(
                Train_label,
                Train_label, 
                batch_size=2,
                epochs=200, 
                callbacks=[tensorboard_callback2],
                validation_data=(Val_label, Val_label),

                verbose = 0,

            )
    

            model_partion = ML2  # include here your original model

            layer_name_input = "A4"
            layer_name_output = "B4"
            ML_new_2 = Model(inputs=model_partion.get_layer(layer_name_input).input,
                                            outputs=model_partion.get_layer(layer_name_output).output)

            plot_model(ML_new_2,to_file='New_model.png',show_shapes=True)


            ML_new_2.compile(
                optimizer='adam',
                loss='mean_squared_error',
                metrics=['mse','mae'])
            
            historyML_new_2= ML_new_2.fit(
                
                Train_input,
                Train_label,
                batch_size=2,
                epochs=200, 
                validation_data=(Val_input, Val_label),
                callbacks=[tensorboard_callbackNew_2],
                verbose = 0,
            )


            print("ML_new_2")
            resultsMl_new_2 = ML_new_2.evaluate(Test_input, Test_label, batch_size=5)
            print("%s: %.2f%%" % (ML_new_2.metrics_names[1], resultsMl_new_2[1]*100))
            print("test loss, test acc:", resultsMl_new_2)
            predicted_by_model= ML_new_2.predict(input_user)
            return input_user, predicted_by_model  #, 0 , 0, 0 , ml_past, Ngen_net, mod

        
        else:
            print("Which pass?",r)
            #Data
            train_size= math.floor(len(df)*(80/100))
            val_size= int((len(df)-train_size)/2)
            test_size= int(len(df)-train_size-val_size)

            train_pre =[]
            for rto in range(train_size):
                train_pre.append((df["Wavelength"][:train_size][rto],df["Observance"][:train_size][rto]))
        
            val_pre =[]
            for ret in range(train_size,train_size+val_size):
                val_pre.append((df["Wavelength"][train_size:train_size+val_size][ret],df["Observance"][train_size:train_size+val_size][ret]))
        
            test_pre =[]
            for retr in range(train_size+val_size,train_size+val_size+test_size):
                test_pre.append((df["Wavelength"][train_size+val_size:train_size+val_size+test_size][retr],df["Observance"][train_size+val_size:train_size+val_size+test_size][retr]))

            train_label_pre =[]
            for ee in range(train_size):
                train_label_pre.append([df["[Cit]"][:train_size][ee],df["[Ag]"][:train_size][ee],df["[KBr]"][:train_size][ee]])

            val_label_pre =[]
            for eer in range(train_size,train_size+val_size):
                val_label_pre.append([df["[Cit]"][train_size:train_size+val_size][eer],df["[Ag]"][train_size:train_size+val_size][eer],df["[KBr]"][train_size:train_size+val_size][eer]])

            test_label_pre =[]
            for eerr in range(train_size+val_size,train_size+val_size+test_size):
                test_label_pre.append([df["[Cit]"][train_size+val_size:train_size+val_size+test_size][eerr],df["[Ag]"][train_size+val_size:train_size+val_size+test_size][eerr],df["[KBr]"][train_size+val_size:train_size+val_size+test_size][eerr]])



            Train_input = np.array(train_pre)
            Train_label = np.array(train_label_pre)

            Val_input   = np.array(val_pre)
            Val_label   = np.array(val_label_pre)
            
            Test_input  = np.array(test_pre)
            Test_label  = np.array(test_label_pre)


            

            upper_bound_Cit = 8.125
            lower_bound_Cit = 0.3125
            
            #Boundary for Ag
            
            upper_bound_Ag = 0.24375
            lower_bound_Ag = 0.0094
            
            #Boundary for KBr
            upper_bound_KBr = 0.0005
            lower_bound_KBr = 0.00025

            def bounded_output_Cit(x):
                scale = upper_bound_Cit - lower_bound_Cit
                return scale * tf.nn.sigmoid(x) + lower_bound_Cit

            def bounded_output_Ag(x):
                scale = upper_bound_Ag - lower_bound_Ag
                return scale * tf.nn.sigmoid(x) + lower_bound_Ag

            def bounded_output_KBr(x):
                scale = upper_bound_KBr - lower_bound_KBr
                return scale * tf.nn.sigmoid(x) + lower_bound_KBr
            

            from keras.utils.generic_utils import get_custom_objects

            get_custom_objects().update({'bounded_output_Cit': Activation(bounded_output_Cit)})
            get_custom_objects().update({'bounded_output_Ag': Activation(bounded_output_Ag)})
            get_custom_objects().update({'bounded_output_KBr': Activation(bounded_output_KBr)})

            #Model
            A1 = Input(shape=(3,),name='A1')
            A2 = Dense(32, activation='relu',name='A2')(A1)
            A3 = Dense(16, activation='relu',name='A3')(A2)
            #Little change adding three outputs
            A4_1 = Dense(1,activation="relu")(A3)
            A4_2 = Dense(1,activation="relu")(A3)
            #A4_3 = Dense(1,activation="bounded_output_KBr")(A3)

            out = Concatenate()([A4_1,A4_2])
            A4 = Dense(2,name='A4')(out)

            #B1 = Input(shape=(1,),name='B1')
            B1 = Dense(2, activation='relu',name='B1')(A4)
            B2 = Dense(16, activation='relu',name='B2')(B1)
            B3 = Dense(32, activation='relu',name='B3')(B2)

            B4_1 = Dense(1,activation="bounded_output_Cit")(B3)
            B4_2 = Dense(1,activation="bounded_output_Ag")(B3)
            B4_3 = Dense(1,activation="bounded_output_KBr")(B3)

            outB = Concatenate()([B4_1,B4_2,B4_3])
            B4 = Dense(3,name='B4')(outB)

            ##MODEL
            ML0 = Model(inputs=[A1], outputs=[A4])
            plot_model(ML0,to_file='ML0.png',show_shapes=True)

            ML2 = Model(inputs=[A1], outputs=[B4])
            plot_model(ML2,to_file='ML2.png',show_shapes=True)

            ##COMPILE

            ML0.compile(
                optimizer='adam',
                loss='mean_squared_error',
                metrics=['mse','mae'])
            #ML1.compile(
            #    optimizer='adam',
            #    loss='mean_squared_error',
            #    metrics=['mse','mae'])
            ML2.compile(
                optimizer='RMSprop',
                loss='mean_squared_error',
                metrics=['mse','mae'])



            ##HISTORY


            logdir0= "logs0/fit/" + datetime.now().strftime("%Y/%m/%d;%H:%M:%S")
            tensorboard_callback0 = keras.callbacks.TensorBoard(log_dir=logdir0)

            #logdir1= "logs1/fit/" + datetime.now().strftime("%Y/%m/%d;%H:%M:%S")
            #tensorboard_callback1 = keras.callbacks.TensorBoard(log_dir=logdir1)

            logdir2= "logs2/fit/" + datetime.now().strftime("%Y/%m/%d;%H:%M:%S")
            tensorboard_callback2 = keras.callbacks.TensorBoard(log_dir=logdir2)

            logdirNew= "logsNew/fit/" + datetime.now().strftime("%Y/%m/%d;%H:%M:%S")
            tensorboard_callbackNew = keras.callbacks.TensorBoard(log_dir=logdirNew)

            logdirNew_2= "logsNew_2/fit/" + datetime.now().strftime("%Y/%m/%d;%H:%M:%S")
            tensorboard_callbackNew_2 = keras.callbacks.TensorBoard(log_dir=logdirNew_2)

            logdir11= "logs11/fit/" + datetime.now().strftime("%Y/%m/%d;%H:%M:%S")
            tensorboard_callback11 = keras.callbacks.TensorBoard(log_dir=logdir11)

            logdir11_2= "logs11_2/fit/" + datetime.now().strftime("%Y/%m/%d;%H:%M:%S")
            tensorboard_callback11_2 = keras.callbacks.TensorBoard(log_dir=logdir11_2)

            historyML0= ML0.fit(
                Train_label,
                Train_input, 
                batch_size=10,
                epochs=200, 
                callbacks=[tensorboard_callback0],
                validation_data=(Val_label, Val_input),
                verbose = 0,
            )

            #historyML1= ML1.fit(
                
            #    Train_input,
            #    Train_label,
            #    batch_size=2,
            #    epochs=200, 
            #    callbacks=[tensorboard_callback1],
            #    validation_data=(Val_input, Val_label),
            #)

            historyML2= ML2.fit(
                Train_label,
                Train_label, 
                batch_size=2,
                epochs=200, 
                callbacks=[tensorboard_callback2],
                validation_data=(Val_label, Val_label),

                verbose = 0,

            )
            

            #Pre-p
            # Evaluate the model on the test data using `evaluate`
            print("Evaluate on test data")
            results0 = ML0.evaluate(Test_label, Test_input, batch_size=5)
            print("test loss, test acc:", results0)

            # Generate predictions (probabilities -- the output of the last layer)
            # on new data using `predict`
            print("Generate predictions for 3 samples")
            #predictions = model2.predict(np.array([Test_input[1]]))#Test_input)
            predictions0 = ML0.predict(Train_label)#Test_input)

            print("predictions shape:", predictions0.shape)
            #predictions0

            # Evaluate the model on the test data using `evaluate`
            print("Evaluate on test data")
            results2 = ML2.evaluate(Test_label, Test_label, batch_size=5)
            print("test loss, test acc:", results2)

            # Generate predictions (probabilities -- the output of the last layer)
            # on new data using `predict`
            print("Generate predictions for 3 samples")
            #predictions = model2.predict(np.array([Test_input[1]]))#Test_input)
            predictions2 = ML2.predict(Train_label)#Test_input)

            print("predictions shape:", predictions2.shape)
            # predictions2

            ##NEW MODELS

            model_partion = ML2  # include here your original model

            layer_name_input = "A4"
            layer_name_output = "B4"
            ML_new = Model(inputs=model_partion.get_layer(layer_name_input).input,
                                            outputs=model_partion.get_layer(layer_name_output).output)

            plot_model(ML_new,to_file='New_model.png',show_shapes=True)

            ML11 = Model(inputs=[A4], outputs=[B4])
            print("weigths Model 11/no trained", ML11.layers[1].get_weights())
            print("weigths Model NEW/no trained",ML_new.layers[1].get_weights())

            ML_new_2 = ML_new
            ML11_2 = ML11
            print("weigths Model 11_2/no trained",ML11_2.layers[1].get_weights())
            print("weigths Model NEW_2 /no trained",ML_new_2.layers[1].get_weights())



            ML11.compile(
                optimizer='adam',
                loss='mean_squared_error',
                metrics=['mse','mae'])
            ML_new.compile(
                optimizer='adam',
                loss='mean_squared_error',
                metrics=['mse','mae'])
            ML11_2.compile(
                optimizer='adam',
                loss='mean_squared_error',
                metrics=['mse','mae'])
            ML_new_2.compile(
                optimizer='adam',
                loss='mean_squared_error',
                metrics=['mse','mae'])

            historyML11= ML11.fit(
    
                predictions0,
                Train_label,
                batch_size=2,
                epochs=200, 
                validation_data=(Val_input, Val_label),
                callbacks=[tensorboard_callback11],

                verbose = 0,

            )
            historyML11_2= ML11_2.fit(
                
                Train_input,
                Train_label,
                batch_size=2,
                epochs=200, 
                validation_data=(Val_input, Val_label),
                callbacks=[tensorboard_callback11_2],

                verbose = 0,

            )
            historyML_new= ML_new.fit(
                
                predictions0,
                Train_label,
                batch_size=2,
                epochs=200, 
                validation_data=(Val_input, Val_label),
                callbacks=[tensorboard_callbackNew],
                verbose = 0,

            )
            
            historyML_new_2= ML_new_2.fit(
                
                Train_input,
                Train_label,
                batch_size=2,
                epochs=200, 
                validation_data=(Val_input, Val_label),
                callbacks=[tensorboard_callbackNew_2],
                verbose = 0,
            )






            print("weigths Model 11/ Trained", ML11.layers[1].get_weights())
            print("weigths Model NEW/ Trained",ML_new.layers[1].get_weights())
            print("weigths Model 11_2/ Trained",ML11_2.layers[1].get_weights())
            print("weigths Model NEW_2 / Trained",ML_new_2.layers[1].get_weights())

            print("weigths Model TOTAL / Trained",ML2.layers[6].get_weights())

            ##Prediction
            print("Predicting-----")
            print("Predict: prediction0")
            ML_new_pr = ML_new.predict(predictions0)
            print("ML_new_pr",ML_new_pr)

            ML_new_2_pr= ML_new_2.predict(predictions0)
            print("ML_new_2_pr",ML_new_2_pr)

            ML11_pr = ML11.predict(predictions0 )#Test_input)
            print("ML11_pr",ML11_pr)

            ML11_2_pr=ML11_2.predict(predictions0)
            print("ML11_2_pr",ML11_2_pr)


            print("ML_TOTAL",predictions2)

            print("Errors")
            print(np.sum(ML_new_pr-Train_label))
            print(np.sum(ML_new_2_pr-Train_label))
            print(np.sum(ML11_pr-Train_label))
            print(np.sum(ML11_2_pr-Train_label))
            print(np.sum(predictions2-Train_label))


            print("Predict: Train")
            ML_new_tr = ML_new.predict(Train_input)
            print("ML_new_pr",ML_new_tr)

            ML_new_2_tr= ML_new_2.predict(Train_input)
            print("ML_new_2_pr",ML_new_2_tr)

            ML11_tr = ML11.predict(Train_input )#Test_input)
            print("ML11_pr",ML11_tr)

            ML11_2_tr=ML11_2.predict(Train_input)
            print("ML11_2_pr",ML11_2_tr)


            print("ML_TOTAL",predictions2)

            print("Errors")
            ML_new_error = np.sum(ML_new_tr-Train_label)
            ML_new_2_error = np.sum(ML_new_2_tr-Train_label)
            ML11_error = np.sum(ML11_tr-Train_label)
            M11_2_error= np.sum(ML11_2_tr-Train_label)
            print("ML_new",ML_new_error)
            print("ML_new_2",ML_new_2_error)
            print("ML11",ML11_error)
            print("ML11_2",M11_2_error)
            print("predictions2",np.sum(predictions2-Train_label))

            print("Scores")
            # print("Evaluating on test data")
            print("ML_new")
            resultsMl_new = ML_new.evaluate(Test_input, Test_label, batch_size=5)
            print("%s: %.2f%%" % (ML_new.metrics_names[1], resultsMl_new[1]*100))
            print("test loss, test acc:", resultsMl_new)

            print("ML_new_2")
            resultsMl_new_2 = ML_new_2.evaluate(Test_input, Test_label, batch_size=5)
            print("%s: %.2f%%" % (ML_new_2.metrics_names[1], resultsMl_new_2[1]*100))
            print("test loss, test acc:", resultsMl_new_2)

            print("ML11")
            resultsMl11 = ML11.evaluate(Test_input, Test_label, batch_size=5)
            print("%s: %.2f%%" % (ML11.metrics_names[1], resultsMl11[1]*100))
            print("test loss, test acc:", resultsMl11)

            print("ML11_2")
            resultsMl11_2 = ML11_2.evaluate(Test_input, Test_label, batch_size=5)
            print("%s: %.2f%%" % (ML11_2.metrics_names[1], resultsMl11_2[1]*100))
            print("test loss, test acc:", resultsMl11_2)


            ml = {"ML_new", 
                "ML_new_2",
                "ML11",
                "ML11_2"
                }

            ml_errors = {ML_new_error:"ML_new", 
                ML_new_2_error:"ML_new_2",
                ML11_error:"ML11",
                M11_2_error:"ML11_2"
                }
            lest_error = [ML_new_error,ML_new_2_error,ML11_error,M11_2_error]
            ml_to_use= ml_errors[min(lest_error)]
            
            if ml_past == False:
                    choosg= input("Do you want to choose a model? [Yes-y/No]")
                    if choosg == "Yes" or choosg=="y":
                        while True:

                            choosg_ml = input("Choose ML_new/ ML_new_2/ ML11/ ML11_2")
                            if choosg_ml == "ML_new":
                                predicted_by_model = ML_new.predict(input_user)
                                break
                            elif choosg_ml == "ML_new_2":
                                predicted_by_model = ML_new_2.predict(input_user)
                                break
                            elif choosg_ml == "ML11":
                                predicted_by_model = ML11.predict(input_user)
                                break
                            elif choosg_ml == "ML11_2":
                                predicted_by_model = ML11_2.predict(input_user)
                                break

                            else:
                                print("Type again")
                                # choosg_ml = input("Choose ML_new/ ML_new_2/ ML11/ ML11_2")


                    elif choosg == "No":

                        print("You choose automatic")

                        if ml_to_use == "ML_new":
                                predicted_by_model = ML_new.predict(input_user)
                                
                        elif ml_to_use == "ML_new_2":
                                predicted_by_model = ML_new_2.predict(input_user)
                                
                        elif ml_to_use == "ML11":
                                predicted_by_model = ML11.predict(input_user)
                                
                        elif ml_to_use == "ML11_2":
                                predicted_by_model = ML11_2.predict(input_user)
                                

                    else:
                        print("No input recorded")
                        if ml_to_use == "ML_new":
                                predicted_by_model = ML_new.predict(input_user)
                                
                        elif ml_to_use == "ML_new_2":
                                predicted_by_model = ML_new_2.predict(input_user)
                                
                        elif ml_to_use == "ML11":
                                predicted_by_model = ML11.predict(input_user)
                                
                        elif ml_to_use == "ML11_2":
                                predicted_by_model = ML11_2.predict(input_user)

            else:
                if ml_to_use == "ML_new":
                        predicted_by_model = ML_new.predict(input_user)
                                
                elif ml_to_use == "ML_new_2":
                        predicted_by_model = ML_new_2.predict(input_user)
                                
                elif ml_to_use == "ML11":
                        predicted_by_model = ML11.predict(input_user)
                                
                elif ml_to_use == "ML11_2":
                        predicted_by_model = ML11_2.predict(input_user)
            ml_past = True
            # from tensorboard import program

            # tracking_address = logdir # the path of your log file.

            #if __name__ == "__main__":
            # tb = program.TensorBoard()
            # tb1= program.TensorBoard()
            # tb2= program.TensorBoard()
            # tb3= program.TensorBoard()
            # tb4= program.TensorBoard()
            # tb5= program.TensorBoard()

            # tb.configure(argv=[None, '--logdir', logdir0])
            # tb1.configure(argv=[None, '--logdir', logdir2])
            # tb2.configure(argv=[None, '--logdir', logdirNew])
            # tb3.configure(argv=[None, '--logdir', logdirNew_2])
            # tb4.configure(argv=[None, '--logdir', logdir11])
            # tb5.configure(argv=[None, '--logdir', logdir11_2])

            # url = tb.launch()
            # url1 = tb1.launch()
            # url2 = tb2.launch()
            # url3 = tb3.launch()
            # url4 = tb4.launch()
            # url5 = tb5.launch()

            # print("ML0")
            # print(f"Tensorflow listening on {url}")
            # print("")
            # print("")

            # print("-------")
            # print("ML2")
            # print(f"Tensorflow listening on {url1}")
            # print("")

            # print("-------")
            # print("ML_New")
            # print(f"Tensorflow listening on {url2}")
            # print("")

            # print("-------")
            # print("ML_New_2")
            # print(f"Tensorflow listening on {url3}")
            # print("")

            # print("-------")
            # print("ML_11")
            # print(f"Tensorflow listening on {url4}")
            # print("")

            # print("-------")
            # print("ML_11_2")
            # print(f"Tensorflow listening on {url5}")

            upper_bound_Cit = 8.125
            lower_bound_Cit = 0.3125
                    
            #Boundary for Ag
                    
            upper_bound_Ag = 0.24375
            lower_bound_Ag = 0.0094
                    
            #Boundary for KBr
            upper_bound_KBr = 0.0005
            lower_bound_KBr = 0.00025
            out2 = predicted_by_model[0][2]
            out0 = predicted_by_model[0][0]
            out1 = predicted_by_model[0][1]

            Nextt_net= False
            Ngen_net=0
            while True:
                print("Checking creations model",(out2*(200)/0.005 + out0*(200)/ 12.5+ out1*(200)/0.375))
                if (out2*(200)/0.005 + out0*(200)/ 12.5+ out1*(200)/0.375) > 130:
                    out2 = out2 - 4*(out2/100)
                    out0 = out0 - 4*(out0/100)
                    out1 = out1 - 4*(out1/100)
                    Nextt_net = True
                    Ngen_net +=1
                else:

                    if out2 == upper_bound_KBr:
                        out1 = 0
                        out0  = 0

                    if out0 == upper_bound_Cit:
                        out2 = 0
                        out1  = 0

                    if out1 == upper_bound_Ag:
                        out0 = 0
                        out2  = 0
                    Nextt_net= True
                    break
                

                            
            print("N of rep",Ngen_net)
            print("Concen after checking (T)",out0,out1,out2)

            predicted_by_model[0][0] =out0
            predicted_by_model[0][1] =out1
            predicted_by_model[0][2] =out2
            print("")       
            print("predicted_by_model",predicted_by_model)
            mod=ML_new_2
            # time.sleep(50)
            return input_user, predicted_by_model, 0 , 0, 0 , ml_past, Ngen_net, mod





            # history= modelN.fit(
            #     Train_input,
            #     Train_label, 
            #     batch_size=2,
            #     epochs=n_epochs, 
            #     callbacks=[tensorboard_callback],
            #     validation_data=(Val_input, Val_label),
            # )
            


            # # history.history
            # print("Evaluating on test data")
            # results = modelN.evaluate(Test_input, Test_label, batch_size=5)
            # print("%s: %.2f%%" % (modelN.metrics_names[1], results[1]*100))
            # print("test loss, test acc:", results)

            # # save model and architecture to single file
            # modelN.save("modelN.h5")
            # print("Saved modelN to disk")        
            
            #Tensorboard


            # def launchTensorBoard():
            #     #os.system('reload_ext tensorboard')
            #     os.system('tensorboard --logdir=' + logdir)
            #     return

            # import threading
            # t = threading.Thread(target=launchTensorBoard, args=([]))
            # t.start()

            
            
            
            # print("b predict",Test_input)

            #print(",",modelCall["cacheErrorAvg"])
            #plots_error_avg(modelCall)
            # predictQuestion = input("Do you want to make a prediction: [Yes-y / No ]")
            
            # if predictQuestion == "Yes" or predictQuestion=="y":
            #     prediction_user_input = input("Please enter W-O:")

            print("Generate predictions for 3 samples")
            # y_pred = modelN.predict(np.array(Test_input))#Test_input)
            # y_pred = modelN.predict(Test_input)
            print("Input_us",input_user)
            y_pred = modelN.predict(input_user)
            print("predictions shape:", y_pred.shape)
            print("predictions:",y_pred)
            return input_user, y_pred, 0 , 0, 0

        
                # #prediction = predictLinearModel(predict,modelCall["ParamsToUse"]["Theta"], modelCall["ParamsToUse"]["Bias"])
                # prediction = predictLinearModelInverse(predict,modelCall["ParamsToUse"]["Theta"], modelCall["ParamsToUse"]["Bias"])
                # print("Predicted concentration [] given a wavelenght", prediction)
                # #ADDING DELETE IF NO PROB
                # breakpoint()

                # return {"inputPredictor":predict, "prediction":prediction , "par_theta":modelCall["ParamsToUse"]["Theta"], "par_bias":modelCall["ParamsToUse"]["Bias"] }
            
            # else:
                
            #     #ADDING DELETE IF NO PROB
            #     sys.exit()
            #     breakpoint()
                
            #     
            return 0, 0, 0 , 0, 0
        
        

    
    
    def Oldtraining(self, df,input_user,r):


            def getting_params(self, concentration,wavelength):
                print("Parasn", concentration)
                W = sum(wavelength*(concentration-np.mean(concentration))) / sum((concentration-np.mean(concentration))**2)
                b = np.mean(wavelength) - W*np.mean(concentration)
                print("--->", W,b)
                return W, b
            
            #Training //Computing the parameters
            print("---->Model training")
            print("Computing W and b")

            W, b = getting_params(self,df['Concentration'],df['Wavelength']) 

            #making predictions based on our current data to see plot and the error
            print("making predictions based on our current data to see plot and the error")
            train_prediction= df['Concentration'] * W + b
            #our error 
            training_error= df['Wavelength']-train_prediction
            print("----")
            print("Train_prediction:",train_prediction)
            print("----")
            print("Error", training_error)
            #ploting
            train_fig = plt.figure(num=None, figsize=(4, 4),dpi=300, facecolor='w', edgecolor='k') 
            plt.plot(df['Concentration'], train_prediction, color='red',label="Predicted Wavelength by Linear Model")
            plt.scatter(df['Concentration'], df['Wavelength'], label="Training Data")
            plt.xlabel("[KBr] Concentration (mM)")
            plt.ylabel("Wavelength (nm)")
            plt.legend(prop={"size":6})
            plt.show()
            train_fig.savefig("training-"+str(r)+"png",dpi=train_fig.dpi)
            #User input for Wavelength wanted


            #conputing the inverse --> from Wavelength to Concentration
            input_user= input_user
            while True:
                user_concentration = (input_user- b) / W
                print("Model predictied concentration given the wavelength: ",user_concentration)
                if user_concentration < 0.00025 or user_concentration >0.003:
                    print("Sorry, the wanted wavelength is not reached given the current concentration of KBr allowed to process")
                    input_user= input("Please ENTER the desire Wavelength: ")
                    input_user= float(input_user)
                    
                    #user_concentration = (input_user- b) / W

                else:

                    break


            print("------------")
            print("Making prediction")
            print("passing ", user_concentration, " to robot")
            return input_user, user_concentration, train_prediction , W, b
    

    ##OLD TRAIN
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

        #NOTE we may get fancier in the future here and do more preprocessing
        # if self.scan_bounds:
        #     #if you only want to pay attention in bounds, train on those vals
        #     processedX = X[:, self.scan_bounds[0]:self.scan_bounds[1]]
        # else:
        #     processedX = X
        #update the data with the new scans
        time.sleep(40)
        # print('<<ML>> training')
        with self.model_lock:
        #     if isinstance(self.X,np.ndarray):
        #         self.X = np.concatenate((self.X, processedX))
        #         self.y = np.concatenate((self.y, y))
        #     else:
        #         self.X = processedX
        #         self.y = y
        #     print("model fitting on X", self.X)
        #     print("model fitting on y", self.y)
        #     self.model.fit(self.X, self.y)
        # print('<<ML>> done training'
            print("X and Y")
            print(X,y) 
            print("--------")
            print('<<ML>> training')

            # ml_model_trained = MainModel(self,10,3,0.05,X,y)

            print('<<ML>> done training')

            return 0 #MainModel(self, X,y)



#####################



















































##########################################################################
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
        super().__init__(model, max_iters) #don't have a model
        self.scan_bounds = scan_bounds
        if scan_bounds:
            #if you only want to pay attention in bounds, predict on those vals
            self.FINAL_SPECTRA = final_spectra[:, scan_bounds[0]:scan_bounds[1]]
        else:
            self.FINAL_SPECTRA = final_spectra
        self.y_shape = y_shape
        self.batch_size = batch_size
        self.duplication = duplication

    def generate_seed_rxns(self):
        '''
        This method is called before the model is trained to generate a batch of training
        points  
        returns:  
            np.array: (batch_size,n_features) 
        '''
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
        return np.repeat(y_pred, self.duplication, axis=0);
 
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
        #NOTE we may get fancier in the future here and do more preprocessing
        if self.scan_bounds:
            #if you only want to pay attention in bounds, train on those vals
            processedX = X[:, self.scan_bounds[0]:self.scan_bounds[1]]
        else:
            processedX = X
        #update the data with the new scans
        time.sleep(40)
        print('<<ML>> training')
        with self.model_lock:
            if isinstance(self.X,np.ndarray):
                self.X = np.concatenate((self.X, processedX))
                self.y = np.concatenate((self.y, y))
            else:
                self.X = processedX
                self.y = y
            print("model fitting on X", self.X)
            print("model fitting on y", self.y)
            self.model.fit(self.X, self.y)
        print('<<ML>> done training')
