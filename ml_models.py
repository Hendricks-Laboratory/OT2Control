import time #TODO delete this debugging only
from abc import ABC
from abc import abstractmethod
import threading
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt
import  pandas as pd
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
        y_shape is the number of reagents you're guessing at in recipes
    '''
    def __init__(self, model, max_iters=np.inf):
        self.curr_iter = 0
        self.max_iters = max_iters
        self.quit = False
        self.model = model
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
        pass
        # train_thread = [thread for thread in threading.enumerate() \
        #                     if thread.name == 'train thread']
        # if train_thread:
        #     train_thread = train_thread[0]
        #     train_thread.join() #wait till you're done training

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
        pass
    @abstractmethod
    def training(self):
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
        upper_bound = 0.003
        lower_bound = 0.00025
        # recipes = np.random.rand(self.batch_size, self.y_shape) \
        #         * (upper_bound - lower_bound) + lower_bound
	    # print("seed,", recipes)
        #recipes = np.repeat(recipes, self.duplication, axis=0)
        recipes =  np.random.rand(1,number_recipes) * (upper_bound - lower_bound) + lower_bound 
        #recipes =  np.random.rand(1,1) * (2.5 - 0.2) + 0.2
        print("our recipes", recipes)
        #recipes =  np.random.rand(1,3) * (2.5 - 0.2) + 0.2 

        return recipes
    # def plots_error_avg(model):
    #     print(len(model["cacheErrorAvg"]))
    #     min_index=np.min(model["cacheErrorAvg"])
    #     max_index=np.max(model["cacheErrorAvg"])
    #     print("NNNNNNNNN",model["cacheErrorAvg"])
    #     print(min_index)
    #     errorsSca=[]
    #     for i in range(len(model["cacheErrorAvg"])):
    #         errorsSca.append((model["cacheErrorAvg"][i]-min_index)/(max_index- min_index))
    #     print("AAAAAAAAA",errorsSca)
    #     plt.plot([i for i in range(1,model["break_epoch"]+1)],errorsSca)
    #     plt.savefig('pic.png')

    # def predict(self):
    #     '''
    #     This call should wait on the training thread to complete if it is has not been collected
    #     yet.  
    #     params:  
    #         int n_predictions: the number of instances to predict  
    #     returns:  
    #         np.array: shape is n_predictions, y.shape. Features are pi e-2  
    #     '''
    #     super().predict()
    #     with self.model_lock:
    #         y_pred = self.model.predict(self.FINAL_SPECTRA)
    #     print("predicted", y_pred)
    #     breakpoint()
    #     return np.repeat(y_pred, self.duplication, axis=0);

    # def predict(N,recipes,learning_rate,X=None):
    # def predictLinearModel(self, X, W, b):
    # #Overcome Overflow dtype=np.uint32
    #     X= np.array([X],dtype=float)
    #     W= np.array([W],dtype=float)
    #     b= np.array([b],dtype=float)
    #     print(X)
    #     print(W)
    #     print(b)
        
    #     Y_hat = X * W + b
    #     return Y_hat
    def prediction(self, modelCall):
        #modelCall= LinearModel(N,recipes,learning_rate,X)
        #super().predict()
        # with self.model_lock:
        #     y_pred = self.model.predict(self.FINAL_SPECTRA)
        # print("predicted", y_pred)
        # breakpoint()
        #return np.repeat(y_pred, self.duplication, axis=0);
        def predictLinearModel(X, W, b):
            #Overcome Overflow dtype=np.uint32
            X= np.array([[X]],dtype=float)
            W= np.array([W],dtype=float)
            b= np.array([b],dtype=float)
            print("predicting",X)
            print("predictor W",W)
            print("predictor W",b)
            
            Y_hat = X * W + b
            return Y_hat

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
            return {"inputPredictor":predict, "prediction":prediction , "par_theta":modelCall["ParamsToUse"]["Theta"], "par_bias":modelCall["ParamsToUse"]["Bias"] }
        else:
            return {"prediction":0, "par_theta":modelCall["ParamsToUse"]["Theta"], "par_bias":modelCall["ParamsToUse"]["Bias"] }
 
    # def mainModel(self,N,recipes,learning_rate,X=None,Y=None):
    
    #     cacheError=[]
    #     cacheErrorAvg=[]
    #     cacheTheta=[]
    #     cacheBias =[]
    #     cacheParamsToUse={}
    #     breakEpoch=0
    #     Y_values=[]
        
        
    #     for i in range(N):
            
    #         ##
            
    #         if i !=0:
    #             theta = cacheTheta[i-1]-learning_rate*cacheError[i-1]/(recipes)
    #             bias = cacheBias[i-1]-learning_rate*cacheBias[i-1]/(recipes)
    #             print(theta.shape,"NNN")
    #             print(bias.shape,"NNN")
                
    #         else:
    #             #Generating random parameters at the beginning
    #             #theta = np.random.rand(recipes,1)
    #             #bias  = np.random.rand(recipes,1)
    #             #bias  = random.uniform(0,1)
    #             X_mean=np.mean(X)
    #             theta= Y*(X-X_mean) / (X-X_mean)**2
    #             bias= np.mean(Y) - theta*np.mean(X)

            
            
    #         ##
    #         #Linear Model
            
    #         #X_mean=np.mean(X)
    #         #theta= Y*(X-X_mean) / (X-X_mean)**2
    #         print("Bias", bias)
    #         print("Theta.shape", theta.shape)
    #         print("Bias.shape", bias.shape)
    #         print("X Shape", X.shape)


    #         Y_hat= theta@X.T + bias
    #         print("Y_hat",Y_hat)
            
    #         ##SEND X
    #         ##back y
            
    #         #Computer error
            
    #         #Simple error
    #         #error= Y_hat-Y.T
    #         #
    #         print("Dii", np.abs(Y_hat-Y))#np.sqrt(np.sum((Y_hat-Y.T)**2,axis=1)))
    #         error= np.abs(Y_hat-Y)# np.sum(np.abs(Y_hat-Y.T),axis=1)#np.sqrt(np.sum((Y_hat-Y.T)**2,axis=1))  #Y_hat-Y.T#np.sum((Y_hat-Y.T).T@(Y_hat-Y.T))
            
            
            
            
    #         print("Y",Y)
    #         print("-----------------------")
    #         print("Error avg",np.sum(error)/np.sqrt(recipes))
    #         print("Error in epoch #"+str(i+1))
            
    #         print(error)
    #         print()
    #         print()
        
            
            
    #         #Saving in cache error and theta
            
    #         cacheErrorAvg.append(np.sum(error)/(2*recipes))
    #         cacheError.append(error)
    #         cacheTheta.append(theta)
    #         cacheBias.append(bias)
            
            
    #         #Callback
            
    #         #if error keeps increasing we stop and return the previous last parameter
    #         #after three epochs 
            
    #         if i > 2:
    #             if np.sum(error)/(2*recipes) < np.sum(cacheError[i-2])/(2*recipes) + 5:
    #                 breakEpoch= i
    #                 print("n",i)
    #                 print("Last param",np.sum(cacheTheta[i-2])/(2*recipes))
    #                 print("Last param",np.sum(cacheBias[i-2])/(2*recipes))
    #                 cacheParamsToUse["Theta"]= np.sum(cacheTheta[i-2]/(2*recipes))
    #                 cacheParamsToUse["Bias"] = np.sum(cacheBias[i-2]/(2*recipes))
    #                 break
    #             else:
    #                 pass
    #         if i==N-1:
    #             breakEpoch= i
    #             cacheParamsToUse["Theta"]= np.sum(cacheTheta[i-2]/(2*recipes))
    #             cacheParamsToUse["Bias"] = np.sum(cacheBias[i-2]/(2*recipes))
            
    #     return {"receipt":X, "cacheErrorAvg":cacheErrorAvg, "cacheTheta":cacheTheta,"cacheBias":cacheBias, "break_epoch":breakEpoch+1, "ParamsToUse":cacheParamsToUse}


    

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
            train_fig = plt.figure(figsize=(5,7)) 
            plt.plot(df['Concentration'], train_prediction, color='red',label="Predicted Wavelength")
            plt.scatter(df['Concentration'], df['Wavelength'], label="Training Data")
            plt.xlabel("Concentration")
            plt.ylabel("Wavelength")
            plt.legend()
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
                    input_user= input("Please eneter the desire Wavelength: ")
                    input_user= float(input_user)
                    
                    #user_concentration = (input_user- b) / W

                else:

                    break


            print("------------")
            print("Making prediction")
            print("passing ", user_concentration, " to robot")
            return input_user, user_concentration, train_prediction , W, b
    
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
        # #NOTE we may get fancier in the future here and do more preprocessing
        # if self.scan_bounds:
        #     #if you only want to pay attention in bounds, train on those vals
        #     processedX = X[:, self.scan_bounds[0]:self.scan_bounds[1]]
        # else:
        #     processedX = X



        
            
            #plt.plot(df['Concentration'], train_prediction, color='red',label="Predicted Wavelength")
            #plt.scatter(df['Concentration'], df['Wavelength'], label="Training Data")
            #plt.scatter(user_concentration,input_user,label="Predicted Value by Model")
            #plt.xlabel("Concentration")
            #plt.ylabel("Wavelength")
            #plt.legend()

        # def MainModel(X,Y):
        #     #passing to list
        #     X_df = [X[0][r] for r in range(len(X[0]))]
        #     Y_df = [Y[0][m] for m in range(len(Y[0]))]
        #     #Checking that they are list Y_df
        #     print("Checking if X and Y are list now: ", type(X_df),type(Y_df))

        #     ##creadting the dataframe
        #     df = pd.DataFrame(list(zip(X_df, Y_df)),columns =['Concentration', 'Wavelength'])
        #     #sorting by Concentration
        #     df = df.sort_values(by=['Concentration'])

        #     input_user= input("Please eneter the desire Wavelength: ")
        #     input_user= float(input_user)

        #     W_list=[]
        #     b_list=[]
        #     ##Training 
        #     #Receive data from robot
        #     for r in range(10):
        #         print("Epoch", r+1)
        #         user_concentration, train_prediction, W, b = training(self, df,input_user,r)
        #         W_list.append(W)
        #         b_list.append(b)
        #         ##pas to the robot THE USER_CONCEN
        #         Robot_answer= 465
        #         print("Robot send back a wavelenght of", Robot_answer)
        #         print("User input was ", input_user)
        #         test_error= Robot_answer- input_user
        #         print("Our test error is ", test_error)
                
        #         ##Plot
        #         fig = plt.figure(figsize=(5,8)) 
        #         plt.plot(df['Concentration'], train_prediction, color='red',label="Predicted Wavelength")
        #         plt.scatter(df['Concentration'], df['Wavelength'], label="Training Data")
        #         plt.scatter(user_concentration,input_user,label="Predicted Value by Model")
        #         plt.xlabel("Concentration")
        #         plt.ylabel("Wavelength")
        #         plt.legend()
        #         plt.show()
        #         fig.savefig("predictions-"+str(r)+"png",dpi=fig.dpi)
                
                # if test_error < 10 and test_error>-10:
                #     print("The model is trained----")
                #     break
                # else:
                #     print("Model would try to reduce the error")
                #     if r == 9:
                #         print("---->Done training<----")
                #     else:
                #         print("-----")
                #     new_data = {'Concentration': user_concentration, 'Wavelength': input_user}
                #     df = df.append(new_data, ignore_index = True)
                # r += 1
                





        #update the data with the new scans
        # def mainModel(self,N,recipes,learning_rate,X=None,Y=None):
        #     cacheError=[]
        #     cacheErrorAvg=[]
        #     cacheTheta=[]
        #     cacheBias =[]
        #     cacheParamsToUse={}
        #     breakEpoch=0
        #     Y_values=[]
        
        
        #     for i in range(N):
            
        #         ##
            
        #         if i !=0:
        #             theta = cacheTheta[i-1]-learning_rate*cacheError[i-1]/(recipes)
        #             bias = cacheBias[i-1]-learning_rate*cacheBias[i-1]/(recipes)
        #             print(theta.shape,"NNN")
        #             print(bias.shape,"NNN")
                 
        #         else:
        #             #Generating random parameters at the beginning
        #             #theta = np.random.rand(recipes,1)
        #             #bias  = np.random.rand(recipes,1)
        #             #bias  = random.uniform(0,1)
        #             X_mean=np.mean(X)
        #             theta= Y*(X-X_mean) / (X-X_mean)**2
        #             bias= np.mean(Y) - theta*np.mean(X)

            
            
        #         ##
        #         #Linear Model
            
        #         #X_mean=np.mean(X)
        #         #theta= Y*(X-X_mean) / (X-X_mean)**2
        #         print("Bias", bias)
        #         print("Theta.shape", theta.shape)
        #         print("Bias.shape", bias.shape)
        #         print("X Shape", X.shape)


        #         Y_hat= theta@X.T + bias
        #         print("Y_hat",Y_hat)
            
        #         ##SEND X
        #         ##back y
            
        #         #Computer error
            
        #         #Simple error
        #         #error= Y_hat-Y.T
        #         #
        #         print("Dii", np.abs(Y_hat-Y))#np.sqrt(np.sum((Y_hat-Y.T)**2,axis=1)))
        #         error= np.abs(Y_hat-Y)# np.sum(np.abs(Y_hat-Y.T),axis=1)#np.sqrt(np.sum((Y_hat-Y.T)**2,axis=1))  #Y_hat-Y.T#np.sum((Y_hat-Y.T).T@(Y_hat-Y.T))
            
            
            
            
        #         print("Y",Y)
        #         print("-----------------------")
        #         print("Error avg",np.sum(error)/(2*recipes))
        #         print("Error in epoch #"+str(i+1))
            
        #         print(error)
        #         print()
        #         print()
        
            
            
        #         #Saving in cache error and theta
                
        #         cacheErrorAvg.append(np.sum(error)/(2*recipes))
        #         cacheError.append(error)
        #         cacheTheta.append(theta)
        #         cacheBias.append(bias)
                
                
        #         #Callback
                
        #         #if error keeps increasing we stop and return the previous last parameter
        #         #after three epochs 
                
        #         if i > 2:
        #             if np.sum(error)/(2*recipes) > np.sum(cacheError[i-2])/(2*recipes) + 5:
        #                 breakEpoch= i
        #                 print("n",i)
        #                 print("Error avg",np.sum(cacheError[i-2])/(2*recipes))
        #                 print("Last param",np.sum(cacheTheta[i-2])/(2*recipes))
        #                 print("Last param",np.sum(cacheBias[i-2])/(2*recipes))
        #                 cacheParamsToUse["Theta"]= np.sum(cacheTheta[i-2]/(2*recipes))
        #                 cacheParamsToUse["Bias"] = np.sum(cacheBias[i-2]/(2*recipes))
        #                 break
        #             else:
        #                 pass
        #         if i==N-1:
        #             breakEpoch= i
        #             cacheParamsToUse["Theta"]= np.sum(cacheTheta[i-2]/(2*recipes))
        #             cacheParamsToUse["Bias"] = np.sum(cacheBias[i-2]/(2*recipes))
                
        #     return {"receipt":X, "cacheErrorAvg":cacheErrorAvg, "cacheTheta":cacheTheta,"cacheBias":cacheBias, "break_epoch":breakEpoch+1, "ParamsToUse":cacheParamsToUse}       
        # #time.sleep(40)
        # #time.sleep(10)
        print("X and Y")
        print(X,y) 
        print("--------")
        print('<<ML>> training')

        # ml_model_trained = MainModel(self,10,3,0.05,X,y)

        print('<<ML>> done training')

        return 0 #MainModel(self, X,y)


        # with self.model_lock:
            # if isinstance(self.X,np.ndarray):
            #     self.X = np.concatenate((self.X, processedX))
            #     self.y = np.concatenate((self.y, y))
            # else:
            #     self.X = processedX
            #     self.y = y
            # print("model fitting on X", self.X)
            # print("model fitting on y", self.y)
            # self.model.fit(self.X, self.y)
        # print('<<ML>> done training')


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
