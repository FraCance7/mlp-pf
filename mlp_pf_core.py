import os
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import tensorflow as tf
from tensorflow.keras import Sequential # type: ignore
from tensorflow.keras.layers import Dense # type: ignore
from tqdm import tqdm
from metrics import Metrics


class MLP_PF:
    def __init__(
        self,
        dir_project   = "" ,
        test_pack     = "battery5",
        training_pack = "battery5",
        model_name    = "mlp_model",
        var_x         = "Cycle",
        var_y         = "qMax",
        degradation   = "positive",
        horizon       = 200,
        y_failure     = 8300,
        Ns            = 500,
        hidden_nodes  = 3,
        epochs        = 500,
        s2z           = 0.1,
        s2x0          = 0.01,
        s2x1          = 100,
        s2x2          = 0.001,
        alpha         = 0.2,     
        random_seed   = True,
        save_data_fig = False,
        show_fig      = False,
        save_weights  = False,
        load_weights  = True,
        hydra         = True,
        name_exp      = "experiment"

    ):
        self.test_pack     = test_pack
        self.training_pack = training_pack
        self.model_name    = model_name
        self.var_x         = var_x
        self.var_y         = var_y
        self.degradation   = degradation
        self.horizon       = horizon
        self.y_failure     = y_failure
        self.save_data_fig = save_data_fig
        self.show_fig      = show_fig
        self.save_weights  = save_weights
        self.load_weights  = load_weights
        self.Ns            = Ns
        self.hidden_nodes  = hidden_nodes
        self.epochs        = epochs
        self.s2z           = s2z
        self.s2x0          = s2x0
        self.s2x1          = s2x1
        self.s2x2          = s2x2
        self.alpha         = alpha
        self.dir_project   = dir_project
        self.random_seed   = random_seed
        self.hydra         = hydra
        self.name_exp      = name_exp

    def BuildMLP(self):
        '''
        Builds the multi-layer perceptron with keras.
        2 layers are created, one with hidden nodes and sigmoid activation function, 
        the other with outpud nodes, with linear activation function.

        Outputs:

        model - keras MLP with 2 layers

        '''
        output_nodes = 1

        first_layer  = Dense(self.hidden_nodes, activation = 'selu')
        #second_layer = Dense(self.hidden_nodes, activation='sigmoid')
        output_layer = Dense(output_nodes, activation='linear')

        model = Sequential([tf.keras.Input(shape=(1,))])
        model.add(first_layer)
        #model.add(second_layer)
        model.add(output_layer)

        return model

    def TrainingMLP(self, model, texp):
        '''
        Once the MLP is built (see BuildMLP method), trains it with 
        data read from csv file. Please see examples on data directory for
        the format of the csv file.

        Inputs:

        model -- Keras model
        texp -- array of floats, expanded time, from 1st to final time we want to predict results.


        Outputs:

        param:numpy array -  weights and biases of the MLP
        y0: numpy array - training data, output
        t0: numpy array - training data, input
        ymean: float - mean of output training data
        ystd: float - standard deviation of output training data
        pred_train: numpy array, unnormalized prediction from normalized texp data.

        Options:

        load_weights: if True, load a h5 file containing weights (no fit performed)
        save_weights: if True, save the new weights in a h5 file 
        save_data_fig: if True, save the figure in training.pdf file and prediction from training dataset in a csv file.

        '''

        # read train data and normalize
        train = pd.read_csv(self.dir_project + f'data/{self.training_pack}.csv')
        y0 = np.array(train[self.var_y])
        t0 = np.array(train[self.var_x])

        tnorm_train = (t0-np.mean(t0))/np.std(t0)
        ymean = np.mean(y0)
        ystd = np.std(y0)
        ytrain = (y0-ymean)/ystd

        # normalize test data
        tmean = np.mean(texp)
        tstd = np.std(texp)
        tnorm = (texp-tmean)/tstd

        # prepare the model for training
        model.compile(loss=tf.keras.losses.mse,
                      optimizer=tf.keras.optimizers.Adam(),
                      metrics=['mse']
                      )
        callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor="loss",
                    min_delta=1e-8,
                    patience=50,
                    restore_best_weights=True,
                )
        ]

        # Fit the model, according to options if weights exist
        if self.save_weights == True:
            # Save the weights and fit model
            model.fit(tnorm_train, ytrain, epochs = self.epochs, callbacks=callbacks)
            model.save_weights(self.dir_project + f'models/{self.model_name}.h5')
        elif self.load_weights == True:
            # Load the weights and no fit needed
            model.load_weights(self.dir_project + f'models/{self.model_name}.h5')
        else:
            model.fit(tnorm_train, ytrain, epochs = self.epochs, callbacks=callbacks)

        # Generate figure
        pred_train = model(tnorm)
        pred_train = pred_train * ystd + ymean

        pred_train_fit = model(tnorm_train)
        pred_train_fit = pred_train_fit * ystd + ymean



        param = []
        param.append(model.trainable_weights)        
        param = np.array(self.serialised_params(param))

        return param, ymean, ystd, y0, t0, pred_train, pred_train_fit

    def calculate_failure(self):
        '''
        Calculate the cycle of failure and the ground truth RUL.
        '''
        # degradation
        if self.degradation == "positive":
            #self.yfailure = np.min(self.y) + 0.01 * np.min(self.y)
            self.yfailure = self.y_failure
            tfail = np.where(self.y <= self.yfailure)[0][0]

        elif self.degradation == "negative":
            #self.yfailure = np.max(self.y) - 0.01 * np.max(self.y)
            self.yfailure = self.y_failure
            tfail = np.where(self.y >= self.yfailure)[0][0]
        x_failure = self.t[tfail]
        Tfail = int(x_failure) # Cycle of failure (actual cycle, not the index of array self.t)

        GT_rul = x_failure - self.t # Ground Truth RUL
        GT_rul = GT_rul[GT_rul >= 0]

        self.y_lim_min = np.min(self.y) - 0.1 * np.min(self.y)
        self.y_lim_max = np.max(self.y) + 0.1 * np.max(self.y)

        return GT_rul, Tfail

    def curve_check(self, pred_nn):
        '''


        '''
        # Initialize failed_particles as a zero array with the same shape as pred_nn
        failed_particles = np.zeros(pred_nn.shape[0])

        if self.degradation == "positive":
            # Check if each prediction in pred_nn has at least one value under the threshold
            condition1 = np.any(pred_nn < self.yfailure, axis=1)
            # Check if the current value is higher than the threshold
            condition2 = pred_nn[:, self.current_cycle] > self.yfailure
            # Set failed_particles to 1 where both conditions are met
            failed_particles[(condition1 & condition2)] = 1

        elif self.degradation == "negative":
            # Check if each prediction in pred_nn has at least one value under the threshold
            condition1 = np.any(pred_nn > self.yfailure, axis=1)
            # Check if the current value is higher than the threshold
            condition2 = pred_nn[:, self.current_cycle] < self.yfailure

            condition3 = np.all(np.diff(pred_nn, axis=1) >= 0, axis=1)
            failed_particles[(condition1 & condition2 & condition3)] = 1

        return failed_particles  

    def update_weights(self, ytmp, ytmp_diff, pred_nn):
        '''
        Update the weights of the particles.
        '''
        s2z_eye_inv = (1 / self.s2z) * np.eye(len(ytmp))
        self.wn[:, self.current_cycle] = np.exp(-0.5 * np.sum(np.matmul(ytmp_diff ** 2, s2z_eye_inv), axis=1))

        # Find indices where pred_nn is less than self.yfailure
        self.y_failed[:, self.current_cycle] = self.curve_check(pred_nn)

        # Set wn to 1e-300 where self.y_failed is 0
        self.wn[self.y_failed[:, self.current_cycle] == 0, self.current_cycle] = 1e-300

        # Weight normalization and updating
        self.wn[:, self.current_cycle] = self.wn[:, self.current_cycle] / np.sum(self.wn[:, self.current_cycle])
        self.wncum[:, self.current_cycle] = np.cumsum(self.wn[:, self.current_cycle])  # Cumulative sum of the weights

    def resample_particles(self, x):
        '''
        Resample the particles.
        '''
        x_old = x[:, :, self.current_cycle]
        for ii in range(self.Ns):
            rand_num = np.random.rand()
            # Position of particle to resample
            res_index = np.where(self.wncum[:, self.current_cycle] >= rand_num)[0][0]
            # Effective resample  
            x[ii, :, self.current_cycle] = x_old[res_index, :]
        return x

    def rul_estimation(self, pred_nn, texp):

        # Calculate prediction statistics
        self.mean_prediction = np.mean(pred_nn, axis=0)
        variance_prediction = np.var(pred_nn, axis=0)
        self.lower_bound_prediction = self.mean_prediction - np.sqrt(variance_prediction)
        self.upper_bound_prediction = self.mean_prediction + np.sqrt(variance_prediction)

        failure_positions = np.zeros(self.Ns)
        for jj in range(self.Ns):
            if self.degradation == "positive":
                indices = np.where(pred_nn[jj, :] < self.yfailure)[0]
            elif self.degradation == "negative":
                indices = np.where(pred_nn[jj, :] > self.yfailure)[0]
            if indices.size == 0:
                failure_positions[jj] = 0
            else:
                failure_positions[jj] = texp[indices[0]]

        failure_positions = failure_positions[failure_positions > 0]  # Position of failure
        self.num_failures[self.current_cycle] = len(failure_positions)  # Number of particles reaching failure

        if failure_positions.size != 0:
            self.mean_failure_position[self.current_cycle] = np.mean(failure_positions)  # Mean of failure_positions
            self.var_failure_position[self.current_cycle] = np.var(failure_positions)
            self.estimated_rul[self.current_cycle] = self.mean_failure_position[self.current_cycle] - self.t[self.current_cycle]  # Estimation of RUL
            self.rul_lower_bound[self.current_cycle] = self.estimated_rul[self.current_cycle] - np.sqrt(self.var_failure_position[self.current_cycle])
            self.rul_upper_bound[self.current_cycle] = self.estimated_rul[self.current_cycle] + np.sqrt(self.var_failure_position[self.current_cycle])
            self.current_time[self.current_cycle] = self.t[self.current_cycle]
        else:
            self.mean_failure_position[self.current_cycle] = np.nan
            self.var_failure_position[self.current_cycle] = np.nan
            self.estimated_rul[self.current_cycle] = np.nan
            self.rul_lower_bound[self.current_cycle] = np.nan
            self.rul_upper_bound[self.current_cycle] = np.nan
            self.current_time[self.current_cycle] = np.nan
    
    def update_state_noise(self):
        '''
        Method to update the state noise
        '''
        self.s2x[self.current_cycle] = self.s2x0 * math.exp(-self.current_cycle/self.s2x1) + self.s2x2

    def run(self):
        '''

        Main method to execute the model prediction process, 
        including data loading, model building, training, 
        and result plotting.

        '''

        # Set random seed for reproducibility. 
        # WARNING: This works only if the model is pre-loaded. If the model is trained, the seed is not set.
        if self.random_seed == True:
            np.random.seed(42)
            tf.random.set_seed(42)

        # init save directory for this method and called methods
        if self.save_data_fig == True:
            if self.hydra == True:
                self.save_path = f'results'  
            else:
                self.save_path = f'results/{self.name_exp}'
            os.makedirs(self.save_path, exist_ok=True)
            os.makedirs(self.save_path+"/data", exist_ok=True)
            os.makedirs(self.save_path+"/figures", exist_ok=True)
        
        # Load the test data. var_x and var_y are the columns name of the csv file.
        parameters  = pd.read_csv(self.dir_project + f'data/{self.test_pack}.csv')
        self.t      = np.array(parameters[self.var_x])
        self.y      = np.array(parameters[self.var_y])
        Nt          = len(self.t)

        # texp is the expanded time, from 1st to final time we want to predict results.
        texp = np.arange(1, self.horizon)
        Texp = int(texp[-1])

        # Initialize state noise
        self.current_cycle = 0
        self.s2x = np.zeros(Texp)
        self.s2x[0] = self.s2x0 * math.exp(-self.current_cycle/self.s2x1) + self.s2x2
        
        # Calculate the cycle of failure and the ground truth RUL
        GT_rul, Tfail = self.calculate_failure()

        # Build and train the MLP
        model = self.BuildMLP()
        Nx = model.count_params()
        (x0, 
         ymean, 
         ystd, 
         y0, 
         t0, 
         pred_train, 
         pred_train_fit) = self.TrainingMLP(model, texp)

        tmean = np.mean(t0)
        tstd = np.std(t0)
        tnorm = (texp-tmean)/tstd

        # Initialize the weights and biases of the CustomMLP, used in place of tensorflow to speed up the computation 
        nb_hidden_neur = [el.units for el in model.layers[:-1]]
        activations = [el.activation for el in model.layers[:]]
        models = CustomMLP(self.Ns, nb_hidden_neur, activations)

        # x is the state of the particles, x0 is the initial state
        x = np.zeros((self.Ns, Nx, Texp))
        x[:, :, 0] = x0

        # Initialize the arrays to store the predictions, weights and cumulative weights
        pred_nn                     = np.zeros((self.Ns, Texp))  # create an array to store the predictions
        self.wn                     = np.zeros((self.Ns, Tfail)) # create an array to store the likelihoods
        self.wncum                  = np.zeros((self.Ns, Tfail)) # create an array to store the cumulative likelihoods
        self.num_failures           = np.zeros(Nt)
        self.mean_failure_position  = np.zeros(Nt)
        self.var_failure_position   = np.zeros(Nt)
        self.estimated_rul          = np.zeros(Nt)
        self.rul_lower_bound        = np.zeros(Nt)
        self.rul_upper_bound        = np.zeros(Nt)
        self.current_time           = np.zeros(Nt)
        self.y_failed               = np.zeros((self.Ns, Texp))

        font_size = 16
        if self.show_fig == True:
            fig = plt.figure(figsize=(10, 10))
            gs = gridspec.GridSpec(2, 2)

            ax1 = plt.subplot(gs[0, 0])
            ax2 = plt.subplot(gs[0, 1])
            ax3 = plt.subplot(gs[1, :])  # This creates a subplot in the second row that spans both columns
            axs = [ax1, ax2, ax3]
        
        # Loop as new observations come in
        for self.current_cycle in tqdm(range(1, Tfail)):

            # Process equation, describing the evolution of the particles.
            x[:, :, self.current_cycle] = x[:, :, self.current_cycle-1] + np.sqrt(self.s2x[self.current_cycle-1]) * np.random.randn(self.Ns, Nx)

            weights, biases = self.deserialise_params(models, x[:, :, self.current_cycle])
            models.set_params(weights, biases)

            ## PREDICTION STAGE ## 

            # gk is the normalized prediction of the particles.
            gk = models(np.expand_dims(tnorm, axis=1))
            gk = gk.T
            # preds are the predictions of the particles denormalized.
            preds = gk*ystd+ymean
            pred_nn = preds
            preds = preds.T

            # difference between the prediction and the actual value.
            ytmp        = (self.y[:self.current_cycle+1]-ymean)/ystd
            ytmp_diff   = ytmp - gk[:, :self.current_cycle+1]

            ## UPDATE STAGE ##

            # Update the weights of the particles.
            self.update_weights(ytmp, ytmp_diff, pred_nn)
            # Resample the particles.
            x = self.resample_particles(x)

            #Estimate statistics related to the RUL predictions
            self.rul_estimation(pred_nn, texp)            

            #Plot and/or save figures
            if self.show_fig == True:
                self.plot_results(axs,
                                  texp,
                                  Tfail, 
                                  GT_rul, 
                                  pred_nn
                                  )
                
            # Save the data and figures related to specific cycles    
            if self.save_data_fig==True:
                if self.current_cycle in [1, 100, 140]:
                    self.save_plot_history(texp, 
                                   pred_train, 
                                   pred_nn, 
                                   font_size
                                   )

            # Update state noise
            self.update_state_noise()

        # Computes evaluation metrics
        metrics = Metrics(GT_rul[1:], 
                          self.estimated_rul[1:self.current_cycle+1], 
                          self.rul_upper_bound[1:self.current_cycle+1], 
                          self.rul_lower_bound[1:self.current_cycle+1], 
                          self.alpha
                          )
        
        cic     = metrics.compute_cic()
        cic_25  = metrics.compute_cic_25()
        cre     = metrics.compute_cre()
        cre_25  = metrics.compute_cre_25()
        beta    = metrics.compute_beta()
        beta_25 = metrics.compute_beta_25()

        if self.save_data_fig==True:
            self.save_fig_data(font_size, 
                               t0, 
                               y0, 
                               pred_train_fit, 
                               GT_rul, 
                               Tfail, 
                               cic, 
                               cic_25, 
                               cre, 
                               cre_25, 
                               beta, 
                               beta_25)

        plt.close('all')

        return(cic, cre, beta, cic_25, cre_25, beta_25)

    # Serialisation and deserialisation methods. Insert an example
    def serialise_params_1(self, params):
        serialised_params = []
        for param in params:
            serialised_params.append(param.numpy().ravel())
        return np.concatenate(serialised_params)

    def serialised_params(self, param_simil):
        return [self.serialise_params_1(init_param) for init_param in param_simil]

    def deserialise_params(self, models, s_params):
            res = [[], []]
            ind_cursor = 0
            for i in range(len(models.weights)):
                res[0].append(
                    np.reshape(
                        s_params[
                            :,
                            ind_cursor : ind_cursor
                            + models.weights[i].shape[-2]
                            * models.weights[i].shape[-1],
                        ],
                        newshape=(
                            s_params.shape[0],
                            models.weights[i].shape[-2],
                            models.weights[i].shape[-1],
                        ),
                    )
                )
                ind_cursor = (
                    ind_cursor
                    + models.weights[i].shape[-2] * models.weights[i].shape[-1]
                )
                res[1].append(
                    np.expand_dims(
                        s_params[
                            :, ind_cursor : ind_cursor + models.biases[i].shape[-1]
                        ],
                        axis=1,
                    )
                )
                ind_cursor = ind_cursor + models.biases[i].shape[-1]
            return res

    def save_plot_history(self, texp, pred_train, pred_nn, font_size):
        '''

        Inputs:

        texp:
        pred_train:
        pred_nn:
        lower_bound_prediction:
        V_up:
        font_size:

        Outputs

        - a data_cycle_k.csv file.
        - a predictions_cycle_k.pdf figure in pdf file.

        '''
        # save data in csv file
        np.savetxt(f'{self.save_path}/data/data_cycle_{self.current_cycle}.csv', pred_nn, delimiter=',')

        # open a figure
        fig, host = plt.subplots(figsize=(8,5))
        plt.cla()

        # plot data
        if self.current_cycle == 1:
            plt.plot(texp, pred_train, 'r-', label = 'Training')
        plt.scatter(texp[:self.current_cycle+1], self.y[:self.current_cycle+1], edgecolor="blue", marker='o',facecolor='red')
        plt.scatter(self.t, self.y, edgecolor="blue", marker='o',facecolor='none')
        predictions_label_added = False

        for j in range(0, self.Ns, 100):
            plt.plot(pred_nn[j, :], color="green", linestyle='--')
            if not predictions_label_added:
                plt.plot([], [], color="green", linestyle='--', label='Predictions')
                predictions_label_added = True
        #
        plt.fill_between(texp, self.lower_bound_prediction, self.upper_bound_prediction, color='green', alpha=0.1, label='5 to 95 percentile') 
        plt.axhline(y=self.yfailure, color='gray', linestyle='--', label='Threshold') 
        plt.axvline(x=self.current_cycle, color='red', linestyle='--', alpha=0.5, label='Current Cycle self.current_cycle')

        # set plot parameters
        plt.ylim([self.y_lim_min, self.y_lim_max])
        plt.xlabel(self.var_x, fontsize=font_size)
        plt.ylabel(self.var_y, fontsize=font_size)
        plt.tick_params(axis='both', which='major', labelsize=14)

        plt.legend(fontsize=13)
        plt.tight_layout()

        plt.savefig(f'{self.save_path}/figures/predictions_cycle_{self.current_cycle}.pdf')
        plt.close()

    def save_fig_data(self, font_size, t0, y0, pred_train_fit, GT_rul, Tfail, cic, cic_25, cre, cre_25, beta, beta_25):
        # Plot the training data and the predictions
        fig, host = plt.subplots(figsize=(8, 5))
        plt.cla()
        plt.plot(t0, pred_train_fit, 'r-', label='Training')
        plt.scatter(t0, y0, edgecolor="blue", marker='o',facecolor='none', label='Train data') 
        plt.scatter(self.t, self.y, color="yellow", marker='o', alpha=0.5, label='Test data')
        plt.ylim([self.y_lim_min, self.y_lim_max])
        plt.xlabel(f'{self.var_x}')
        plt.ylabel(f'{self.var_y}')
        plt.title(f'Training on {self.test_pack}')
        plt.legend()
        plt.tight_layout()
        # save pred_train in a csv file
        np.savetxt(f'{self.save_path}/data/pred_train.csv', pred_train_fit, delimiter=',')
        plt.savefig(f'{self.save_path}/figures/training.pdf') 
        
        # Plot the RUL
        fig, host = plt.subplots(figsize=(8,5))
        plt.cla()
        plt.plot(self.t[:Tfail], GT_rul, 'b-', label='True RUL')
        plt.plot(self.t[1:self.current_cycle+1], self.estimated_rul[1:self.current_cycle+1], "r--", label="Mean Predicted RUL")
        plt.plot(self.t[1:self.current_cycle+1], self.rul_lower_bound[1:self.current_cycle+1], "g--", label="Percentiles of Predicted RUL")
        plt.plot(self.t[1:self.current_cycle+1], self.rul_upper_bound[1:self.current_cycle+1], "g--")
        plt.xlabel('Cycles', fontsize=font_size)
        plt.ylabel('RUL', fontsize=font_size)
        plt.tick_params(axis='both', which='major', labelsize=14)
        plt.legend(fontsize=13)
        plt.tight_layout()
        # save the figure
        plt.savefig(f'{self.save_path}/figures/rul.pdf')
        
        # Create a DataFrame with your data
        df = pd.DataFrame({
            'Cycles': self.t[1:self.current_cycle+1],
            'True RUL': GT_rul[1:self.current_cycle+1],
            'Mean Predicted RUL': self.estimated_rul[1:self.current_cycle+1],
            'self.rul_lower_bound': self.rul_lower_bound[1:self.current_cycle+1],
            'self.rul_upper_bound': self.rul_upper_bound[1:self.current_cycle+1]
        })
        # Save the DataFrame to a CSV file
        df.to_csv(f'{self.save_path}/data/data_RUL.csv', index=False)
       
       # Define your properties and hyperparameters
        metrics = {
            'CIC': cic,
            'CIC_25': cic_25,
            'CRE': cre,
            'CRE_25': cre_25,
            'Beta': beta,
            'Beta_25': beta_25
        }
        # Save properties in a txt file.
        with open(f'{self.save_path}/metrics.txt', 'w') as f:
            # Write the properties
            f.write('Metrics:\n')
            for property, value in metrics.items():
                f.write(f'{property}: {value}\n')

    def plot_results(self, axs, texp, Tfail, GT_rul, pred_nn):
        axs[0].cla()
        axs[0].plot(self.t[:Tfail], GT_rul, 'b-')
        axs[0].plot(self.t[1:self.current_cycle], self.estimated_rul[1:self.current_cycle], "r--")
        axs[0].plot(self.t[1:self.current_cycle], self.rul_lower_bound[1:self.current_cycle], "g--")
        axs[0].plot(self.t[1:self.current_cycle], self.rul_upper_bound[1:self.current_cycle], "g--")
        axs[0].set_ylabel('RUL')
        axs[0].set_xlabel('Cycles')

        axs[1].cla()
        axs[1].plot(self.t[1:self.current_cycle], self.s2x[1:self.current_cycle], "g--")
        axs[1].set_xlim([0, Tfail])
        axs[1].set_ylim([0.0001, 0.005])
        axs[1].set_ylabel('State Noise')
        axs[1].set_xlabel(f'{self.var_x}')

        axs[2].cla() # clear the current axis
        axs[2].set_ylim([self.y_lim_min, self.y_lim_max])
        axs[2].scatter(self.t, self.y, edgecolor="blue", marker='o',facecolor='none')
        axs[2].scatter(texp[:self.current_cycle+1], self.y[:self.current_cycle+1], edgecolor="blue", marker='o',facecolor='red')
        counter = 0
        for j in range(0, self.Ns, 100):
            if counter >= 10:
                break
            axs[2].plot(pred_nn[j, :], c="g")
            counter += 1
        axs[2].plot(texp, self.mean_prediction, 'k--')
        axs[2].fill_between(texp, self.lower_bound_prediction, self.upper_bound_prediction, color='green', alpha=0.2)  
        axs[2].axhline(y=self.yfailure, color='gray', linestyle='--')
        axs[2].axvline(x=self.current_cycle, color='red', linestyle='--', alpha=0.5)
        axs[2].set_ylabel(f'{self.var_y}')
        axs[2].set_xlabel(f'{self.var_x}')

        plt.pause(0.1)

class CustomMLP:
    def __init__(self, nb_clones, nb_neurons_hidden_layers, activations):
        self.weights = [np.zeros((1, nb_clones, 1, nb_neurons_hidden_layers[0]))]
        self.biases = [np.zeros((nb_clones, 1, nb_neurons_hidden_layers[0]))]
        nb_neurons_prev = nb_neurons_hidden_layers[0]
        for nb_neurons in nb_neurons_hidden_layers[1:] + [1]:
            self.weights.append(np.zeros((nb_clones, nb_neurons_prev, nb_neurons)))
            self.biases.append(np.zeros((nb_clones, 1, nb_neurons)))
            nb_neurons_prev = nb_neurons
        self.activations = activations

    def set_params(self, weights: list, biases: list):
        self.weights = [np.expand_dims(weights[0], axis=0)] + weights[1:]
        self.biases = biases

    def __call__(self, X):
        res = self.activations[0](np.tensordot(X, self.weights[0], 1) + self.biases[0])

        for weights, biases, activation in zip(
            self.weights[1:], self.biases[1:], self.activations[1:]
        ):
            res = activation(np.matmul(res, weights) + biases)

        return(res.squeeze())

    

        
