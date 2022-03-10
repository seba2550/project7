# BMI 203 Project 7: Neural Network


# Importing Dependencies
import enum
from functools import cache
from logging import PlaceHolder

from pkg_resources import yield_lines
import numpy as np
from typing import List, Dict, Tuple
from numpy.typing import ArrayLike


# Neural Network Class Definition
class NeuralNetwork:
    """
    This is a neural network class that generates a fully connected Neural Network.

    Parameters:
        nn_arch: List[Dict[str, Union(int, str)]]
            This list of dictionaries describes the fully connected layers of the artificial neural network.
            e.g. [{'input_dim': 64, 'output_dim': 32, 'activation': 'relu'}, {'input_dim': 32, 'output_dim': 8, 'activation' : 'sigmoid'}] will generate a
            2 layer deep fully connected network with an input dimension of 64, a 32 dimension hidden layer
            and an 8 dimensional output.
        lr: float
            Learning Rate (alpha).
        seed: int
            Random seed to ensure reproducibility.
        batch_size: int
            Size of mini-batches used for training.
        epochs: int
            Max number of epochs for training.
        loss_function: str
            Name of loss function.

    Attributes:
        arch: list of dicts
            This list of dictionaries describing the fully connected layers of the artificial neural network.
    """
    def __init__(self,
                 nn_arch: List[Dict[str, int]],
                 lr: float,
                 seed: int,
                 batch_size: int,
                 epochs: int,
                 convergence_thresh: float,
                 loss_function: str):
        # Saving architecture\\\\\
        self.arch = nn_arch
        # Saving hyperparameters
        self._lr = lr
        self._seed = seed
        self._epochs = epochs
        self._loss_func = loss_function
        self._convergence_thresh = convergence_thresh # Added this, user input for determining threshold for convergence (based on difference between losses between two epochs)
        self._batch_size = batch_size
        # Initializing the parameter dictionary for use in training
        self._param_dict = self._init_params()

        # Input validation for loss function
        assert self._loss_func == "mse" or "bce", "Loss function must be one of: MSE, BCE"

    def _init_params(self) -> Dict[str, ArrayLike]:
        """
        DO NOT MODIFY THIS METHOD!! IT IS ALREADY COMPLETE!!

        This method generates the parameter matrices for all layers of
        the neural network. This function returns the param_dict after
        initialization.

        Returns:
            param_dict: Dict[str, ArrayLike]
                Dictionary of parameters in neural network.
        """
        # seeding numpy random
        np.random.seed(self._seed)
        # defining parameter dictionary
        param_dict = {}
        # initializing all layers in the NN
        for idx, layer in enumerate(self.arch):
            layer_idx = idx + 1
            input_dim = layer['input_dim']
            output_dim = layer['output_dim']
            # initializing weight matrices
            param_dict['W' + str(layer_idx)] = np.random.randn(output_dim, input_dim) * 0.1
            # initializing bias matrices
            param_dict['b' + str(layer_idx)] = np.random.randn(output_dim, 1) * 0.1
        return param_dict

    def _single_forward(self,
                        W_curr: ArrayLike,
                        b_curr: ArrayLike,
                        A_prev: ArrayLike,
                        activation: str) -> Tuple[ArrayLike, ArrayLike]:
        """
        This method is used for a single forward pass on a single layer.

        Args:
            W_curr: ArrayLike
                Current layer weight matrix.
            b_curr: ArrayLike
                Current layer bias matrix.
            A_prev: ArrayLike
                Previous layer activation matrix.
            activation: str
                Name of activation function for current layer.

        Returns:
            A_curr: ArrayLike
                Current layer activation matrix.
            Z_curr: ArrayLike
                Current layer linear transformed matrix.
        """
        Z_curr = A_prev.dot(W_curr.T) + b_curr.T
        #Z_curr = np.transpose(np.matmul(W_curr, np.transpose(A_prev))) # Had to transpose A_prev and the product of the mat mult to get this to work
        #Z_curr += b_curr.flatten()
        # Apply the specified activation function to our current layer linear transformed matrix. That is now our current layer activation matrix
        # BUT FIRST, some input verification
        activation = activation.lower() # Let's convert the input to lowercase to make everyone's lives easier
        # Unit test to verify we're getting ReLU or Sigmoid as input and not anything funky
        assert activation == 'relu' or activation == 'sigmoid', "Activation function must be one of: ReLU, Sigmoid"

        if activation == 'relu':
            A_curr = self._relu(Z_curr)
        elif activation == 'sigmoid':
            A_curr = self._sigmoid(Z_curr)

        return A_curr, Z_curr

    def forward(self, X: ArrayLike) -> Tuple[ArrayLike, Dict[str, ArrayLike]]:
        """
        This method is responsible for one forward pass of the entire neural network.

        Args:
            X: ArrayLike
                Input matrix with shape [batch_size, features].

        Returns:
            output: ArrayLike
                Output of forward pass.
            cache: Dict[str, ArrayLike]:
                Dictionary storing Z and A matrices from `_single_forward` for use in backprop.
        """
        print('Forwaaaaaaaard')
        cache = {} # Initialize an empty dictionary for our matrices cache
        A_prev = X # At the start, the input matrix is our A matrix
        cache['A0'] = A_prev
        # for idx, layer in enumerate(self.arch): # Iterate over the network using the same approach from the _init_params method
        #     next_idx = idx + 1 # Index to go through the network
        #     string_next_idx = str(next_idx) # Cast the index to string for accesing the params dict
        #     activation_func_curr = layer['activation'] # Get the activation function for our current layer from the 'arch' dictionary
        #     W_curr = self._param_dict['W' + string_next_idx] # Get the weights for our current layer from the params dictionary
        #     b_curr = self._param_dict['b' + string_next_idx] # Get the biases for our current layer from the params dictionary

        #     A_curr, Z_curr = self._single_forward(W_curr, b_curr, A_prev, activation_func_curr) # Use the _single_forward function to get current layer activation matrix and current layer linear transformed matrix

        #     cache["A" + string_next_idx] = A_curr # Store the A matrix in our cache
        #     cache["Z" + string_next_idx] = Z_curr # As well as our current Z matrix
            
        #     A_prev = A_curr # Update!
        for layer in range(1, len(self.arch) + 1):
            print('Forward function, going through layer ' + str(layer))
            W_curr = self._param_dict['W' + str(layer)]
            b_curr = self._param_dict['b' + str(layer)]
            activation_func = self.arch[layer - 1]['activation']
            A_curr, Z_curr = self._single_forward(W_curr, b_curr, A_prev, activation_func)

            cache['A' + str(layer)] = A_curr
            cache['Z' + str(layer)] = Z_curr

            A_prev = A_curr
            print('Switcheroo done')

        return A_curr, cache # Activation matrix for final layer is the output, and we also return the cache dictionary

    def _single_backprop(self,
                         W_curr: ArrayLike,
                         b_curr: ArrayLike,
                         Z_curr: ArrayLike,
                         A_prev: ArrayLike,
                         dA_curr: ArrayLike,
                         activation_curr: str) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
        """
        This method is used for a single backprop pass on a single layer.

        Args:
            W_curr: ArrayLike
                Current layer weight matrix.
            b_curr: ArrayLike
                Current layer bias matrix.
            Z_curr: ArrayLike
                Current layer linear transform matrix.
            A_prev: ArrayLike
                Previous layer activation matrix.
            dA_curr: ArrayLike
                Partial derivative of loss function with respect to current layer activation matrix.
            activation_curr: str
                Name of activation function of layer.

        Returns:
            dA_prev: ArrayLike
                Partial derivative of loss function with respect to previous layer activation matrix.
            dW_curr: ArrayLike
                Partial derivative of loss function with respect to current layer weight matrix.
            db_curr: ArrayLike
                Partial derivative of loss function with respect to current layer bias matrix.
        """
        # Get our minibatch size
        mini_batch_size = A_prev.shape[1] 

        # Depending on the specified activation function, get the derivative of the current layer Z matrix
        if activation_curr == 'relu':
            dZ_curr = self._relu_backprop(dA_curr, Z_curr)
        elif activation_curr == 'sigmoid':
            dZ_curr = self._sigmoid_backprop(dA_curr, Z_curr)

        dW_curr = (A_prev.T).dot(dA_curr * dZ_curr).T
        db_curr = np.sum((dA_curr*dZ_curr), axis = 0).reshape(b_curr.shape)
        dA_prev = (dA_curr * dZ_curr).dot(W_curr)
        

        return dA_prev, dW_curr, db_curr

    def backprop(self, y: ArrayLike, y_hat: ArrayLike, cache: Dict[str, ArrayLike]):
        """
        This method is responsible for the backprop of the whole fully connected neural network.

        Args:
            y (array-like):
                Ground truth labels.
            y_hat: ArrayLike
                Predicted output values.
            cache: Dict[str, ArrayLike]
                Dictionary containing the information about the
                most recent forward pass, specifically A and Z matrices.

        Returns:
            grad_dict: Dict[str, ArrayLike]
                Dictionary containing the gradient information from this pass of backprop.
        """
        grad_dict = {} # Initialize empty dictionary which will hold all gradient information for the backprop pass
        #dA_prev = self._binary_cross_entropy_backprop(y, y_hat) # Calculate derivative of loss function wrt final activation/results 
    
        # for prev_idx, layer in list(enumerate(self.arch))[::-1]: # Iterate backwards through the network
        #     idx_curr = prev_idx + 1 # Adding one to our current index gets us one layer back through the network, which is the layer we wanna work with
        #     string_idx_curr = str(idx_curr) # Do the string casting for the index again
        #     string_idx_prev = str(prev_idx) # And for the idx, which really holds the index for the layer we started the iteration with
        #     activation_curr = layer['activation'] # Get the activation function for the current layer
        #     dA_curr = dA_prev # dA switcheroo between layers
        #     # Get the necessary matrices to run the single layer backprop
        #     W_curr = self._param_dict['W' + string_idx_curr]
        #     b_curr = self._param_dict['b' + string_idx_curr]
        #     Z_curr = cache["Z" + string_idx_curr]
        #     A_prev = cache["A" + string_idx_curr]

        #     # Get the activation function for the layer
        #     activation_curr = layer['activation']

        #     # Now do backprop for a single layer
        #     dA_prev, dW_curr, db_curr = self._single_backprop(W_curr, b_curr, Z_curr, A_prev, dA_curr, activation_curr)

        #     # Finally, store the gradient information for each pass of backprop in the gradients dictionary
        #     grad_dict["dA" + string_idx_curr] = dA_prev
        #     grad_dict["dW" + string_idx_curr] = dW_curr
        #     grad_dict["db" + string_idx_curr] = db_curr
        #     print(grad_dict)
        if self._loss_func == 'mse':
            dA_curr = self._mean_squared_error(y, y_hat)
        else:
            dA_curr = self._binary_cross_entropy(y, y_hat)

        for layer in range(len(self.arch), 0, -1):
            W_curr = self._param_dict['W' + str(layer)]
            b_curr = self._param_dict['b' + str(layer)]
            Z_curr = cache['Z' + str(layer)]
            A_prev = cache['A' + str(layer - 1)]
            activation_curr = self.arch[layer - 1]['activation']
            dA_prev, dW_curr, db_curr = self._single_backprop(W_curr, b_curr, Z_curr, A_prev, dA_curr, activation_curr)

            grad_dict['dA_prev' + str(layer)] = dA_prev
            grad_dict['dW_curr' + str(layer)] = dW_curr
            grad_dict['db_curr' + str(layer)] = db_curr

            dA_curr = dA_prev
             
        return grad_dict # Return the gradients dictionary
            


    def _update_params(self, grad_dict: Dict[str, ArrayLike]):
        """
        This function updates the parameters in the neural network after backprop. This function
        only modifies internal attributes and thus does not return anything

        Args:
            grad_dict: Dict[str, ArrayLike]
                Dictionary containing the gradient information from most recent round of backprop.

        Returns:
            None
        """
        for layer in range(1, len(self.arch)):
            self._param_dict['W' + str(layer)] = self._param_dict['W' + str(layer)] - self._lr * grad_dict['dW_curr' + str(layer)] # Update the weights. We do this by multiplying the specified learning rate and gradient info (dW), and then subtract that from the current layer weights
            self._param_dict['b' + str(layer)] = self._param_dict['b' + str(layer)] - self._lr * grad_dict['db_curr' + str(layer)] # Do exactly the same updating but for the biases
            
    def fit(self,
            X_train: ArrayLike,
            y_train: ArrayLike,
            X_val: ArrayLike,
            y_val: ArrayLike) -> Tuple[List[float], List[float]]:
        """
        This function trains the neural network via training for the number of epochs defined at
        the initialization of this class instance.
        Args:
            X_train: ArrayLike
                Input features of training set.
            y_train: ArrayLike
                Labels for training set.
            X_val: ArrayLike
                Input features of validation set.
            y_val: ArrayLike
                Labels for validation set.

        Returns:
            per_epoch_loss_train: List[float]
                List of per epoch loss for training set.
            per_epoch_loss_val: List[float]
                List of per epoch loss for validation set.
        """
        per_epoch_loss_train = [] # Initialize empty lists for placing the losses per epoch (for training and validation, respectively)
        per_epoch_loss_val = []

        per_epoch_accuracy_train = [] # Also initialize empty lists for storing the accuracy of the model per epoch
        per_epoch_accuracy_val = []

        # If the target vector is one-dimensional reshape it so that we can use it along with the predictions vector (this is due to a ValueError I was getting when calculating MSE)
        # if len(y_train.shape) == 1:
        #     y_train = y_train.reshape((len(y_train), 1))
        # if len(y_val.shape) == 1:
        #     y_val = y_val.reshape((len(y_val), 1))
        # Find the specified loss function, and pass it to a variable for easier loss calculation within the epoch. Otherwise would have to write the loops twice, once for each loss function
        if self._loss_func == 'bce':
            loss_func = self._binary_cross_entropy
        else:
            loss_func = self._mean_squared_error

        val_loss_diff = 1 
        epoch = 1
        while epoch < self._epochs: # This loop runs until we reach the user-specified number of epochs
            print(epoch)
            shuffled_data = np.concatenate([X_train, np.expand_dims(y_train, 1)], axis = 1)
            np.random.shuffle(shuffled_data)

            X_train = shuffled_data[:, :-1]
            y_train = shuffled_data[:, -1].flatten()

            n_batches = int(X_train.shape[0]/self._batch_size) + 1

            X_batch = np.array_split(X_train, n_batches)
            y_batch = np.array_split(y_train, n_batches)

            for X_train, y_train in zip(X_batch, y_batch):
                print('Forward pass')
                y_hat, cache = self.forward(X_train)
                loss = loss_func(y_train, y_hat)
                per_epoch_loss_train.append(loss)
                print(per_epoch_loss_train)
                print('Backprop pass')
                grad_dict = self.backprop(y_train, y_hat, cache)
                print('Updating parameters')
                self._update_params(grad_dict)
                print('Finished updating')
                y_pred, cache = self.forward(X_val)
                print('Validation forward complete')
                print(y_pred)
                val_loss = loss_func(y_val, y_pred)
                print('Validation loss is: ' + val_loss)
                per_epoch_loss_val.append(val_loss)
                break
            epoch += 1
        return per_epoch_loss_train, per_epoch_loss_val 

    def predict(self, X: ArrayLike) -> ArrayLike:
        """
        This function returns the prediction of the neural network model.

        Args:
            X: ArrayLike
                Input data for prediction.

        Returns:
            y_hat: ArrayLike
                Prediction from the model.
        """
        y_hat, cache = self.forward(X) # Just ignore the cache
        return y_hat

    def _sigmoid(self, Z: ArrayLike) -> ArrayLike:
        """
        Sigmoid activation function.

        Args:
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            nl_transform: ArrayLike
                Activation function output.
        """
        return 1/(1+np.exp(-Z))

    def _relu(self, Z: ArrayLike) -> ArrayLike:
        """
        ReLU activation function.

        Args:
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            nl_transform: ArrayLike
                Activation function output.
        """
        return np.maximum(0, Z) # Element-wise, if Z > 0 it returns Z, otherwise returns 0

    def _sigmoid_backprop(self, dA: ArrayLike, Z: ArrayLike):
        """
        Sigmoid derivative for backprop.

        Args:
            dA: ArrayLike
                Partial derivative of previous layer activation matrix.
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            dZ: ArrayLike
                Partial derivative of current layer Z matrix.
        """
        dZ = self._sigmoid(Z) * (1 - self._sigmoid(Z))
        return dZ

    def _relu_backprop(self, dA: ArrayLike, Z: ArrayLike) -> ArrayLike:
        """
        ReLU derivative for backprop.

        Args:
            dA: ArrayLike
                Partial derivative of previous layer activation matrix.
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            dZ: ArrayLike
                Partial derivative of current layer Z matrix.
        """
        dZ = Z > 0 # Returns True if Z > 0, which is interpreted as 1; False if Z <= 0, which is interpreted as 0
        return dZ

    def _binary_cross_entropy(self, y_hat: ArrayLike, y: ArrayLike) -> float:
        """
        Binary cross entropy loss function.

        Args:
            y_hat: ArrayLike
                Predicted output.
            y: ArrayLike
                Ground truth output.

        Returns:
            loss: float
                Average loss over mini-batch.
        """
        y_hat[y_hat == 0] = 0.0001
        y_hat[y_hat == 1] = 0.9999
        return -np.mean(((1 - y) * np.log(1 - y_hat)) + (y * np.log(y_hat))) # Same implementation of BCE that I used for project 6 (if it ain't broke, don't fix it)

    def _binary_cross_entropy_backprop(self, y_hat: ArrayLike, y: ArrayLike) -> ArrayLike:
        """
        Binary cross entropy loss function derivative.

        Args:
            y_hat: ArrayLike
                Predicted output.
            y: ArrayLike
                Ground truth output.

        Returns:
            dA: ArrayLike
                partial derivative of loss with respect to A matrix.
        """
        m = y.shape[1]
        y_hat[y_hat == 0] = 0.0001 # Avoid problems with taking the log of 0 and/or 1
        y_hat[y_hat == 1] = 0.9999
        return (1/m) * (-(y/y_hat) + ((1-y) * np.log(1- y_hat)))

    def _mean_squared_error(self, y: ArrayLike, y_hat: ArrayLike) -> float:
        """
        Mean squared error loss.

        Args:
            y: ArrayLike
                Ground truth output.
            y_hat: ArrayLike
                Predicted output.

        Returns:
            loss: float
                Average loss of mini-batch.
        """
        return np.mean(np.square(y - y_hat))

    def _mean_squared_error_backprop(self, y: ArrayLike, y_hat: ArrayLike) -> ArrayLike:
        """
        Mean square error loss derivative.

        Args:
            y_hat: ArrayLike
                Predicted output.
            y: ArrayLike
                Ground truth output.

        Returns:
            dA: ArrayLike
                partial derivative of loss with respect to A matrix.
        """
        return 2 * (y_hat - y)/ len(y)