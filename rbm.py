import matplotlib.pyplot as plt
import numpy as np


class RBM:
    """RBM - Class for implementing the Restrictive Boltzmann Machine
       Contains methods for:
            * training the RBM - train()
            * visualising the fantasy images - inference()
    """
    def __init__(self, input_size=1, hidden_layer_size=1, time_steps=1):
        """__init__
           Parameters:
                input_size: size of visible input
                hidden_layer_size: size of hidden vector
                time_steps: no. of time steps for Gibbs chain
        """
        # m = visible input size
        # n = reduced hidden vector size
        # k = time steps for Gibbs chain
        self.m = input_size
        self.n = hidden_layer_size
        self.k = time_steps
        
        # W = weight matrix
        # b = visible layer bias vetor
        # c = hidden layer bias vector
        # Initialising the weight matrix and bias vectors
        self.W = np.random.uniform(low=-4*np.sqrt(6/(input_size+hidden_layer_size)), high=4*np.sqrt(6/(input_size+hidden_layer_size)), size=(input_size, hidden_layer_size))
        self.b = np.zeros((input_size, 1))
        self.c = np.zeros((1, hidden_layer_size))

    def train(self, data=[], learning_rate=0.01, alpha=0.5, beta=0.00005, epochs=50, batch_size=32):
        """train - method for training the RBM. Uses momentum and L2 weight 
           decay for training using contrastive divergence. Updates the weights/bias 
           in the model object
           Parameters:
                data: training data of the form (no. of samples x visible input)
                learning_rate: learning rate for weight update, default value is 0.01
                alpha: momemtum metaparameter, default value is 0.5
                beta: weight decay hyperparameter, default value is 5e-5
                epochs: maximum no. of epochs for training, default value is 50
                batch_size: batch size for training, default value is 32
        """
        print("Training begins...")
        
        # Initialize weight gradients to zero
        dw = np.zeros((self.m, self.n))
        db = np.zeros((self.m, 1))
        dc = np.zeros((1, self.n))
        
        # N = no. of samples
        N = data.shape[0]
        # Compute batch count from no. of samples and batch size
        batch_count = int(N/batch_size - 1) if N%batch_size == 0 else int(N/batch_size)
        
        if batch_count == 0 and N != 0:
            batch_count = 1
            batch_size = N

        # Run the training
        for epoch in range(epochs):
            print("Running epoch {}".format(epoch+1)) 
            for k in range(batch_count):
                for j in range(batch_size):
                    i = batch_size * k + j
                    # Read visible input
                    v_t = data[i, :].reshape((self.m, 1)).copy()
                    # Run Gibbs chain for k time steps
                    for t in range(self.k):
                        # Sample hidden output
                        h_t = self.sample_hidden(v_t)
                        # Sample visible output
                        v_t = self.sample_visible(h_t)
                    
                    # Compute p(h|v_0) and p(h|v_k)
                    v_0 = data[i, :].reshape((self.m, 1)).copy()
                    p_h_v_0 = self.sigmoid(np.dot(np.transpose(v_0), self.W) + self.c)
                    v_k = v_t.copy()
                    p_h_v_k = self.sigmoid(np.dot(np.transpose(v_k), self.W) + self.c)
                    
                    # Compute gradient based on contrastive divergence
                    dw = dw + np.dot(v_0, p_h_v_0) - np.dot(v_k, p_h_v_k)
                    db = db + v_0 - v_k
                    dc = dc + p_h_v_0 - p_h_v_k
                
                # Update weights and biases
                # W = alpha*W + learning_rate * dw + beta * norm(W)
                self.W = alpha * self.W + learning_rate * dw + beta * np.linalg.norm(self.W) #/ np.linalg.norm(dw)
                # b = alpha * b + learning_rate * db
                self.b = alpha * self.b + learning_rate * db
                # c = alpha * c + learning_rate * dc
                self.c = (alpha - 0.2) * self.c + learning_rate * dc
           
            print("Free energy = {}".format(self.free_energy(data)))
            # Print values to monitor value explosion
            # print("Max value in W is {}".format(np.max(self.W)))
            # print("Min value in W is {}".format(np.min(self.W)))
            # print("Max value in b is {}".format(np.max(self.b)))
            # print("Min value in b is {}".format(np.min(self.b)))
            # print("Max value in c is {}".format(np.max(self.c)))
            # print("Min value in c is {}".format(np.min(self.c)))

    def inference(self, v):
        """inference - visualize fantasy images
           Parameters:
                v: visible input
        """
        print("Inference begins...")
        
        # v_t is the input image
        v_t = v.reshape((self.m, 1)).copy()
        # Get no. of time steps from RBM object
        k = self.k
        
        # Plot the visible input
        plt.subplot(k+1, 1, 1)
        plt.imshow(v.reshape(28, 28), cmap='gray')
        
        # Plot the fantasy images
        for t in range(self.k):
            # Sample hidden vector
            h_t = self.sample_hidden(v_t)
            # Sample fantasy image
            v_t = self.sample_visible(h_t)        
            
            plt.subplot(k+1, 1, t+2)
            plt.imshow(v_t.reshape(28, 28), cmap='gray')
        
        # Display all images
        plt.show()

    def sigmoid(self, x):
        """sigmoid - compute sigmoid : y = 1/(1 + e^(-x))
           Parameter:
                x: input
           Return:
                y: sigmoid output
        """
        z = np.exp(x) 
        y = z / (1 + z)
        return y
    
    def sample_hidden(self, v):
        """sample_hidden - sample hidden vector from Bernoulli distribution
           Parameter:
                v: visible input
           Return:
                h: sampled hidden vector
        """
        p_h_v = self.sigmoid(np.dot(np.transpose(v), self.W) + self.c)
        h = np.random.binomial(n=1, p=p_h_v, size=p_h_v.shape)
        return h
    
    def sample_visible(self, h):
        """sample_visible - sample fantasy output from Bernoulli distribution
           Parameter:
                h: sampled hidden vector
           Return:
                v: visible input
        """
        p_v_h = self.sigmoid(np.dot(self.W, np.transpose(h)) + self.b)
        v = np.random.binomial(n=1, p=p_v_h, size=p_v_h.shape)
        return v
    
    def free_energy(self, v):
        """free_energy - compute free energy for the given visible input and 
                         hidden vector as 
                         E(v,h) = - sum(b_i*v_i) - sum(c_j*h_j) - sum(v_i*h_j*W_ij)
                         where,
                            i iterates over visible input
                            j iterates over hidden vector
           Parameters:
                v: visible input
           Return :
                energy: energy computed
        """
        h = np.dot(v, self.W) + np.tile(self.c, (v.shape[0], 1))

        vbias_term = np.dot(v, self.b)
        h_term = np.sum(np.log(1 + np.exp(h)), axis=1)
        energy = -np.sum(vbias_term) - np.sum(h_term)
        energy = energy/ v.shape[0]
        # energy = - np.sum(np.multiply(self.b, v)) - np.sum(np.multiply(self.c, h)) - np.sum(np.multiply(np.dot(v,h), self.W))
        return energy

    def save_weights(self, filename='model'):
        """save_weights - saves weight matrix and bias vectors to a npy file
           Parameter:
                filename: name for output model file
        """
        # Convert weight matrix and bias vectors to 1D array and concatenate them
        w = np.reshape(self.W, (self.W.size,1))
        b = np.reshape(self.b, (self.b.size,1))
        c = np.reshape(self.c, (self.c.size,1))
        weights = np.append(w, b)
        weights = np.append(weights, c)
        # Save the 1D concatenated array to file
        np.save(filename, weights)            
        
    def load_weights(self, filename='model.npy'):
        """load_weights - load weight matrix and bias vectors from a npy file
                          into the object
           Parameter:
                filename: name for input model file
        """
        # Read from model file
        weights = np.load(filename)
        # Load model into weight matrix/ bias vectors of the object
        self.W = np.reshape(weights[0:self.m * self.n], (self.m, self.n))
        self.b = np.reshape(weights[self.m * self.n: self.m * self.n + self.m], (self.m, 1))
        self.c = np.reshape(weights[self.m * self.n + self.m:], (1, self.n))

