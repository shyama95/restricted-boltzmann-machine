import matplotlib.pyplot as plt
import numpy as np

class RBM:
    def __init__(self, input_size=1, hidden_layer_size=1, time_steps=1):
        # m = input size
        # n = hidden layer size
        # k = time steps

        self.m = input_size
        self.n = hidden_layer_size
        self.k = time_steps
        
        # W = weight matrix
        # b = visible layer bias vetor
        # c = hidden layer bias vector
        self.W = np.zeros((input_size, hidden_layer_size))
        self.b = np.zeros((input_size, 1))
        self.c = np.zeros((1, hidden_layer_size))

    def train(self, data=[], learning_rate=0.01):
        print("Training begins...")
        dw = np.zeros((self.m, self.n))
        db = np.zeros((self.m, 1))
        dc = np.zeros((1, self.n))
        
        N = data.shape[0]

        for i in range(N):
            v_t = data[i, :].reshape((self.m, 1)).copy()

            for t in range(self.k):
                h_t = self.sample_hidden(v_t)
                v_t = self.sample_visible(h_t)

            v_0 = data[i, :].reshape((self.m, 1)).copy()
            p_h_v_0 = self.sigmoid(np.dot(np.transpose(v_0), self.W) + self.c)
            v_k = v_t.copy()
            p_h_v_k = self.sigmoid(np.dot(np.transpose(v_k), self.W) + self.c)

            dw = dw + np.dot(v_0, p_h_v_0) - np.dot(v_k, p_h_v_k) 
            db = db + v_0 - v_k 
            dc = dc + p_h_v_0 - p_h_v_k

        self.W = self.W + learning_rate * dw
        self.b = self.b + learning_rate * db
        self.c = self.c + learning_rate * dc
        
        # print("W shape is {}".format(self.W.shape))
        # print("b shape is {}".format(self.b.shape))
        # print("c shape is {}".format(self.c.shape))

    def inference(self, v):
        print("Inference begins...")
        v_t = v.reshape((self.m, 1)).copy()
        k = self.k

        plt.subplot(k+1, 1, 1)
        plt.plot(v.reshape(28, 28))

        for t in range(self.k):
            h_t = self.sample_hidden(v_t)
            # print("H shape is {}".format(h_t.shape))
            v_t = self.sample_visible(h_t)        
            # print("V shape is {}".format(v_t.shape))
            
            plt.subplot(k+1, 1, t+2)
            plt.plot(v_t.reshape(28, 28))

        plt.show()

    def sigmoid(self, x):
        return 1.0/(1.0 + np.exp(-x))
    
    def sample_hidden(self, v):
       # print("V input to sample_hidden shape is {}".format(v.shape))
       # print("W shape is {}".format(self.W.shape))
       print("c shape is {}".format(self.c.shape))
       p_h_v = self.sigmoid(np.dot(np.transpose(v), self.W) + self.c)
       # print("p_h_v shape is {}".format(p_h_v.shape))
       h = np.random.binomial(n=1, p=p_h_v, size=p_h_v.shape)
       # print("H shape is {}".format(h.shape))
       return h
    
    def sample_visible(self, h):
       p_v_h = self.sigmoid(np.dot(self.W, np.transpose(h)) + self.b)
       # print("p_v_h shape is {}".format(p_v_h.shape))
       v = np.random.binomial(n=1, p=p_v_h, size=p_v_h.shape)
       # print("V shape is {}".format(v.shape))
       return v

