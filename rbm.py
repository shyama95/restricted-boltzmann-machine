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
        self.W = np.random.uniform(low=-4*np.sqrt(6/(input_size+hidden_layer_size)), high=4*np.sqrt(6/(input_size+hidden_layer_size)), size=(input_size, hidden_layer_size))
        self.b = np.zeros((input_size, 1))
        self.c = np.zeros((1, hidden_layer_size))

    def train(self, data=[], learning_rate=0.01, epochs=50, batch_size=32):
        print("Training begins...")

        alpha = 0.5
        beta = 0.00005
        dw = np.zeros((self.m, self.n))
        db = np.zeros((self.m, 1))
        dc = np.zeros((1, self.n))
        
        N = data.shape[0]
        batch_count = int(N/batch_size - 1) if N%batch_size == 0 else int(N/batch_size)
        
        if batch_count == 0 and N != 0:
            batch_count = 1
            batch_size = N

        # print("Batch count is {}".format(batch_count))
        # print("data is {}".format(data))

        # print("W is {}".format(self.W))
        # print("b is {}".format(self.b))

        for epoch in range(epochs):
            print("Running epoch {}".format(epoch+1))
            for k in range(batch_count):
                for j in range(batch_size):
                    i = batch_size * k + j
                    v_t = data[i, :].reshape((self.m, 1)).copy()

                    # print("Max in v is {}".format(np.max(v_t)))

                    for t in range(self.k):
                        h_t = self.sample_hidden(v_t)
                        v_t = self.sample_visible(h_t)

                    v_0 = data[i, :].reshape((self.m, 1)).copy()
                    p_h_v_0 = self.sigmoid(np.dot(np.transpose(v_0), self.W) + self.c)
                    v_k = v_t.copy()
                    p_h_v_k = self.sigmoid(np.dot(np.transpose(v_k), self.W) + self.c)
                   
                    # print("v_0 size is {}, v_k size is {}".format(v_0.shape, v_k.shape))
                    # print("Visible input is {} and fantasy output is {}".format(v_0, v_k)) 
                    # print("Difference : v_0 - v_k is {}".format(v_0 - v_k))
                    dw = dw + np.dot(v_0, p_h_v_0) - np.dot(v_k, p_h_v_k)
                    db = db + v_0 - v_k
                    dc = dc + p_h_v_0 - p_h_v_k

                    # print("dw is {}".format(dw))
                    # print("db is {}".format(db))
                    # print("dc is {}".format(dc))
                    
                    # print("db shape is {}".format(db.shape))
                
                self.W = alpha * self.W + learning_rate * dw + beta * np.linalg.norm(self.W) #/ np.linalg.norm(dw)
                self.b = alpha * self.b + learning_rate * db #/ np.linalg.norm(db)
                self.c = (alpha - 0.2) * self.c + learning_rate * dc #/ np.linalg.norm(dc)
            
            print("Max value in W is {}".format(np.max(self.W)))
            print("Min value in W is {}".format(np.min(self.W)))
            print("Max value in b is {}".format(np.max(self.b)))
            print("Min value in b is {}".format(np.min(self.b)))
            print("Max value in c is {}".format(np.max(self.c)))
            print("Min value in c is {}".format(np.min(self.c)))
        
        # print("W shape is {}".format(self.W.shape))
        # print("b shape is {}".format(self.b.shape))
        # print("c shape is {}".format(self.c.shape))
        

    def inference(self, v):
        print("Inference begins...")
        v_t = v.reshape((self.m, 1)).copy()
        k = self.k

        plt.subplot(k+1, 1, 1)
        plt.imshow(v.reshape(28, 28), cmap='gray')

        for t in range(self.k):
            h_t = self.sample_hidden(v_t)
            # print("H shape is {}".format(h_t.shape))
            v_t = self.sample_visible(h_t)        
            # print("V shape is {}".format(v_t.shape))
            
            plt.subplot(k+1, 1, t+2)
            plt.imshow(v_t.reshape(28, 28), cmap='gray')

        plt.show()

    def sigmoid(self, x):
        # print("Sigmoid input is {}".format(x))
        # print("Sigmoid input shape is {}".format(x.shape))
        z = np.exp(x) 
        y = z / (1 + z)
        # y = 1.0/(1.0 + np.exp(-x))
        # print("Sigmoid output is {}".format(y))
        # print("Sigmoid output shape is {}".format(y.shape))
        return y
    
    def sample_hidden(self, v):
       # print("V input to sample_hidden shape is {}".format(v.shape))
       # print("W shape is {}".format(self.W.shape))
       # print("c shape is {}".format(self.c.shape))
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
    
    def free_energy(self, v, h):
        energy = - np.sum(np.multiply(self.b, v)) - np.sum(np.multiply(self.c, h)) - np.sum(np.multiply(np.dot(v,h), self.W))
        return energy

    def save_weights(self, filename='model'):
        w = np.reshape(self.W, (self.W.size,1))
        b = np.reshape(self.b, (self.b.size,1))
        c = np.reshape(self.c, (self.c.size,1))
        weights = np.append(w, b)
        weights = np.append(weights, c)
        np.save(filename, weights)            
        
    def load_weights(self, filename='model.npy'):
        weights = np.load(filename)
        self.W = np.reshape(weights[0:self.m * self.n], (self.m, self.n))
        self.b = np.reshape(weights[self.m * self.n: self.m * self.n + self.m], (self.m, 1))
        self.c = np.reshape(weights[self.m * self.n + self.m:], (1, self.n))



