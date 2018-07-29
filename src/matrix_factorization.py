import numpy as np
from src.general_utilities import batching


class MatrixFactorization():
    """
    Class implementing Matrix Factorization. Libraries required: Numpy

    Methods definition:
    __init__: initializes the class, defines the initial weight and bias matrices
        :param n_users: users cardinality (int)
        :param n_items: items cardinality (int)
        :param emb_size: size of the embedding (int)
        :param lr: learning rate. 1e-3 by default (float)
        :param _lambda: L2 regularization strength, the greater, the stronger.
        0.1 by default (float)
        :return: None (Void)

    fit: runs the batch training process using stochastic gradient descent
        :param x: iterable containing (user, item) tuples (iterable, e.g. list of tuples)
        :param y: iterable containing the target variable to be learned. It should have the
        same length as x (iterable)
        :param batch_size: size of the batch to train with (int)
        :return: None (Void)

    train_on_batch: runs a training step on a given batch using gradient
    descent
        :param x: iterable containing (user, item) tuples (iterable, e.g. list of tuples)
        :param y: iterable containing the target variable to be learned. It should have the
        same length as x (iterable)
        :return: None (Void)

    predict: given a set of users and items, predicts the outcome and returns it. Batch computation
    has been used in order to make the method memory-scalable
        :param x: iterable containing (user, item) tuples (iterable, e.g. list of tuples)
        :param batch_size: size of the batch to train with (int)
        :return: the outcomes belonging to the (user, item) tuples at the input, in the
        same order (np.array with shape (len(x), 1))

    evaluate: predicts the outcome for the given input 'x' and compares it with the desired
    output 'y' using the Mean Squared Error metric. Batch computation has been used in order 
    to make the method memory-scalable
        :param x: iterable containing (user, item) tuples (iterable, e.g. list of tuples)
        :param y: iterable containing the target variable to be learned. It should have the
        same length as x (iterable)
        :param batch_size: size of the batch to train with (int)
        :return: MSE (float)
    """
    def __init__(self, n_users, n_items, emb_size, lr=0.001, _lambda = 0.1):
        self.lr = lr
        self._lambda = _lambda
        self.u_emb = np.random.normal(size=(n_users, emb_size))
        self.i_emb = np.random.normal(size=(n_items, emb_size))
        self.u_bias = np.zeros(shape=(n_users, 1))
        self.i_bias = np.zeros(shape=(n_items, 1))
        
    def fit(self, x, y, batch_size):
        batcher = batching([list(x), list(y)], n=batch_size, return_incomplete_batches=True)
        for batch_x, batch_y in batcher:
            self.train_on_batch(batch_x, batch_y)
        
    def train_on_batch(self, x, y):
        u, i = list(zip(*list(x)))
        p = self.predict(x, batch_size = len(x))
        e = (p-y)
        dwi = e*self.u_emb[list(u)] + self._lambda*self.i_emb[list(i)]
        dwu = e*self.i_emb[list(i)] + self._lambda*self.u_emb[list(u)]
        dbi = e + self._lambda*self.i_bias[list(i)]
        dbu = e + self._lambda*self.u_bias[list(u)]
        self.i_emb[list(i)] = self.i_emb[list(i)] - self.lr*dwi
        self.u_emb[list(u)] = self.u_emb[list(u)] - self.lr*dwu
        self.i_bias[list(i)] = self.i_bias[list(i)] - self.lr*dbi
        self.u_bias[list(u)] = self.u_bias[list(u)] - self.lr*dbu
    
    def predict(self, x, batch_size):
        batcher = batching([list(x)], n=batch_size, return_incomplete_batches=True)
        preds = []
        for batch_x in batcher:
            batch_x = batch_x[0]
            u, i = list(zip(*list(batch_x)))
            preds.append(np.sum(self.u_emb[list(u)] * self.i_emb[list(i)] \
                          + self.u_bias[list(u)] \
                          + self.i_bias[list(i)], axis=1, keepdims=True))
        preds = np.row_stack(preds)
        return preds
    
    def evaluate(self, x, y, batch_size):
        return np.mean((np.round(self.predict(x, batch_size)) - y)**2)