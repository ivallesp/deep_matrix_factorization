import tensorflow as tf
import numpy as np
from src.general_utilities import batching



class DeepFactorization():
    """
    Class implementing Deep Matrix Factorization. Libraries required: Numpy and Tensorflow

    Methods definition:
    __init__: initializes the class, defines the initial weight and bias matrices
        :param n_users: users cardinality (int)
        :param n_items: items cardinality (int)
        :param emb_size: size of the embedding (int)
        :param lr: learning rate. 1e-3 by default (float)
        :param _lambda: Keep probability for dropout regularization. The greater, the stronger.
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
        tf.reset_default_graph()
        self.ph_keep_prob = tf.placeholder(dtype=tf.float32, shape=(None), name="keep_prob")
        self.ph_u_ids=tf.placeholder(dtype=tf.int32, shape=(None,), name="u_ids_ph")
        self.ph_i_ids=tf.placeholder(dtype=tf.int32, shape=(None,), name="u_ids_ph")
        self.ph_y = tf.placeholder(dtype=tf.float32, shape=(None, 1), name="y")
        self.u_emb = tf.get_variable(shape=(n_users, emb_size), name="U_Embedding")
        self.i_emb = tf.get_variable(shape=(n_items, emb_size), name="I_Embedding")
        self.u_bias = tf.get_variable(shape=(n_users, 1), name="U_Bias")
        self.i_bias = tf.get_variable(shape=(n_items, 1), name="I_Bias")
        u=tf.nn.embedding_lookup(self.u_emb, self.ph_u_ids) #+ tf.nn.embedding_lookup(self.u_bias, self.ph_u_ids) 
        i=tf.nn.embedding_lookup(self.i_emb, self.ph_i_ids) #+ tf.nn.embedding_lookup(self.i_bias, self.ph_i_ids)
        ub=tf.nn.embedding_lookup(self.u_bias, self.ph_u_ids)
        ib=tf.nn.embedding_lookup(self.i_bias, self.ph_i_ids)
        ub=tf.nn.dropout(ub, self.ph_keep_prob)
        ib=tf.nn.dropout(ib, self.ph_keep_prob)
        u=tf.nn.dropout(u, self.ph_keep_prob)
        i=tf.nn.dropout(i, self.ph_keep_prob)
        self.mfac = tf.reduce_sum(u*i + ub + ib, axis=1, keepdims=True)
        self.mfac = tf.nn.dropout(self.mfac, self.ph_keep_prob)
        emb = tf.concat([u, i], axis=1)
        emb = tf.nn.dropout(emb, self.ph_keep_prob)
        h = tf.layers.Dense(128, tf.nn.swish)(emb)
        h = tf.nn.dropout(h, self.ph_keep_prob)
        h = tf.layers.Dense(64, tf.nn.swish)(h)
        h = tf.nn.dropout(h, self.ph_keep_prob)
        h = tf.layers.Dense(32, tf.nn.swish)(h)
        h = tf.nn.dropout(h, self.ph_keep_prob)
        self.output = tf.layers.Dense(1)(h) + self.mfac
        self.loss = tf.losses.mean_squared_error(self.ph_y, self.output)
        self.op = tf.train.RMSPropOptimizer(learning_rate=lr).minimize(self.loss)
        self.sess=tf.Session()
        self.sess.run(tf.global_variables_initializer())
        
    def fit(self, x, y, batch_size):
        batcher = batching([list(x), list(y)], n=batch_size, return_incomplete_batches=True)
        for batch_x, batch_y in batcher:
            self.train_on_batch(batch_x, batch_y)
        
    def train_on_batch(self, x, y):
        u_ids, i_ids = list(zip(*list(x)))                              
        self.sess.run(self.op, feed_dict={self.ph_u_ids: u_ids, 
                                    self.ph_i_ids: i_ids,
                                    self.ph_y: y,
                                    self.ph_keep_prob: 1-self._lambda})
        
    
    def predict(self, x, batch_size):
        batcher = batching([list(x)], n=batch_size, return_incomplete_batches=True)
        preds = []
        for batch_x in batcher:
            batch_x = batch_x[0]
            u_ids, i_ids = list(zip(*list(batch_x)))
            preds.append(self.sess.run(self.output, feed_dict={self.ph_u_ids: u_ids, 
                                                      self.ph_i_ids: i_ids,
                                                      self.ph_keep_prob: 1.0}))
        preds = np.row_stack(preds)
        return preds
    
    def evaluate(self, x, y, batch_size):
        return np.mean((np.round(self.predict(x, batch_size)) - y)**2)
    