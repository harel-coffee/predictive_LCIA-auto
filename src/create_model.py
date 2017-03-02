'''
Created on Feb 23, 2017
create the predictive LCIA model in tensorflow
@author: runsheng
'''
import numpy as np
import pandas as pd

import tensorflow as tf
from sklearn import cross_validation
from sklearn.datasets import load_boston
from sklearn.metrics import r2_score

from tensorflow.contrib import learn
from sklearn import preprocessing

BATCH_SIZE = 1
class single_layer_model:
    def __init__(self):
        pass
    
    def _init_weight(self,shape):
        weights = tf.random_normal(shape,stddev=0.1)
        return tf.Variable(weights)
    
    def _feedforward(self,X,w1,w2):
        h1 = tf.nn.sigmoid(tf.matmul(X,w1))
        y_ = tf.matmul(h1,w2)
        return y_
        
    def build(self,input_size,num_neroun,output_size, learning_rate=0.01):
        '''
        build the structure of the neural net
        '''
        #create placeholders for X and y
        self.X = tf.placeholder("float",shape=[None,input_size])
        self.y = tf.placeholder("float",shape=[None,output_size])
        
        #weights
        self.w1 = self._init_weight((input_size,num_neroun))
        self.w2 = self._init_weight((num_neroun,output_size))
        
        #init feedforward
        y_ = self._feedforward(self.X, self.w1, self.w2)
        self.pred = y_
        
        #init backpropagation
#         cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_,self.y))
        cost = tf.reduce_mean(tf.square(y_ - self.y))
        
        #add regularization term
        
        #update weights
        updates = tf.train.AdagradOptimizer(learning_rate).minimize(cost)
        
        #saver
        
        return updates, cost
    
    def train(self,trn_X,trn_Y,tst_X,tst_Y,
              num_epoch=200,num_neroun=64,learning_rate=0.01,
              verbose=True):
        '''
        train the neural nets
        '''
        #layer sizes:
        x_size = trn_X.shape[1]
        h_size = num_neroun
        y_size = trn_Y.shape[1]
        
        #init cost/update function
        updates, cost = self.build(input_size=x_size,num_neroun=h_size,output_size=y_size,learning_rate=0.01)
        
        #init session
        init = tf.global_variables_initializer()
        costs = []
        with tf.Session() as sess:
            sess.run(init)
            for epoch in range(num_epoch):
                for i in range(0,len(trn_X),BATCH_SIZE):
                    sess.run(updates, feed_dict={self.X:trn_X[i:i+BATCH_SIZE], self.y:trn_Y[i:i+BATCH_SIZE]})

                trn_cost = r2_score(trn_Y,sess.run(self.pred, feed_dict={self.X:trn_X, self.y:trn_Y}))
                tst_cost = r2_score(tst_Y,sess.run(self.pred, feed_dict={self.X:tst_X, self.y:tst_Y}))
                costs.append(tst_cost)
                
                print epoch, trn_cost,tst_cost
#                 print("Epoch = %d, Training Cost = %.2f%, Testing Costs = %.2f%" % (epoch+1, trn_cost, tst_cost))
            pred_y = sess.run(self.pred,feed_dict={self.X:tst_X,self.y:tst_Y})
            
            for (y,y_hat) in zip(test_y,pred_y)[0:10]:
                print y,y_hat

def load_lcia_data(descs_p, target_p):
    X = pd.read_csv(descs_p,header=0,index_col=None)
    X = X.fillna(X.mean)
    y = pd.read_csv(target_p,header=0,index_col=None)
    return X.values,y.values
           
if __name__ == '__main__':
#     boston = learn.datasets.load_dataset('boston')
#     X, y = boston.data, boston.target
    descs_p = '../data/descs/descs_Feb22_166.csv'
    target_p = '../data/target/CED.csv'
    X,y = load_lcia_data(descs_p, target_p)

#     y.resize( y.size, 1 )

    train_x, test_x, train_y, test_y = cross_validation.train_test_split(
                                    X, y, test_size=0.2, random_state=42)
    scaler = preprocessing.StandardScaler( )
    train_x = scaler.fit_transform( train_x )
    test_x  = scaler.transform( test_x )
    
    this_model = single_layer_model()
    
    this_model.train(train_x, train_y, test_x, test_y, num_epoch=400, num_neroun=18, learning_rate=0.01)
    
    
    
    
    
    