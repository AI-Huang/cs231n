from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

# @RefLink    : https://zhuanlan.zhihu.com/p/21485970


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    num_classes = W.shape[1]
    num_train = X.shape[0]
    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    for i in range(num_train):
        scores = X[i].dot(W)
        scores = np.exp(scores)  # softmax loss
        prob = scores / np.sum(scores)

        # 归一化概率
        # prob = np.exp(f - f_max) / np.sum(np.exp(f - f_max),
        #   axis=1, keepdims=True)
        for j in range(num_classes):
            if j == y[i]:
                loss += - np.log(prob[y[i]])
                dW[:, j] += - (1 - prob[y[i]]) * X[i]
                continue
            # no loss for other j
            dW[:, j] += prob[y[i]] * X[i]

    loss /= num_train
    loss += reg * np.sum(W * W)

    dW /= num_train
    dW += 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    num_classes = W.shape[1]
    num_train = X.shape[0]
    num_dim = X.shape[1]
    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    scores = X.dot(W)
    scores = np.exp(scores)  # softmax scores, n_train * n_classes
    prob = np.divide(scores.T, np.sum(scores, axis=1)).T

    one_hot = np.eye(num_classes)[y]
    for i in range(num_train):
        dW += np.tile(X[i], (num_classes, 1)).T * (prob[i, y[i]] * (1-one_hot[i])
                                                   - (1-prob[i, y[i]]) * one_hot[i])
    # dW += np.tile(X[i], (num_classes, 1)).T

    loss += - np.log(prob[np.arange(num_train), y])
    loss = np.sum(loss)
    loss /= num_train
    loss += reg * np.sum(W * W)

    dW /= num_train
    dW += 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
