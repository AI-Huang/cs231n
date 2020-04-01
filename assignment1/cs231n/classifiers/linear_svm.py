from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

from tqdm import tqdm


def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

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
    dW = np.zeros(W.shape)  # initialize the gradient as zero
    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    # h = 0.00001  # delta h
    # hW = W - h
    for i in range(num_train):
        scores = X[i].dot(W)
        # hscores = X[i].dot(hW)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            margin = scores[j] - correct_class_score + 1  # note delta = 1
            if j == y[i]:
                if margin > 0:
                    dW[:, j] += X[i].T  # current class j
                continue
            if margin > 0:
                loss += margin
                dW[:, y[i]] += -X[i].T  # other class i

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    dW /= num_train
    dW += 2 * reg * W         # (1)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0]
    num_classes = W.shape[1]
    scores = X.dot(W)  # N by C
    scores_correct = scores[np.arange(num_train), y]  # 1 by N
    scores_correct = np.reshape(scores_correct, (num_train, 1))  # N by 1

    margins = scores - scores_correct + 1.0  # N by C
    margins[margins <= 0] = 0.0              # hinge loss

    incorrect_class = np.ones(margins.shape)
    incorrect_class = incorrect_class > 0  # bool array
    # ignore correct class for loss
    incorrect_class[np.arange(num_train), y] = False
    loss += np.sum(margins[incorrect_class]) / num_train
    loss += reg * np.sum(W * W)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    margins[margins > 0] = 1.0                         # 示性函数 N by C
    margins[incorrect_class] = - \
        margins[incorrect_class]  # set -1  for j <> y[i]
    n_identity = np.sum(margins, axis=1)
    # print(n_identity.shape) # 500
    # for i in range(num_train):
    # dW[:, y[i]] += X[i] * n_identity[i]
    # print(X.shape)  # (500, 3073)
    _ = X.T.dot(np.diagflat(n_identity))
    # print(_.shape)  # (3073, 500)
    # map (3073, 500) to 3073 by 10
    for i in range(num_train):
        dW[:, y[i]] += _[:, i]

    # not work~
    # dW[:, y] += X.T.dot(np.diagflat(n_identity))[:,
        #  np.arange(num_train)]
    dW /= num_train
    dW += 2 * reg * W
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
