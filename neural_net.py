import numpy as np
import logging
log = logging.getLogger(__name__)


class TwoLayerNet(object):
  """
  A two-layer fully-connected neural network. The net has an input dimension of
  N, a hidden layer dimension of H, and performs classification over C classes.
  We train the network with a softmax loss function. The network uses a ReLU 
  nonlinearity after the first fully connected layer.

  In other words, the network has the following architecture:

  input - fully connected layer - ReLU - fully connected layer - softmax

  The outputs of the second fully-connected layer are the scores for each class.
  """

  def __init__(self, input_size, hidden_size, output_size, std=1e-4):
    """
    Initialize the model. Weights are initialized to small random values and
    biases are initialized to zero. Weights and biases are stored in the
    variable self.params, which is a dictionary with the following keys:

    W1: First layer weights; has shape (D, H)
    b1: First layer biases; has shape (H,)
    W2: Second layer weights; has shape (H, C)
    b2: Second layer biases; has shape (C,)

    Inputs:
    - input_size: The dimension D of the input data.
    - hidden_size: The number of neurons H in the hidden layer.
    - output_size: The number of classes C.
    """
    self.params = {}
    self.params['W1'] = std * np.random.randn(input_size, hidden_size)
    self.params['b1'] = np.zeros(hidden_size)
    self.params['W2'] = std * np.random.randn(hidden_size, output_size)
    self.params['b2'] = np.zeros(output_size)

  def loss(self, X, y=None):
    """
    Compute the loss and gradients for a two layer fully connected neural
    network.

    Inputs:
    - X: Input data of shape (N, D). Each X[i] is a training sample.
    - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
      an integer in the range 0 <= y[i] < C. This parameter is optional; if it
      is not passed then we only return scores, and if it is passed then we
      instead return the loss and gradients.

    Returns:
    If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
    the score for class c on input X[i].

    If y is not None, instead return a tuple of:
    - loss: data loss for this batch of training samples.
    - grads: Dictionary mapping parameter names to gradients of those parameters
      with respect to the loss function; has the same keys as self.params.
    """
    # Unpack variables from the params dictionary
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    N, D = X.shape
    C, = b2.shape

    # Compute the forward pass

    # fc1
    h1 = np.dot(X, W1) + b1
    # relu
    h1[h1 <= 0] = 0
    # fc2
    scores = np.dot(h1, W2) + b2

    # If the targets are not given then jump out, we're done
    if y is None:
      return scores

    # Compute the loss
    
    scores -= np.max(scores)  # for numeric stability
    scores_exp = np.exp(scores)
    softmax = scores_exp / np.sum(scores_exp, axis=1)[:, np.newaxis]

    # cross-entropy
    loss = - np.average(np.log(softmax[np.arange(N), y]))

    # Backward pass: compute gradients
    grads = {} 

    # softmax
    onehot = np.zeros((N, C))
    onehot[np.arange(N), y] = 1
    grad_softmax = softmax - onehot

    # fc2
    grad_W2 = h1.T.dot(grad_softmax) / N
    grad_b2 = np.sum(grad_softmax, axis=0) / N

    # relu
    grad_h1 = grad_softmax.dot(W2.T)
    grad_h1[h1 <= 0] = 0

    # fc1
    grad_W1 = X.T.dot(grad_h1) / N
    grad_b1 = np.sum(grad_h1, axis=0) / N

    grads['W2'] = grad_W2
    grads['b2'] = grad_b2
    grads['W1'] = grad_W1
    grads['b1'] = grad_b1

    return loss, grads

  def train(self, X, y, X_val, y_val,
        learning_rate=1e-3, learning_rate_decay=0.95,
        num_iters=100, batch_size=200):
    """
    Train this neural network using stochastic gradient descent.

    Inputs:
    - X: A numpy array of shape (N, D) giving training data.
    - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
      X[i] has label c, where 0 <= c < C.
    - X_val: A numpy array of shape (N_val, D) giving validation data.
    - y_val: A numpy array of shape (N_val,) giving validation labels.
    - learning_rate: Scalar giving learning rate for optimization.
    - learning_rate_decay: Scalar giving factor used to decay the learning rate
      after each epoch.
    - num_iters: Number of steps to take when optimizing.
    - batch_size: Number of training examples to use per step.
    """
    num_train = X.shape[0]
    iterations_per_epoch = max(num_train / batch_size, 1)

    # Use SGD to optimize the parameters in self.model
    loss_history = []
    train_acc_history = []
    val_acc_history = []

    for it in range(num_iters):

      indices = np.random.choice(num_train, batch_size)
      X_batch = X[indices]
      y_batch = y[indices]

      # Compute loss and gradients using the current minibatch
      loss, grads = self.loss(X_batch, y=y_batch)
      loss_history.append(loss)

      # parameter update
      for key in self.params:
        self.params[key] -= learning_rate * grads[key]

      if it % 100 == 0:
        log.info('iteration %d / %d: loss %f' % (it, num_iters, loss))

      # Every epoch, check train and val accuracy and decay learning rate.
      if it % iterations_per_epoch == 0:
        # Check accuracy
        train_acc = (self.predict(X_batch) == y_batch).mean()
        val_acc = (self.predict(X_val) == y_val).mean()
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)

        # Decay learning rate
        learning_rate *= learning_rate_decay

    return {
      'loss_history': loss_history,
      'train_acc_history': train_acc_history,
      'val_acc_history': val_acc_history,
    }

  def predict(self, X):
    """
    Use the trained weights of this two-layer network to predict labels for
    data points. For each data point we predict scores for each of the C
    classes, and assign each data point to the class with the highest score.

    Inputs:
    - X: A numpy array of shape (N, D) giving N D-dimensional data points to
      classify.

    Returns:
    - y_pred: A numpy array of shape (N,) giving predicted labels for each of
      the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
      to have class c, where 0 <= c < C.
    """

    scores = self.loss(X)  # y not given, return scores
    y_pred = np.argmax(scores, axis=1)

    return y_pred
