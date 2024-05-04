from scipy.special import softmax
import numpy as np


class Actor(object):
    """
    Allowing to change the parameters of the actor's policy externally.
    """

    def __init__(self, s_size, a_size):
        self.ns = s_size
        self.na = a_size

    def set_theta(self, theta):
        self.theta = theta.reshape(self.na, self.ns)

    def fc(self, x):
        return self.theta @ x

    def forward(self, x):
        x = self.fc(x)
        return softmax(x)

    def act(self, state):
        probs = self.forward(state)
        return np.argmax(probs)
