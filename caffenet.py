import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np

class CaffeNet(chainer.Chain):

    def __init__(self):
        super(self.__class__, self).__init__(
            conv1=L.Convolution2D(3, 96, ksize=11, stride=4),
            conv2=L.Convolution2D(96, 256, ksize=5, pad=2),
            conv3=L.Convolution2D(256, 384, ksize=3, pad=1),
            conv4=L.Convolution2D(384, 384, ksize=3, pad=1),
            conv5=L.Convolution2D(384, 256, ksize=3, pad=1),
            fc6=L.Linear(9216, 4096),
            fc7=L.Linear(4096, 4096),
            fc8=L.Linear(4096, 1000),
        )

    def __call__(self, x, t, train=True):
        y = self.forward(x, train=train)
        self.loss = F.softmax_cross_entropy(y, t)
        self.accuracy = F.accuracy(y, t)
        return self.loss

    def forward(self, x, train=False):
        h = F.relu(self.conv1(x))
        h = F.max_pooling_2d(h, ksize=3, stride=2)
        h = F.local_response_normalization(h, k=5, n=5, alpha=1e-4, beta=0.75)
        h *= np.power(5, 0.75)
        h = F.relu(self.conv2(h))
        h = F.max_pooling_2d(h, ksize=3, stride=2)
        h = F.local_response_normalization(h, k=5, n=5, alpha=1e-4, beta=0.75)
        h *= np.power(5, 0.75)
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = F.relu(self.conv5(h))
        h = F.max_pooling_2d(h, ksize=3, stride=2)
        h = F.dropout(F.relu(self.fc6(h)), train=train)
        h = F.dropout(F.relu(self.fc7(h)), train=train)
        h = self.fc8(h)
        return h
