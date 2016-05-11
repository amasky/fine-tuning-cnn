import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np

class CaffeNet(chainer.Chain):

    def __init__(self):
        self.caffe = chainer.Chain(
            conv1 = L.Convolution2D(3, 96, ksize=11, stride=4),
            conv2 = L.Convolution2D(96, 256, ksize=5, pad=2),
            conv3 = L.Convolution2D(256, 384, ksize=3, pad=1),
            conv4 = L.Convolution2D(384, 384, ksize=3, pad=1),
            conv5 = L.Convolution2D(384, 256, ksize=3, pad=1),
            fc6 = L.Linear(9216, 4096),
            fc7 = L.Linear(4096, 4096),
        )
        self.fine = chainer.Chain(
            fc8ft = F.Linear(4096, 17),
        )

    def __call__(self, x, t, train=True):
        y = self.forward(x, train=train)
        self.loss = F.softmax_cross_entropy(y, t)
        self.accuracy = F.accuracy(y, t)
        return self.loss

    def forward(self, x, train=False):
        self.data = x
        self.conv1 = F.relu(self.caffe.conv1(self.data))
        self.pool1 = F.max_pooling_2d(self.conv1, ksize=3, stride=2)
        self.norm1 = F.local_response_normalization(self.pool1,
                            k=5, n=5, alpha=0.0001, beta=0.75)*np.power(5, 0.75)
        self.conv2 = F.relu(self.caffe.conv2(self.norm1))
        self.pool2 = F.max_pooling_2d(self.conv2, ksize=3, stride=2)
        self.norm2 = F.local_response_normalization(self.pool2,
                            k=5, n=5, alpha=0.0001, beta=0.75)*np.power(5, 0.75)
        self.conv3 = F.relu(self.caffe.conv3(self.norm2))
        self.conv4 = F.relu(self.caffe.conv4(self.conv3))
        self.conv5 = F.relu(self.caffe.conv5(self.conv4))
        self.pool5 = F.max_pooling_2d(self.conv5, ksize=3, stride=2)
        self.fc6 = F.dropout(F.relu(self.caffe.fc6(self.pool5)), train=train)
        self.fc7 = F.dropout(F.relu(self.caffe.fc7(self.fc6)), train=train)
        self.fc8 = self.fine.fc8ft(self.fc7)
        return self.fc8
