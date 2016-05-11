import chainer
import chainer.functions as F
import chainer.links as L

class CNN(chainer.Chain):

    def __init__(self):
        super(CNN, self).__init__(
            conv1 = L.Convolution2D(3, 32, ksize=5, pad=2),
            conv2 = L.Convolution2D(32, 64, ksize=3, pad=1),
            conv3 = L.Convolution2D(64, 128, ksize=3, pad=1),
            conv4 = L.Convolution2D(128, 128, ksize=3, pad=1),
            fc5 = L.Linear(8192, 1024),
            fc6 = L.Linear(1024, 10),
        )

    def clear(self):
        self.loss = None
        self.accuracy = None

    def __call__(self, x, t, train=True):
        self.clear()
        y = self.forward(x, train=train)
        self.loss = F.softmax_cross_entropy(y, t)
        self.accuracy = F.accuracy(y, t)
        return self.loss

    def forward(self, x, train=False):
        self.h_input = x
        self.h_conv1 = F.relu(self.conv1(self.h_input))
        self.h_conv2 = F.relu(self.conv2(self.h_conv1))
        self.h_pool2 = F.max_pooling_2d(self.h_conv2, ksize=2, stride=2)
        self.h_conv3 = F.relu(self.conv3(self.h_pool2))
        self.h_conv4 = F.relu(self.conv4(self.h_conv3))
        self.h_pool4 = F.max_pooling_2d(self.h_conv4, ksize=2, stride=2)
        self.h_fc5 = F.dropout(F.relu(self.fc5(self.h_pool4)), train=train)
        self.h_fc6 = self.fc6(self.h_fc5)
        return self.h_fc6
