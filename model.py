from mxnet.gluon import nn,loss

class MnistModel(nn.Block):

    def __init__(self,**kwargs):
        super(MnistModel,self).__init__(**kwargs)
        self.blk = nn.Sequential()
        self.blk.add(nn.Conv2D(channels=6, kernel_size=5, activation='relu'),
                      nn.MaxPool2D(pool_size=2, strides=2),
                      nn.Conv2D(channels=16, kernel_size=3, activation='relu'),
                      nn.MaxPool2D(pool_size=2, strides=2),
                      nn.Flatten(),
                      nn.Dense(120, activation="relu"),
                      nn.Dense(84, activation="relu"),
                      nn.Dense(10))
        self.softmax = loss.SoftmaxCrossEntropyLoss()

    def forward(self,x):
        output=self.blk(x)
        return output

    def acc(self,output,label):
        return (output.argmax(axis=1) == label.astype('float32')).mean().asscalar()

    def loss(self,output,label):
        loss_result=self.softmax(output,label)
        return loss_result