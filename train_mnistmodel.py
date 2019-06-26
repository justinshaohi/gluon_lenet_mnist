import os
import time
import argparse

from mxnet import ndarray,init,gpu,autograd,gluon
from model import *

os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT']='0'

parser = argparse.ArgumentParser()
parser.add_argument('-t',action='store_true',help='declare to train')
parser.add_argument('-p',action='store_true',help='declare to test model predict')

batch_size=256


def gen_train_data():
    data_transforms = gluon.data.vision.transforms.Compose(
        [gluon.data.vision.transforms.ToTensor(), gluon.data.vision.transforms.Normalize(0.13, 0.31)])

    mnist_train = gluon.data.vision.datasets.MNIST(train=True)
    mnist_train = mnist_train.transform_first(data_transforms)
    train_data = gluon.data.DataLoader(dataset=mnist_train, batch_size=batch_size, shuffle=True, num_workers=8)

    mnist_valid = gluon.data.vision.datasets.MNIST(train=False)
    mnist_valid = mnist_valid.transform_first(data_transforms)
    valid_data = gluon.data.DataLoader(dataset=mnist_valid, batch_size=batch_size, shuffle=True, num_workers=8)

    return train_data,valid_data

def train(net,epoch_num,train_data,valid_data):
    net.initialize(init=init.Xavier(), ctx=gpu())
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.1})
    for epoch in range(epoch_num):
        train_loss, train_acc, valid_acc = 0., 0., 0.
        tic = time.time()
        for data, label in train_data:
            data = data.as_in_context(gpu())
            label = label.as_in_context(gpu())
            with autograd.record():
                output = net.forward(data)
                loss = net.loss(output, label)
            loss.backward()
            trainer.step(batch_size)
            train_loss = train_loss + loss.mean().asscalar()
            train_acc = train_acc + net.acc(output, label)

        for _valid_data, _valid_label in valid_data:
            _valid_data = _valid_data.as_in_context(gpu())
            _valid_label = _valid_label.as_in_context(gpu())
            valid_output = net.forward(_valid_data)
            valid_acc = valid_acc + net.acc(valid_output, _valid_label)

        print("Epoch %d: loss %.3f, train acc %.3f, test acc %.3f, in %.1f sec" % (
            epoch, train_loss / len(train_data), train_acc / len(train_data),
            valid_acc / len(valid_data), time.time() - tic))
    net.save_parameters('lenet_mnist.params')


def predict(net):
    net.load_parameters('lenet_mnist.params',ctx=gpu())
    transformer = gluon.data.vision.transforms.Compose([
        gluon.data.vision.transforms.ToTensor(),
        gluon.data.vision.transforms.Normalize(0.13, 0.31)])
    mnist_valid = gluon.data.vision.datasets.MNIST(train=False)
    X, y = mnist_valid[:10]
    test_data=ndarray.ones((10,1,28,28))
    for i in range(X.shape[0]):
        x = transformer(X[i])
        test_data[i]=x
    test_data=test_data.as_in_context(gpu())
    preds=net.forward(test_data)

    result=preds.argmax(axis=1).astype('int32').asnumpy()
    print 'predict result %s,lable %s'%(str(result),str(y))




if __name__=='__main__':
    args = parser.parse_args()

    net = MnistModel()
    if args.t:
        train_data,valid_data=gen_train_data()
        train(net,50,train_data,valid_data)
    elif args.p:
        predict(net)
    else:
        print 'not statement -t or -p'



