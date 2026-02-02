import numpy as np
import torch
import random
from torchvision import datasets
from matplotlib import pyplot as plt
import faulthandler
#faulthandler.enable()
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


train_dataset = datasets.MNIST(root='./data', train=True, download=False)
test_dataset = datasets.MNIST(root='./data', train=False, download=False)
data = train_dataset.data.numpy()
targets = train_dataset.targets.numpy()

batch_size = 60
epochs = 20
step_size = .01


def w_init(size, w='He Norm'):
    if w == 'He Norm':
        return np.random.normal(scale=np.sqrt(2/size[0]), size=size)
    if w == 'He Uni':
        limit = np.sqrt(6/size[0])
        return np.random.uniform(low=-limit, high=limit, size=size)


def b_init(size, b='Small'):
    if b == 'Zero':
        return np.zeros(size)
    if b == 'Small':
        return np.full(size, .1)


class Tensor:
    def __init__(self, value):
        if isinstance(value, np.ndarray):
            self.value = value.astype(np.float32)
        else:
            self.value = np.array(value, dtype=np.float32)
        # structure
        self.graph = {0: None}
        # values at graph indices
        self.node = {0: self.value}
        # computations
        self.comp = {0: None}
        self.record = True
   
    def detach(self):
        self.record = False
   
    # only input tensors to argument
    def sum(self, addend, record=True):
        temp = Tensor(np.add(self.value, addend.value))
        if record and self.record:
            indices = np.array([max(self.graph), max(addend.graph) + len(self.graph)])
            temp.graph = self.graph | {len(self.graph) + k : (len(self.graph) + v) if v is not None else None for k,v in addend.graph.items()}
            temp.node = self.node | {len(self.graph) + k : v for k,v in addend.node.items()}
            temp.comp = self.comp | {len(self.comp) + k : v if v!=None else None for k,v in addend.comp.items()}
            temp.graph[max(temp.graph)+1] = indices
            temp.node[max(temp.node)+1] = temp.value
            temp.comp[max(temp.comp)+1] = 'add'
        return temp


    # can take scalar or ndarray or tensor
    def prod(self, factor, record=True):
        if isinstance(factor, int):
            factor = Tensor(np.full_like(self.value,factor))
        if isinstance(factor, np.ndarray):
            factor = Tensor(factor)
        temp = Tensor(np.multiply(self.value, factor.value))
        if record and self.record:
            indices = np.array([max(self.graph), max(factor.graph) + len(self.graph)])
            temp.graph = self.graph | {len(self.graph) + k : (len(self.graph) + v) if v is not None else None for k,v in factor.graph.items()}
            temp.node = self.node | {len(self.graph) + k : v for k,v in factor.node.items()}
            temp.comp = self.comp | {len(self.comp) + k : v if v!=None else None for k,v in factor.comp.items()}
            temp.graph[max(temp.graph)+1] = indices
            temp.node[max(temp.node)+1] = temp.value
            temp.comp[max(temp.comp)+1] = 'multiply'
        return temp
   
    def relu(self, record=True):
        temp = Tensor(np.maximum(0,self.value))
        if record and self.record:
            temp.graph = self.graph
            temp.node = self.node
            temp.comp = self.comp
            temp.graph[max(temp.graph)+1] = max(self.graph)
            temp.node[max(temp.node)+1] = temp.value
            temp.comp[max(temp.comp)+1] = 'relu'
        return temp
   
    def softmax(self, axis=None, record=True):
        if axis is None:
            axis = len(np.shape(self.graph))-1
        values = np.exp(self.value)
        total = np.sum(values, axis=axis, keepdims=True)
        temp = Tensor(np.divide(values,total))
        if record and self.record:
            temp.graph = self.graph
            temp.node = self.node
            temp.comp = self.comp
            temp.graph[max(temp.graph)+1] = max(self.graph)
            temp.node[max(temp.node)+1] = temp.value
            temp.comp[max(temp.comp)+1] = 'softmax'
        return temp
   
    # only input tensors to argument
    def dot(self, vector, record=True):
        temp = Tensor(np.array(self.value) @ np.array(vector.value))
        if record and self.record:
            indices = np.array([max(self.graph), max(vector.graph) + len(self.graph)])
            temp.graph = self.graph | {len(self.graph) + k : (len(self.graph) + v) if v is not None else None for k,v in vector.graph.items()}
            temp.node = self.node | {len(self.graph) + k : v for k,v in vector.node.items()}
            temp.comp = self.comp | {len(self.comp) + k : v if v!=None else None for k,v in vector.comp.items()}
            temp.graph[max(temp.graph)+1] = indices
            temp.node[max(temp.node)+1] = temp.value
            temp.comp[max(temp.comp)+1] = 'dot prod'
        return temp


class Network:
    def __init__(self, LR=.01):
        self.LR = LR


    # dont change layer or L in outside calls
    def backprop(self, pred, target, loss='L2', layer=-1, L=None):
        if layer is not None:    
            pred.detach()
            if layer == -1:
                layer = max(pred.graph)
            if L is None:
                if (loss=='CrossEntropy' and pred.comp[layer]=='softmax'):
                    L = pred.sum(target.prod(-1))
                    self.backprop(pred, target, layer=pred.graph[layer], L=L)
            elif pred.comp[layer]=='relu':
                grad = (pred.node[layer] > 0).astype(int)
                L = L.prod(Tensor(grad))
                self.backprop(pred, target, layer=pred.graph[layer], L=L)
            elif pred.comp[layer]=='add':
                x,y = pred.graph[layer]
                x,y = pred.node[x], pred.node[y]
                dx = L.prod(np.ones_like(x))
                dy = L.prod(np.ones_like(y))
                gx,gy = dx.value, dy.value
                gx,gy = (np.mean(gx, axis=0) if gx.ndim>0 else gx), (np.mean(gy, axis=0) if gy.ndim>0 else gy)
                x -= gx * self.LR
                y -= gy * self.LR
                self.backprop(pred, target, layer=pred.graph[layer][0], L=dx)
                self.backprop(pred, target, layer=pred.graph[layer][1], L=dy)
            elif pred.comp[layer]=='multiply':
                x,y = pred.graph[layer]
                x,y = pred.node[x], pred.node[y]
                dx = L.prod(y)
                dy = L.prod(x)
                gx,gy = dx.value, dy.value
                gx,gy = (np.mean(gx, axis=0) if gx.ndim>0 else gx), (np.mean(gy, axis=0) if gy.ndim>0 else gy)
                x -= gx * self.LR
                y -= gy * self.LR
                self.backprop(pred, target, layer=pred.graph[layer][0], L=dx)
                self.backprop(pred, target, layer=pred.graph[layer][1], L=dy)
            elif pred.comp[layer]=='dot prod':
                x,y = pred.graph[layer]
                x,y = pred.node[x], pred.node[y]
                dx = L.dot(Tensor(y.T))
                dy = Tensor(x.T).dot(L)
                gx,gy = dx.value, dy.value
                gx,gy = (np.mean(gx, axis=0) if gx.ndim>0 else gx), (np.mean(gy, axis=0) if gy.ndim>0 else gy)
                x -= gx * self.LR
                y -= gy * self.LR
                self.backprop(pred, target, layer=pred.graph[layer][0], L=dx)
                self.backprop(pred, target, layer=pred.graph[layer][1], L=dy)

    def loss(self, prob, label, loss='L2'):
        if loss == 'CrossEntropy':
            #print(prob.value)
            #print(label.value)
            true = prob.value[np.arange(len(label.value))][label.value.astype(int)]
            return np.mean(-np.log(true))

class Linear(Network):
    # input: tensor, n2: int
    def __init__(self, n1, n2):
        self.weights = Tensor(w_init((n1,n2)))
        self.biases = Tensor(b_init((n2)))
       
    # input: vector or batch
    def __call__(self, input):
        temp = input.dot(self.weights)
        return temp.sum(self.biases)


class Relu(Network):
    def __init__(self):
        pass
       
    def __call__(self, input):
        return input.relu()


class Softmax(Network):
    def __init__(self):
        pass
   
    def __call__(self, input):
        return input.softmax()


class FCN(Network):
    # undordered
    def __init__(self):
        super().__init__(LR=step_size)
        self.fc1 = Linear(784, 256)
        self.relu = Relu()
        self.fc2 = Linear(256, 10)
        self.softmax = Softmax()
   
    # only input tensors
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x


myNet = FCN()
inputs, labels = (zip(*[(data[i], targets[i]) for i in np.random.permutation(60000)]))
inputs = np.array(inputs)/255
labels = np.array(labels)
'''
a = Tensor([1,2,3])
b = Tensor([4,5,6])
c = a.sum(b)
d = Tensor([-1,0,1])
e = c.prod(d)
f = e.relu()
print(f.node)
print(f.graph)
'''


for epoch in range(epochs):
    print(epoch)
    batches = inputs.reshape(-1, batch_size, 784)
    batches_l = labels.reshape(-1, batch_size)
   
    for batch, label in zip(batches, batches_l):
        batch = Tensor(batch)
        label = Tensor(label)
        pred = myNet.forward(batch)
        #print(pred.value)
        target = Tensor([[1 if label.value[b]==i else 0 for i in range(10)] for b in range(batch_size)])
        myNet.backprop(pred, target, loss='CrossEntropy')
        print(myNet.loss(pred, label, loss='CrossEntropy'))


#plt.imshow(number, cmap='Grays')
#plt.show()