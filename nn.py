"""
    PytTrace.nn:
    Store defined operator and Models
"""
import torch
import numpy as np
from copy import deepcopy
from utils import generate_grad,generate_grad_1d, refine_para, ones_like, zeros_like
from tensor_ import tensor_
import init
from base import Operator, Input
# import .init
import torch.nn.functional as F
from base import Tensor_
from torch import Tensor
from optim import Adam
class Linear(Operator):
    def __init__(self, in_shape, out_shape):
        super(Linear, self).__init__()
        self._name = f"Linear layer ({in_shape} X {out_shape})"
        self.in_shape = in_shape
        self.out_shape = out_shape        
        
        self.weight = torch.ones(out_shape, in_shape).cuda()
        self.bias   = torch.ones(out_shape).cuda()
        self._init_weight()
        
        
    def _init_weight(self):
        fan_in, _ = init.calculate_fan(self.weight)
        init.kaiming_uniform_(self.weight)
        init.kaiming_uniform_(self.bias, k=fan_in)
    
    @Operator.Pass
    def __call__(self, 
                 x : tensor_):
        try:
            assert x.size()[-1] == self.in_shape
        except AssertionError:
            raise TypeError(f"the Linear layer {self.in_shape}X{self.out_shape} can't accept {x.size()}")

        x = torch.matmul(x, self.weight.t()) + self.bias
        return x
        
    @Operator.Back
    def backward(self, grad, show=False):
        if show:
            print("↘︎", self.__repr__())
        assert grad.size()[-1] == self.out_shape
        b_grad = 0.0
        W_grad = 0.0
        before_x = self._father.getOutput()
        for batch_i in range(grad.size()[0]):
            grad_i = grad[batch_i]
            x = before_x[batch_i]
            b_grad += grad_i
            W_grad += grad_i.unsqueeze(-1)*(x.repeat(self.out_shape, 1))
        #print(b_grad.size(), W_grad.size())
        #print(W_grad)
        self.weight.grad = W_grad.float()
        self.bias.grad   = b_grad.float()
        next_grad = torch.matmul(grad, self.weight)
        return next_grad
    
    def parameters(self):
        for para in [self.weight, self.bias]:
            yield para
    
class ReLU(Operator):
    def __init__(self):
        super(ReLU, self).__init__()
        self._name = "ReLU()"
        
    @Operator.Pass
    def __call__(self, 
                 x : tensor_):
        x = F.relu(x)
        return x
    
    @Operator.Back
    def backward(self, grad, show=False):
        if show:
            print("↘︎", self.__repr__())
        before_x = self._father.getOutput()
        grad[before_x <= 0] = 0.
        return grad

class Gather_last(Operator):
    def __init__(self, where):
        super(Gather_last, self).__init__()
        self._name = "Gather()"
        self.where = torch.Tensor(where.float()).long()
        self.index = torch.Tensor(np.arange(len(where))).long()
        
    
    @Operator.Pass
    def __call__(self, x):
        return tensor_(x[self.index, self.where].unsqueeze(-1))
    
    @Operator.Back
    def backward(self, grad, show=False):
        if show:
            print("↘︎", self.__repr__())       
        before_x = self._father.getOutput()
        indice_grad = zeros_like(before_x)
        indice_grad[self.index, self.where] = grad.squeeze()
        return indice_grad

        
class Conv2d(Operator):
    """
        simply conv2d
        Only support constant(0) padding
    """
    def __init__(self,
                 in_channel,
                 filters,
                 kernel_size,
                 stride=1,
                 padding=0):
        super(Conv2d, self).__init__()
        self.in_channel = in_channel
        self.out_channel = filters
        self.kernel = refine_para(kernel_size)
        self.stride = refine_para(stride)
        self.padding= refine_para(padding)
        self._name = f"Conv2d {in_channel} -> {filters}"
        self.filters = torch.zeros((
            self.out_channel,
            self.in_channel,
            self.kernel[0],
            self.kernel[1]
        )).cuda()
        self.weight = self.filters # align name
        self.bias = torch.zeros(
            self.out_channel
        ).cuda()
        
    def _init_weight(self):
        pass
    
    def parameters(self):
        for para in [self.filters, self.bias]:
            yield para
    
    @Operator.Pass
    def __call__(self, x):
        try:
            assert len(x.shape) == 4
        except AssertionError:
            raise AssertionError(f"expected 4-dim tracer, got {x.shape}")
        x = F.conv2d(x, self.filters, self.bias, stride=self.stride, padding=self.padding)
        return tensor_(x)
    @Operator.Back
    def backward(self, grad, show=False):
        
        try:
            assert grad.shape == self._out_x.shape
        except AssertionError:
            raise AssertionError(f"expected grad({self._out_x.shape}), got grad({grad.shape})")
        imgs = self._father.getOutput()
        filter_grad = zeros_like(self.filters)
        bias_grad = zeros_like(self.bias)
        next_grad = generate_grad(imgs, 
                                  self.filters, self.bias, self.kernel, self.stride, self.padding,
                                  filter_grad, bias_grad, grad)
        self.filters.grad = filter_grad
        self.bias.grad = bias_grad
        if show:
            print("↘︎", self.__repr__(), f"  GRAD:{grad.shape} -> {next_grad.shape}")
        return next_grad

class Conv1d(Operator):
    def __init__(self,
                 in_channel,
                 filters,
                 kernel_size,
                 stride=1,):
        super(Conv1d, self).__init__()
        self.in_channel = in_channel
        self.out_channel = filters
        self.kernel=kernel_size
        self._name = f"Conv1d {in_channel} -> {filters}"
        self.stride=stride
        self.filters = torch.zeros((
            self.out_channel,
            self.in_channel,
            self.kernel
        )).cuda()
        init.kaiming_uniform_(self.filters)

        self.weight = self.filters # align name
        self.bias = torch.zeros(
            self.out_channel
        ).cuda()
        init.kaiming_uniform_(self.bias)
        
    def _init_weight(self):
        pass
    
    def parameters(self):
        for para in [self.filters]:
            yield para
    
    @Operator.Pass
    def __call__(self, x):
        try:
            assert len(x.shape) == 3
        except AssertionError:
            raise AssertionError(f"expected 3-dim tensor, got {x.shape}")
        print(type(self.filters))
        x = F.conv1d(x,self.filters, bias=self.bias, stride=self.stride).cuda()
        return x
    
    @Operator.Back
    def backward(self, grad, show=False):
        try:
            assert grad.shape == self._out_x.shape
        except AssertionError:
            raise AssertionError(f"expected grad({self._out_x.shape}), got grad({grad.shape})")
        if show:
            print("↘︎", self.__repr__())
        input = self._father.getOutput()
        filter_grad = zeros_like(self.filters).cuda()
        bias_grad = zeros_like(self.bias).cuda()
        next_grad = generate_grad_1d(input, 
                                  self.filters, self.bias, self.kernel, self.stride,
                                  filter_grad, bias_grad, grad)
        self.filters.grad = filter_grad
        self.bias.grad = bias_grad
        if show:
            print("↘︎", self.__repr__(), f"  GRAD:{grad.shape} -> {next_grad.shape}")
        return next_grad
class MaxPooling1d(Operator):
    def __init__(self,width):
        super(MaxPooling1d,self).__init__()
        self.width=width
    @Operator.Pass
    def __call__(self,x):
        output = torch.zeros((x.shape[0],x.shape[1],x.shape[2]//self.width)).cuda()
        pool_w = x.shape[2] // self.width
        for i in range(0, x.shape[2] - self.width + 1, self.width):
            M = x[:, :, i : i + self.width]
            output[:,:,i // self.width] = M.max(dim=2).values 
        return Tensor_(output.to('cpu')).cuda()
    @Operator.Back
    def backward(self,grad,show=False):
        if show:
            print("↘︎", self.__repr__())
        input = self._father.getOutput()
        next_grad=torch.zeros(input.shape).cuda()
        for m in range(input.shape[0]):
            for n in range(input.shape[1]):
                for i in range(0, input.shape[2] - self.width + 1, self.width):
                    M = (input[m, n, i : i + self.width]).max()
                    for j in range(self.width):
                        if input[m][n][i+j]==M:
                            next_grad[m][n][i+j]+=1
        return next_grad
class GlobalMaxPooling1d(Operator):
    def __init__(self):
        super(GlobalMaxPooling1d,self).__init__()
    @Operator.Pass
    def __call__(self,x):
        output = torch.zeros(x.shape[0:2]).cuda()
        M = x[:, :, :]
        output = M.max(dim=2)
        return output.values
    @Operator.Back
    def backward(self,grad,show=False):
        if show:
            print("↘︎", self.__repr__())
        input = self._father.getOutput()
        next_grad=torch.zeros(input.shape)
        for m in range(input.shape[0]):
            for n in range(input.shape[1]):
                M = (input[m, n, :]).max()
                for j in range(input.shape[2]):
                    if input[m][n][j]==M:
                        next_grad[m][n][j]+=1
        return next_grad
class MSE(Operator):
    def __init__(self):
        super(MSE, self).__init__()
        self._name = "Mean Square Loss"
    
    @Operator.End
    def __call__(self, 
                 x : tensor_,
                 another):
        out_x = F.mse_loss(x, another).unsqueeze(0)
        self._another = another
            
        return out_x
    
    @Operator.Back
    def backward(self, grad=1., show=False):
        if show:
            print("↘︎", self.__repr__())
        before_x = self._father.getOutput()
        next_grad = 2*(before_x - self._another)/float(self._another.size()[0])/float(self._another.size()[1])
        next_grad = next_grad*grad
        return next_grad
class CrossEntropy(Operator):
    def __init__(self):
        super(CrossEntropy,self).__init__()
        self._name="CrossEntropy"
    

    
class Sum(Operator):
    def __init__(self):
        super(Sum, self).__init__()
        self._name = "Sum()"
        
    @Operator.End
    def __call__(self, x):
        return tensor_(torch.sum(x).unsqueeze(-1))
    
    @Operator.Back
    def backward(self, grad=1., show=False):
        if show:
            print("↘︎", self.__repr__())
        before_x = self._father.getOutput()
        next_grad = ones_like(before_x)*grad
        next_grad = next_grad
        return next_grad
  
class Model:
    def __init__(self):
        self._name = "Sequence"
        self.ops = self.construct()
        self.check()
        self.input_layer = Input()
    
    def construct(self):
        raise NotImplementedError
    
    def check(self):
        for op in self.ops:
            if not isinstance(op, Operator) and not isinstance(op, Model):
                raise TypeError("please use Operator class!!")
            
    def forward(self, 
                 x : tensor_):
        x  = self.input_layer(x)
        for op in self.ops:
            x = op(x)
            # print(x)
        return x
    
    def __call__(self, x):
        x = self.input_layer(x)
        return self.forward(x)
    
    def parameters(self):
        for op in self.ops:
            for para in op.parameters():
                if para is not None:
                    yield para
                    
    def load_seq_list(self, seq_list):
        for old, new in zip(self.parameters(), seq_list):
            try:
                assert old.shape == new.shape
                old.data = deepcopy(new.data)
            except AssertionError:
                raise RuntimeError(f"model loading a wrong weight, expect {new.shape}, but got {old.shape}")
    
    def seq_list(self):
        return list(self.parameters())
    
    def __repr__(self, pre=None):
        names = [f"{self._name}("]
        pre = "" if pre is None else pre
        for i, op in enumerate(self.ops):
            this_pre = pre
            if isinstance(op, Model):
                this_pre = pre + '    '
            prefix= pre+f"    ({i}):"
            names.append(prefix + op.__repr__(pre=this_pre))
        names.append(pre+")")
        return '\n'.join(names)
    
    def eval(self):
        for op in self.ops:
            op.eval()
        return self
    
    def train(self):
        for op in self.ops:
            op.train()
        return self
    
    def set_name(self, name):
        self._name = name
        return self
           
class Sequential(Model):
    def __init__(self, *args):
        self.args = args
        super(Sequential, self).__init__()
        
    def construct(self):
        return self.args




    
if __name__ == "__main__":
    print("store operators and models")
    x=Tensor_([[[1,2,3,9,1],[4,5,6,11,2],[7,8,9,5,3],[10,11,12,1,0],[13,14,15,31,7],[16,17,18,9,91]]])
    x=Tensor_(x.permute(0,2,1)).cuda()
    x=x.cuda_required()
    input=x
    y=Tensor_([[0,0,1]]).cuda()
    class CNN(Model):
        def __init__(self):
            self.fc = Sequential(Conv1d(5,32,3),ReLU(),GlobalMaxPooling1d(),Linear(32,3))
            super(CNN, self).__init__()
            self.set_name('Net')
            
        def construct(self):
            return [self.fc]
    
        def forward(self, obs):
            actions = self.fc(obs)
            return actions
    net=CNN()
    criterion=MSE()
    optimizer=Adam(net.parameters())
    for i in range(500):
        print(i)
        optimizer.zero_grad()
        outputs=net(input)
        loss=criterion(outputs,y)
        loss.backward()
        print(loss)
        optimizer.step()
    