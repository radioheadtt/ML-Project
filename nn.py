"""
    PytTrace.nn:
    Store defined operator and Models
"""
import torch
import numpy as np
from copy import deepcopy
from utils import generate_grad, refine_para, ones_like, zeros_like,sigmoid,softmax
from tensor_ import tensor_
import init
from base import Operator, Input
# import .init
import torch.nn.functional as F
        

class Linear(Operator):
    def __init__(self, in_shape, out_shape):
        super(Linear, self).__init__()
        self._name = f"Linear layer ({in_shape} X {out_shape})"
        self.in_shape = in_shape
        self.out_shape = out_shape        
        
        self.weight = torch.ones(out_shape, in_shape)
        self.bias   = torch.ones(out_shape)
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

        x = tensor_(torch.matmul(x, self.weight.t()) + self.bias) 
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
        x = tensor_(F.relu(x))
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

class MaxPool2d(Operator):
    def __init__(self, kernel_size=(2, 2), stride=2):
        super(MyMaxPool2D, self).__init__()
        self.stride = stride
        self.kernel_size = kernel_size
        self.w_height = kernel_size[0]
        self.w_width = kernel_size[1]

    @Operator.Pass
    def __call__(self, x):
        in_height = x.size(0)
        in_width = x.size(1)

        out_height = int((in_height - self.w_height) / self.stride) + 1
        out_width = int((in_width - self.w_width) / self.stride) + 1

        out = torch.zeros((out_height, out_width))

        for i in range(out_height):
            for j in range(out_width):
                start_i = i * self.stride
                start_j = j * self.stride
                end_i = start_i + self.w_height
                end_j = start_j + self.w_width
                out[i, j] = torch.max(x[start_i: end_i, start_j: end_j])
        return out
    @Operator.Back
    def backward(self,grad,show=False):
        if show:
            print("->",self.__repr__())
        before_x=self._father.getOutput()
        
        
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
        ))
        self.weight = self.filters # align name
        self.bias = torch.zeros(
            self.out_channel
        )
        
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

class MSE(Operator):
    def __init__(self):
        super(MSE, self).__init__()
        self._name = "Mean Square Loss"
    
    @Operator.End
    def __call__(self, 
                 x : tensor_,
                 another):
        out_x = tensor_(F.mse_loss(x, another).unsqueeze(0))
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


# class HuberLoss(Operator):
#     def __init__(self):
#         super(HuberLoss, self).__init__()
#         self._name = "HuberLoss"
        
#     @Operator.End
#     def __call__(self, input, target):
#         pass
    
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

class LSTM():
    def lstm_cell_forward(xt, a_prev, c_prev, parameters):
    """
    Arguments:
    xt -- your input data at timestep "t", numpy array of shape (n_x, m).
    a_prev -- Hidden state at timestep "t-1", numpy array of shape (n_a, m)
    c_prev -- Memory state at timestep "t-1", numpy array of shape (n_a, m)
    parameters -- python dictionary containing:
                        Wf -- Weight matrix of the forget gate, numpy array of shape (n_a, n_a + n_x)
                        bf -- Bias of the forget gate, numpy array of shape (n_a, 1)
                        Wi -- Weight matrix of the update gate, numpy array of shape (n_a, n_a + n_x)
                        bi -- Bias of the update gate, numpy array of shape (n_a, 1)
                        Wc -- Weight matrix of the first "tanh", numpy array of shape (n_a, n_a + n_x)
                        bc --  Bias of the first "tanh", numpy array of shape (n_a, 1)
                        Wo -- Weight matrix of the output gate, numpy array of shape (n_a, n_a + n_x)
                        bo --  Bias of the output gate, numpy array of shape (n_a, 1)
                        Wy -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
                        by -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)
    Returns:
    a_next -- next hidden state, of shape (n_a, m)
    c_next -- next memory state, of shape (n_a, m)
    yt_pred -- prediction at timestep "t", numpy array of shape (n_y, m)
    cache -- tuple of values needed for the backward pass, contains (a_next, c_next, a_prev, c_prev, xt, parameters)
    Note: ft/it/ot stand for the forget/update/output gates, cct stands for the candidate value (c tilde),
          c stands for the memory value
    """

    # Retrieve parameters from "parameters"
    Wf = parameters["Wf"]
    bf = parameters["bf"]
    Wi = parameters["Wi"]
    bi = parameters["bi"]
    Wc = parameters["Wc"]
    bc = parameters["bc"]
    Wo = parameters["Wo"]
    bo = parameters["bo"]
    Wy = parameters["Wy"]
    by = parameters["by"]

    # Retrieve dimensions from shapes of xt and Wy
    n_x, m = xt.shape
    n_y, n_a = Wy.shape

    # Concatenate a_prev and xt
    concat = torch.zeros((n_a + n_x, m))
    concat[: n_a, :] = a_prev
    concat[n_a:, :] = xt

    # Compute values for ft, it, cct, c_next, ot, a_next
    ft = torch.sigmoid(torch.matmul(Wf, concat) + bf)
    it = torch.sigmoid(torch.matmul(Wi, concat) + bi)
    cct = torch.tanh(torch.matmul(Wc, concat) + bc)
    c_next = (ft * c_prev) + (it * cct)
    ot = torch.sigmoid(torch.matmul(Wo, concat) + bo)
    a_next = ot * torch.tanh(c_next)

    # Compute prediction of the LSTM cell
    yt_pred = torch.softmax(torch.matmul(Wy, a_next) + by)

    # store values needed for backward propagation in cache
    cache = (a_next, c_next, a_prev, c_prev, ft, it, cct, ot, xt, parameters)

    return a_next, c_next, yt_pred, cache


def lstm_cell_backward(da_next, dc_next, cache):
    """
    Arguments:
    da_next -- Gradients of next hidden state, of shape (n_a, m)
    dc_next -- Gradients of next cell state, of shape (n_a, m)
    cache -- cache storing information from the forward pass
    Returns:
    gradients -- python dictionary containing:
                        dxt -- Gradient of input data at time-step t, of shape (n_x, m)
                        da_prev -- Gradient w.r.t. the previous hidden state, numpy array of shape (n_a, m)
                        dc_prev -- Gradient w.r.t. the previous memory state, of shape (n_a, m, T_x)
                        dWf -- Gradient w.r.t. the weight matrix of the forget gate, numpy array of shape (n_a, n_a + n_x)
                        dWi -- Gradient w.r.t. the weight matrix of the update gate, numpy array of shape (n_a, n_a + n_x)
                        dWc -- Gradient w.r.t. the weight matrix of the memory gate, numpy array of shape (n_a, n_a + n_x)
                        dWo -- Gradient w.r.t. the weight matrix of the output gate, numpy array of shape (n_a, n_a + n_x)
                        dbf -- Gradient w.r.t. biases of the forget gate, of shape (n_a, 1)
                        dbi -- Gradient w.r.t. biases of the update gate, of shape (n_a, 1)
                        dbc -- Gradient w.r.t. biases of the memory gate, of shape (n_a, 1)
                        dbo -- Gradient w.r.t. biases of the output gate, of shape (n_a, 1)
    """

    # Retrieve information from "cache"
    (a_next, c_next, a_prev, c_prev, ft, it, cct, ot, xt, parameters) = cache

    # Retrieve dimensions from xt's and a_next's shape
    n_x, m = xt.shape
    n_a, m = a_next.shape

    # Compute gates related derivatives
    dot = da_next * torch.tanh(c_next) * ot * (1 - ot)
    dcct = (dc_next * it + ot * (1 - torch.square(torch.tanh(c_next))) * it * da_next) * (1 - torch.square(cct))
    dit = (dc_next * cct + ot * (1 - torch.square(torch.tanh(c_next))) * cct * da_next) * it * (1 - it)
    dft = (dc_next * c_prev + ot * (1 - np.square(np.tanh(c_next))) * c_prev * da_next) * ft * (1 - ft)

    concat = torch.cat((a_prev, xt), dim=0)

    # Compute parameters related derivatives.
    dWf = torch.dot(dft, concat.T)
    dWi = torch.dot(dit, concat.T)
    dWc = torch.dot(dcct, concat.T)
    dWo = torch.dot(dot, concat.T)
    dbf = torch.sum(dft, axis=1, keepdims=True)
    dbi = torch.sum(dit, axis=1, keepdims=True)
    dbc = torch.sum(dcct, axis=1, keepdims=True)
    dbo = torch.sum(dot, axis=1, keepdims=True)

    # Compute derivatives w.r.t previous hidden state, previous memory state and input. Use equations (15)-(17). (≈3 lines)
    da_prev = torch.dot(parameters['Wf'][:, :n_a].T, dft) + torch.dot(parameters['Wi'][:, :n_a].T, dit) + torch.dot(
        parameters['Wc'][:, :n_a].T, dcct) + torch.mm(parameters['Wo'][:, :n_a].T, dot)
    dc_prev = dc_next * ft + ot * (1 - torch.square(torch.tanh(c_next))) * ft * da_next
    dxt = torch.dot(parameters['Wf'][:, n_a:].T, dft) + torch.dot(parameters['Wi'][:, n_a:].T, dit) + torch.dot(
        parameters['Wc'][:, n_a:].T, dcct) + torch.dot(parameters['Wo'][:, n_a:].T, dot)

    # Save gradients in dictionary
    gradients = {"dxt": dxt, "da_prev": da_prev, "dc_prev": dc_prev, "dWf": dWf, "dbf": dbf, "dWi": dWi, "dbi": dbi,
                 "dWc": dWc, "dbc": dbc, "dWo": dWo, "dbo": dbo}
    
    return gradients

def lstm_forward(x, a0, parameters):
    """
    Arguments:
    x -- Input data for every time-step, of shape (n_x, m, T_x).
    a0 -- Initial hidden state, of shape (n_a, m)
    parameters -- python dictionary containing:
                        Wf -- Weight matrix of the forget gate, numpy array of shape (n_a, n_a + n_x)
                        bf -- Bias of the forget gate, numpy array of shape (n_a, 1)
                        Wi -- Weight matrix of the update gate, numpy array of shape (n_a, n_a + n_x)
                        bi -- Bias of the update gate, numpy array of shape (n_a, 1)
                        Wc -- Weight matrix of the first "tanh", numpy array of shape (n_a, n_a + n_x)
                        bc -- Bias of the first "tanh", numpy array of shape (n_a, 1)
                        Wo -- Weight matrix of the output gate, numpy array of shape (n_a, n_a + n_x)
                        bo -- Bias of the output gate, numpy array of shape (n_a, 1)
                        Wy -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
                        by -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)
    Returns:
    a -- Hidden states for every time-step, numpy array of shape (n_a, m, T_x)
    y -- Predictions for every time-step, numpy array of shape (n_y, m, T_x)
    caches -- tuple of values needed for the backward pass, contains (list of all the caches, x)
    """

    # Initialize "caches", which will track the list of all the caches
    caches = []

    # Retrieve dimensions from shapes of x and parameters['Wy']
    n_x, m, T_x = x.shape
    n_y, n_a = parameters["Wy"].shape

    # initialize "a", "c" and "y" with zeros
    a = torch.zeros((n_a, m, T_x))
    c = a
    y = torch.zeros((n_y, m, T_x))

    # Initialize a_next and c_next
    a_next = a0
    c_next = torch.zeros(a_next.shape)

    # loop over all time-steps
    for t in range(T_x):
        # Update next hidden state, next memory state, compute the prediction, get the cache
        a_next, c_next, yt, cache = lstm_cell_forward(x[:, :, t], a_next, c_next, parameters)
        # Save the value of the new "next" hidden state in a
        a[:, :, t] = a_next
        # Save the value of the prediction in y
        y[:, :, t] = yt
        # Save the value of the next cell state
        c[:, :, t] = c_next
        # Append the cache into caches
        caches.append(cache)

    # store values needed for backward propagation in cache
    caches = (caches, x)

    return a, y, c, caches


def lstm_backward(da, caches):
    """
    Arguments:
    da -- Gradients w.r.t the hidden states, numpy-array of shape (n_a, m, T_x)
    dc -- Gradients w.r.t the memory states, numpy-array of shape (n_a, m, T_x)
    caches -- cache storing information from the forward pass (lstm_forward)
    Returns:
    gradients -- python dictionary containing:
                        dx -- Gradient of inputs, of shape (n_x, m, T_x)
                        da0 -- Gradient w.r.t. the previous hidden state, numpy array of shape (n_a, m)
                        dWf -- Gradient w.r.t. the weight matrix of the forget gate, numpy array of shape (n_a, n_a + n_x)
                        dWi -- Gradient w.r.t. the weight matrix of the update gate, numpy array of shape (n_a, n_a + n_x)
                        dWc -- Gradient w.r.t. the weight matrix of the memory gate, numpy array of shape (n_a, n_a + n_x)
                        dWo -- Gradient w.r.t. the weight matrix of the save gate, numpy array of shape (n_a, n_a + n_x)
                        dbf -- Gradient w.r.t. biases of the forget gate, of shape (n_a, 1)
                        dbi -- Gradient w.r.t. biases of the update gate, of shape (n_a, 1)
                        dbc -- Gradient w.r.t. biases of the memory gate, of shape (n_a, 1)
                        dbo -- Gradient w.r.t. biases of the save gate, of shape (n_a, 1)
    """

    # Retrieve values from the first cache (t=1) of caches.
    (caches, x) = caches
    (a1, c1, a0, c0, f1, i1, cc1, o1, x1, parameters) = caches[0]

    # Retrieve dimensions from da's and x1's shapes
    n_a, m, T_x = da.shape
    n_x, m = x1.shape

    # initialize the gradients with the right sizes
    dx = np.zeros((n_x, m, T_x))
    da0 = np.zeros((n_a, m))
    da_prevt = np.zeros(da0.shape)
    dc_prevt = np.zeros(da0.shape)
    dWf = np.zeros((n_a, n_a + n_x))
    dWi = np.zeros(dWf.shape)
    dWc = np.zeros(dWf.shape)
    dWo = np.zeros(dWf.shape)
    dbf = np.zeros((n_a, 1))
    dbi = np.zeros(dbf.shape)
    dbc = np.zeros(dbf.shape)
    dbo = np.zeros(dbf.shape)

    # loop back over the whole sequence
    for t in reversed(range(T_x)):
        # Compute all gradients using lstm_cell_backward
        gradients = lstm_cell_backward(da[:, :, t], dc_prevt, caches[t])
        # Store or add the gradient to the parameters' previous step's gradient
        dx[:, :, t] = gradients["dxt"]
        dWf += gradients["dWf"]
        dWi += gradients["dWi"]
        dWc += gradients["dWc"]
        dWo += gradients["dWo"]
        dbf += gradients["dbf"]
        dbi += gradients["dbi"]
        dbc += gradients["dbc"]
        dbo += gradients["dbo"]
    # Set the first activation's gradient to the backpropagated gradient da_prev.
    da0 = gradients["da_prev"]

    # Store the gradients in a python dictionary
    gradients = {"dx": dx, "da0": da0, "dWf": dWf, "dbf": dbf, "dWi": dWi, "dbi": dbi,
                 "dWc": dWc, "dbc": dbc, "dWo": dWo, "dbo": dbo}

    return gradients 
    
if __name__ == "__main__":
    print("store operators and models")