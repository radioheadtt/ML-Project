from typing import Type
from collections import defaultdict
import torch
import math
from utils import ones_like,zeros_like

class _RequiredParameter(object):
    def __repr__(self):
        return "<required parameter>"

required =_RequiredParameter()
class Adam:
    def __init__(self,params,lr=1e-3,betas=(0.9,0.999),eps=1e-8):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <=betas[0]<1.0:
            raise ValueError("Invalid beta 0 parameter:{}".format(betas[0]))
        if not 0.0<=betas[1]<1.0:
            raise ValueError("Invalid bata 1 parameter:{}".format(betas[1]))
        self.defaults=dict(lr=lr,betas=betas,eps=eps)
        self.state=defaultdict(dict)
        self.param_groups=[]
        param_groups=list(params)
        if len(param_groups)==0:
            raise ValueError
        if not isinstance(param_groups[0],dict):
            param_groups=[{"params":param_groups}]
        for param_group in param_groups:
            self.add_param_group(param_group)
    def zero_grad(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    p.grad.zero_()
    def add_param_group(self,param_group):
        params=param_group['params']
        if isinstance(params,torch.Tensor):
            param_group['params']=[params]
        elif isinstance(params,set):
            raise TypeError()
        else:
            param_group['params']=list(params)
        for param in param_group['params']:
            if not isinstance(param,torch.Tensor):
                raise TypeError()
            if not param.is_leaf:
                raise ValueError()
        for name,default in self.defaults.items():
            if default is required and name not in param_group:
                raise ValueError
            else:
                param_group.setdefault(name,default)
        param_set=set()
        for group in self.param_groups:
            param_set.update(set(group['param']))
        if not param_set.isdisjoint(set(param_group['params'])):
            raise ValueError()
        self.param_groups.append(param_group)
    def step(self,closure=None):
        loss=None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad=p.grad.data
                state=self.state[p]
                if len(state)==0:
                    state['step']=0
                    state['exp_avg']=zeros_like(p.data).cuda()
                    state['exp_avg_sq']=zeros_like(p.data).cuda()
                state['step']+=1
                beta1,beta2=group['betas']
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                bias_correction1=1-beta1**state['step']
                bias_correction2=1-beta2**state['step']
                
                exp_avg.mul_(beta1).add_(1-beta1,grad)
                exp_avg_sq.mul_(beta2).addcmul_(1-beta2,grad,grad)
                
                denom=(exp_avg_sq.sqrt()/math.sqrt(bias_correction2)).add_(group['eps'])
                step_size=group['lr']/bias_correction1
                
                p.data.addcdiv_(-step_size,exp_avg,denom)
        return loss