'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from utils import progress_bar
from torch.optim.optimizer import Optimizer, required

# from visdom import Visdom
import numpy as np
import time
import warnings


from torch.nn import _reduction as _Reduction
from torch.autograd import Function
from torch.autograd import Variable


os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# viz = Visdom()



# import cupy as np
# from torch.utils.tensorboard import SummaryWriter
# from torchsummary import summary

class SGD2(Optimizer):
    r"""Implements stochastic gradient descent (optionally with momentum).

    Nesterov momentum is based on the formula from
    `On the importance of initialization and momentum in deep learning`__.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)

    Example:
        >>> optimizer = torch.optim.SGD2(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()

    __ http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf

    .. note::
        The implementation of SGD with Momentum/Nesterov subtly differs from
        Sutskever et. al. and implementations in some other frameworks.

        Considering the specific case of Momentum, the update can be written as

        .. math::
                  v_{t+1} = \mu * v_{t} + g_{t+1} \\
                  p_{t+1} = p_{t} - lr * v_{t+1}

        where p, g, v and :math:`\mu` denote the parameters, gradient,
        velocity, and momentum respectively.

        This is in contrast to Sutskever et. al. and
        other frameworks which employ an update of the form

        .. math::
             v_{t+1} = \mu * v_{t} + lr * g_{t+1} \\
             p_{t+1} = p_{t} - v_{t+1}

        The Nesterov version is analogously modified.
    """

    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(SGD2, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SGD2, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                # print(d_p)
                # d_p = torch.tan(1.3*d_p)

                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                p.data.add_(-group['lr'], d_p)

        return loss

def _get_softmax_dim(name, ndim, stacklevel):
    # type: (str, int, int) -> int
    warnings.warn("Implicit dimension choice for {} has been deprecated. "
                  "Change the call to include dim=X as an argument.".format(name), stacklevel=stacklevel)
    if ndim == 0 or ndim == 1 or ndim == 3:
        ret = 0
    else:
        ret = 1
    return ret


def log_softmax(input, dim=None, _stacklevel=3, dtype=None):
    # type: (Tensor, Optional[int], int, Optional[int]) -> Tensor

    if dim is None:
        dim = _get_softmax_dim('log_softmax', input.dim(), _stacklevel)
    if dtype is None:
        ret = input.log_softmax(dim)
    else:
        ret = input.log_softmax(dim, dtype=dtype)
    return ret


def nll_loss(input, target, weight=None, size_average=None, ignore_index=-100,
             reduce=None, reduction='mean'):
    # type: (Tensor, Tensor, Optional[Tensor], Optional[bool], int, Optional[bool], str) -> Tensor

    if size_average is not None or reduce is not None:
        reduction = _Reduction.legacy_get_string(size_average, reduce)
    dim = input.dim()
    if dim < 2:
        raise ValueError('Expected 2 or more dimensions (got {})'.format(dim))

    if input.size(0) != target.size(0):
        raise ValueError('Expected input batch_size ({}) to match target batch_size ({}).'
                         .format(input.size(0), target.size(0)))
    if dim == 2:
        ret = torch._C._nn.nll_loss(input, target, weight, _Reduction.get_enum(reduction), ignore_index)
    elif dim == 4:
        ret = torch._C._nn.nll_loss2d(input, target, weight, _Reduction.get_enum(reduction), ignore_index)
    else:
        # dim == 3 or dim > 4
        n = input.size(0)
        c = input.size(1)
        out_size = (n,) + input.size()[2:]
        if target.size()[1:] != input.size()[2:]:
            raise ValueError('Expected target size {}, got {}'.format(
                out_size, target.size()))
        input = input.contiguous().view(n, c, 1, -1)
        target = target.contiguous().view(n, 1, -1)
        reduction_enum = _Reduction.get_enum(reduction)
        if reduction != 'none':
            ret = torch._C._nn.nll_loss2d(
                input, target, weight, reduction_enum, ignore_index)
        else:
            out = torch._C._nn.nll_loss2d(
                input, target, weight, reduction_enum, ignore_index)
            ret = out.view(out_size)
    return ret

def cross_entropy(input, target, weight=None, size_average=None, ignore_index=-100,
                  reduce=None, reduction='mean'):
    # type: (Tensor, Tensor, Optional[Tensor], Optional[bool], int, Optional[bool], str) -> Tensor

    if size_average is not None or reduce is not None:
        reduction = _Reduction.legacy_get_string(size_average, reduce)
    return nll_loss(log_softmax(input, 1), target, weight, None, ignore_index, None, reduction)

class _Loss(nn.Module):
    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super(_Loss, self).__init__()
        if size_average is not None or reduce is not None:
            self.reduction = _Reduction.legacy_get_string(size_average, reduce)
        else:
            self.reduction = reduction


class _WeightedLoss(_Loss):
    def __init__(self, weight=None, size_average=None, reduce=None, reduction='mean'):
        super(_WeightedLoss, self).__init__(size_average, reduce, reduction)
        self.register_buffer('weight', weight)


# class CrossEntropyLoss_act(Function):
#     # __constants__ = ['weight', 'ignore_index', 'reduction']
#
#     def __init__(self):
#         # super(CrossEntropyLoss_act, self).__init__(weight, size_average, reduce, reduction)
#         # self.ignore_index = ignore_index
#         self.grad_outputs = 0
#
#     def forward(self, input, target):
#         grad_outputs = 2*cross_entropy(input, target, weight=self.weight, ignore_index=self.ignore_index, reduction=self.reduction)
#
#         return grad_outputs
#         # return -torch.exp(5.0-err)
#     def backward(self, grad_outputs):
#         return input

# class CrossEntropyLoss_act(_WeightedLoss):
#
#     __constants__ = ['weight', 'ignore_index', 'reduction']
#
#     def __init__(self, weight=None, size_average=None, ignore_index=-100,
#                  reduce=None, reduction='mean'):
#         super(CrossEntropyLoss_act, self).__init__(weight, size_average, reduce, reduction)
#         self.ignore_index = ignore_index
#         self.err = 0
#
#     def forward(self, input, target):
#         # F.cross_entropy(input, target, weight=self.weight, ignore_index=self.ignore_index, reduction=self.reduction)
#
#         return cross_entropy(input, target, weight=self.weight, ignore_index=self.ignore_index, reduction=self.reduction)
#         # return -torch.exp(5.0-err)


class CrossEntropyLoss_act(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss_act, self).__init__()

    def forward(self, fc_out, label):
        one_hot_label = torch.FloatTensor(fc_out.shape[0], fc_out.shape[1]).to(device)
        one_hot_label.zero_()
        one_hot_label.scatter_(1, torch.reshape(label, (fc_out.shape[0], 1) ), 1)
        loss = one_hot_label * torch.softmax(fc_out, 1)
        # loss = 1/(torch.sum(torch.log(torch.sum(loss, 1)))/fc_out.shape[0])
        loss = -(torch.sum(torch.log(torch.sum(loss, 1)))/fc_out.shape[0])

        # loss = torch.pow(-0.12768*torch.sum(torch.log(torch.sum(loss, 1)))/fc_out.shape[0], 3)
        # loss = -torch.exp(2.0-5*loss)
        # loss = 5 * torch.atan(loss) + loss

        return loss


class Adadelta_act(Optimizer):
    """Implements Adadelta_act algorithm.

    It has been proposed in `Adadelta_act: An Adaptive Learning Rate Method`__.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        rho (float, optional): coefficient used for computing a running average
            of squared gradients (default: 0.9)
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-6)
        lr (float, optional): coefficient that scale delta before it is applied
            to the parameters (default: 1.0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)

    __ https://arxiv.org/abs/1212.5701
    """

    def __init__(self, params, lr=1.0, rho=0.9, eps=1e-6, weight_decay=0):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= rho <= 1.0:
            raise ValueError("Invalid rho value: {}".format(rho))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, rho=rho, eps=eps, weight_decay=weight_decay)
        super(Adadelta_act, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adadelta_act does not support sparse gradients')
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['square_avg'] = torch.zeros_like(p.data)
                    state['acc_delta'] = torch.zeros_like(p.data)

                square_avg, acc_delta = state['square_avg'], state['acc_delta']
                rho, eps = group['rho'], group['eps']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                square_avg.mul_(rho).addcmul_(1 - rho, grad, grad)
                std = square_avg.add(eps).sqrt_()
                delta = acc_delta.add(eps).sqrt_().div_(std).mul_(grad)
                p.data.add_(-group['lr'], delta)
                acc_delta.mul_(rho).addcmul_(1 - rho, delta, delta)

        return loss

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint_ori_notebook')
# args = parser.parse_args()
args, unknown = parser.parse_known_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint_ori_notebook epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=24)
# print(np.dot(np.linalg.pinv(np.array(trainloader)), np.array(trainloader)))
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
# testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=24)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
# net = VGG('VGG19')
# net = ResNet18()
net = ResNet18_opt_200105()

# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()
net = net.to(device)
# print(torch.__version__)
if device == 'cuda':
    # net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint_ori_notebook.
    print('==> Resuming from checkpoint_ori_notebook..')
    assert os.path.isdir('checkpoint_ori_notebook'), 'Error: no checkpoint_ori_notebook directory found!'
    checkpoint_ori_notebook = torch.load('./checkpoint_ori_notebook/ckpt_opt.pth')
    net.load_state_dict(checkpoint_ori_notebook['net'])
    best_acc = checkpoint_ori_notebook['acc']
    start_epoch = checkpoint_ori_notebook['epoch']
    
criterion = CrossEntropyLoss_act().to(device)
# criterion = nn.CrossEntropyLoss_act(weight = 1000*torch.ones(10).to(device))
# optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

# SGD3 = SGD2()
optimizer = SGD2(net.parameters(), lr=args.lr, momentum=0, weight_decay=0)
# optimizer = Adadelta_act(net.parameters(), lr=args.lr)
# parameters=net.parameters()





# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        acc = 100.*correct/total
    progress_bar(batch_idx, len(trainloader), 'Loss: %.7f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), acc, correct, total))

    return train_loss/(batch_idx+1), acc
    # print(list(net.parameters[]))
    # for i in net.state_dict():
    #      print(i, '\n')
    # optimizer_state_dict =optimizer.state_dict()
    # print('optimizer_state_dict:\n', optimizer_state_dict)
    # # print(list(net.parameters[]))
    # for name, param in net.named_parameters():
    #     if param.requires_grad:
    #         print(name)

    # module_conv1_weight = net.module.conv1.weight

    # print('module.conv1.weight:', net.module.conv1.weight)
    # # module_layer1_0_conv1_weight = net.module.layer1.0.conv1.weight
    # print('module_layer1_0_conv1_weight:', net.module.layer1.conv1.weight)
    # print('\n')

    # print('net.module.conv1.weight:\n', net.module.conv1.weight)
    # net.module.conv1.weight = net.module.conv1.weight + net.module.conv1.weight
    # print(net.module.conv1.weight)
    # weight1 = net.module.conv1.weight
    # weight1 = 2*weight1
    # print(weight1)
    # print(module.layer1.0.bn1.weight )
    # print('is_leaf:', net.module.conv1.weight.is_leaf)
    # with torch.no_grad():
    #     net.module.conv1.weight[0] = 2*net.module.conv1.weight[0]
    # print('is_leaf:', net.module.conv1.weight.is_leaf)

    # print('\n\n\n',net.module.conv1.weight.shape)

    # for name, param in net.named_parameters():
    #     print(name, '      ', param.size())
    # # print(inputs.shape)
    # summary(net, (3, 32, 32))
    #
    # dummy_input = torch.rand(13, 3, 32, 32)
    # model = net()
    # with SummaryWriter(comment='ResNet') as w:
    #     w.add_graph(net, (dummy_input,))


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    # print('is_leaf:', net.module.conv1.weight.is_leaf)
    # with torch.no_grad():
    #     net.module.conv1.weight[:] = 0 * net.module.conv1.weight[:]
    # print('is_leaf:', net.module.conv1.weight.is_leaf)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint_ori_notebook.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint_ori_notebook'):
            os.mkdir('checkpoint_ori_notebook')
        torch.save(state, './checkpoint_ori_notebook/ckpt_opt.pth')
        best_acc = acc

    return test_loss/(batch_idx+1), acc

writer = SummaryWriter()
if __name__ == '__main__':

    # for epoch in range(start_epoch, start_epoch+200):
    for epoch in range(start_epoch, start_epoch + 150):
        train_loss, train_acc = train(epoch)
        test_loss, test_acc = test(epoch)
        writer.add_scalar("Trainning Accuracy", train_acc, epoch)
        writer.add_scalar("Test Accuracy", test_acc, epoch)
        writer.add_scalar("Trainning Loss", train_loss, epoch)
        writer.add_scalar("Test Loss", test_loss, epoch)
        # viz.line([train_loss], [epoch], win='train_loss_reshape_saddle', opts=dict(title='train_loss_reshape_saddle'), update='append')
        # viz.line([train_acc], [epoch], win='train_acc_reshape_saddle', opts=dict(title='train_acc_reshape_saddle'),update='append')
        # viz.line([test_loss], [epoch], win='test_loss_reshape_saddle', opts=dict(title='test_loss_reshape_saddle'),update='append')
        # viz.line([test_acc], [epoch], win='test_acc_reshape_saddle',opts=dict(title='test_acc_reshape_saddle'), update='append')
        # time.sleep(0.5)
    # writer.close()