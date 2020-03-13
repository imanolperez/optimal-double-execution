import numpy as np
import itertools
import torch
torch.set_default_tensor_type(torch.DoubleTensor)
torch.set_default_dtype(torch.float64)


import shuffle


MAX_SHUFFLE_ORDER = 6

class Tensor(object):

    def __init__(self, dim, order):
        self.dim = dim
        self.order = order

        self.value = [torch.zeros([dim] * n) for n in range(order + 1)]



    def __getitem__(self, key):
        assert isinstance(key, tuple), "key is not a tuple"
        return self.value[len(key)][key]

    def __setitem__(self, key, value):
        assert isinstance(key, tuple), "key is not a tuple"

        self.value[len(key)][key] = value

    def __repr__(self):
        return f"Tensor({[val.tolist() for val in self.value]})"

    def __add__(self, other):
        assert self.dim == other.dim, "Dimensions do not agree"

        order = max(self.order, other.order)

        tensor = Tensor(self.dim, order)

        for n in range(order + 1):
            if n < self.order + 1:
                sum1 = self.value[n]
            else:
                sum1 = torch.zeros([self.dim] * n)

            if n < other.order + 1:
                sum2 = other.value[n]
            else:
                sum2 = torch.zeros([self.dim] * n)

            tensor.value[n] = sum1 + sum2

        return tensor

    def __mul__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            tensor = Tensor(self.dim, self.order)

            for n in range(self.order + 1):
                tensor.value[n] = self.value[n] * other

            return tensor
        else:
            return self._prod(other)

    def __rmul__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            return self * other
        else:
            return self._prod(other)

    def __sub__(self, other):
        return self + (-1.) * other

    def __neg__(self):
        return -1. * self

    def _prod(self, other):
        order = self.order + other.order

        tensor = Tensor(self.dim, order)

        for tensor1 in self.value:
            for tensor2 in other.value:
                n = len(tensor1.shape) + len(tensor2.shape)
                prod = tensordot_pytorch(tensor1, tensor2)

                assert len(prod.shape) == n, "Error: wrong shape"

                tensor.value[n] += prod

        return tensor

    def sigkeys(self):
        for n in range(self.order + 1):
            for w in itertools.product(*[list(range(self.dim)) for _ in range(n)]):
                yield w

    def sum_squares(self):
        s = 0.
        for tensor in self.value:
            s += torch.sum(tensor ** 2)

        return s

    def shuffle(self, other, max_order=MAX_SHUFFLE_ORDER):
        order = self.order + other.order
        tensor = Tensor(self.dim, order)

        for w1 in self.sigkeys():
            self_w1 = self[w1]
            if self_w1 == 0.:
                continue

            for w2 in other.sigkeys():
                if len(w1) + len(w2) > max_order:
                    continue

                other_w2 = other[w2]
                if other_w2 == 0.:
                    continue

                shuffled = shuffle.shuffle(w1, w2)
                for v in shuffled:
                    tensor[v] += self[w1] * other[w2]


        return tensor

    def dot(self, other):
        s = 0.

        order = min(self.order, other.order)
        for n in range(order + 1):

            prod = (self.value[n] * other.value[n]).flatten()

            if n == 0:
                s += prod
            else:
                s += torch.sum(prod)

        return s

    def flatten(self):
        v = []
        for val in self.value:
            v = np.r_[v, val.detach().numpy().flatten()]
        return v

def e(i, dim):
    """Returns e_i."""

    assert i < dim, "i must be less than dim"

    tensor = Tensor(dim, 1)
    tensor[(i,)] = 1.

    return tensor

def one(dim):
    tensor = Tensor(dim, 0)
    tensor[tuple()] = 1.

    return tensor

def zero(dim):
    return Tensor(dim, 0)


def sig_to_tensor(sig, dim, order):
    tensor = Tensor(dim, order)
    keys = list(tensor.sigkeys())

    for val, key in zip(sig, keys):
        tensor[key] = val

    return tensor


def tensordot_pytorch(tensor1, tensor2):
    shape = list(tensor1.shape) + list(tensor2.shape)

    for axis in tensor2.shape:
        tensor1 = tensor1.unsqueeze(-1)
    for axis in tensor1.shape:
        tensor2 = tensor2.unsqueeze(0)

    return (tensor1 * tensor2).reshape(shape)

