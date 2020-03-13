import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from time import time
from scipy import optimize

import torch
from torch import optim
torch.set_default_tensor_type(torch.DoubleTensor)
torch.set_default_dtype(torch.float64)

import iisignature

import tensor_algebra as ta

class Optimiser:
    """Solves the double-execution problem with signatures."""

    def __init__(self, ES, N, Lambda1, Lambda2, q0, alpha1, alpha2,
                 phi1, phi2, s0=1, dim=2, n_orders=100, **kwargs):
        self.params = {
                        "Lambda1":    Lambda1,
                        "Lambda2":    Lambda2,
                        "N":          N,
                        "q0":         q0,
                        "alpha1":     alpha1,
                        "alpha2":     alpha2,
                        "phi1":       phi1,
                        "phi2":       phi2,
                        "s0":         s0,
                        "n_orders":   n_orders
                      }

        self.dim = dim
        self.ES = ES
        self.history = []

        # Define the linear functionals we'll be solving for.
        self.l1 = ta.Tensor(self.dim, self.params["N"])
        self.l1.value = [torch.randn([self.dim] * n, requires_grad=True) for n in range(self.params["N"] + 1)]

        self.l2 = ta.Tensor(self.dim, self.params["N"])
        self.l2.value = [torch.randn([self.dim] * n, requires_grad=True) for n in range(self.params["N"] + 1)]



    @staticmethod
    def _sample(path, points=100):
        """Sub-samples the path."""

        times = np.linspace(path[0, 0], path[-1, 0], points)
        idx = np.searchsorted(path[:, 0], times, side="left")
        #idx = np.unique(np.linspace(0, len(path) - 1, points).astype(int))


        return path[idx]

    def value_fn(self, l, s01, s02):
        """Investor's value function."""

        l_dim = len(l)
        l1, l2 = l[:l_dim // 2], l[l_dim // 2 :]
        l1 = ta.sig_to_tensor(l1, self.dim, self.params["N"])
        l2 = ta.sig_to_tensor(l2, self.dim, self.params["N"])

        # Define tensor algebra constants.
        ONE = ta.one(self.dim)
        E1 = ta.e(0, self.dim)
        E2 = ta.e(1, self.dim)
        E3 = ta.e(2, self.dim)

        # Define spot prices.


        # Define polynomial on the dual of the tensor algebra

        gl1 = self.params["Lambda1"] * l1
        gl2 = self.params["Lambda2"] * l2

        pl1 = E2 + s01 * ONE
        pl2 = E3 + s02 * ONE
        ptildel1 = E2 + s01 * ONE - gl1
        ptildel2 = E3 + s02 * ONE - gl2


        wl1 = ptildel1.shuffle(l1) * E1
        wl2 = ptildel2.shuffle(l2) * E1

        ql1 = (self.params["q0"] / self.params["s0"]) * ONE - l1 * E1
        ql2 = -l2 * E1

        ml1 = ql1.shuffle(pl1)

        poly = wl2
        poly += ml1.shuffle(pl2 - self.params["alpha1"] * ml1)
        poly += (ql2 + wl1).shuffle(pl2 - self.params["alpha2"] * ql2 - self.params["alpha2"] * wl1)

        if self.params["phi1"] != 0.:
            # L2 penalty on stock inventory
            poly += -self.params["phi1"] * ql1.shuffle(ql1) * E1

        if self.params["phi2"] != 0.:
            # L2 penalty on foreign currency inventory
            poly += -self.params["phi2"] * (ql2 + wl1).shuffle(ql2 + wl1) * E1

        # Define loss function
        loss = -poly.dot(self.ES)

        return float(loss)


    def train(self, s01=1., s02=1., method="BFGS", **kwargs):
        """Solves the double-liquidation problem."""

        l = np.zeros(2 * len(self.l1.flatten()))


        pbar = tqdm(desc="Optimising")
        t = time()
        res = optimize.minimize(self.value_fn, l, args=(s01, s02,), method=method,
                                callback=lambda _ : pbar.update(), **kwargs)
        dt = time() - t
        pbar.close()

        l_dim = len(res.x)
        l1, l2 = res.x[:l_dim // 2], res.x[l_dim // 2 :]
        self.l1 = ta.sig_to_tensor(l1, self.dim, self.params["N"])
        self.l2 = ta.sig_to_tensor(l2, self.dim, self.params["N"])

        print("Done in {} seconds. Maximum value: {}".format(dt, -res.fun))

    def train_sgd(self, s01=1., s02=1., n_epochs=100, lr=0.01, method="SGD"):
        """Solves the double-liquidation problem with stochastic gradient descent."""

        # Define tensor algebra constants.
        ONE = ta.one(self.dim)
        E1 = ta.e(0, self.dim)
        E2 = ta.e(1, self.dim)
        E3 = ta.e(2, self.dim)

        # Define Torch optimiser.
        if method == "SGD":
            optimiser = optim.SGD(self.l1.value + self.l2.value, lr=lr)
        elif method == "Adam":
            optimiser = optim.Adam(self.l1.value + self.l2.value, lr=lr)
        else:
            raise ValueError("Method {} not supported.".format(method))

        pbar = tqdm(range(n_epochs))

        for _ in pbar:
            optimiser.zero_grad()

            # Define polynomial on the dual of the tensor algebra

            gl1 = self.params["Lambda1"] * self.l1
            gl2 = self.params["Lambda2"] * self.l2

            pl1 = E2 + s01 * ONE
            pl2 = E3 + s02 * ONE
            ptildel1 = E2 + s01 * ONE - gl1
            ptildel2 = E3 + s02 * ONE - gl2


            wl1 = ptildel1.shuffle(self.l1) * E1
            wl2 = ptildel2.shuffle(self.l2) * E1

            ql1 = (self.params["q0"] / self.params["s0"]) * ONE - self.l1 * E1
            ql2 = -self.l2 * E1

            ml1 = ql1.shuffle(pl1)

            poly = wl2
            poly += ml1.shuffle(pl2 - self.params["alpha1"] * ml1)
            poly += (ql2 + wl1).shuffle(pl2 - self.params["alpha2"] * ql2 - self.params["alpha2"] * wl1)

            if self.params["phi1"] != 0.:
                # L2 penalty on stock inventory
                poly += -self.params["phi1"] * ql1.shuffle(ql1) * E1

            if self.params["phi2"] != 0.:
                # L2 penalty on foreign currency inventory
                poly += -self.params["phi2"] * (ql2 + wl1).shuffle(ql2 + wl1) * E1


            # Define loss function
            loss = -poly.dot(self.ES)

            # Train
            loss.backward()
            optimiser.step()

            loss_val = -loss.detach().numpy()
            pbar.set_description(str(loss_val))
            self.history.append(loss_val)




    def plot_results(self, paths, show=True, name1="Equity", name2="FX", return_all=False):
        """Plots results of the optimised signature trading strategy."""

        paths = np.array([self._sample(path, points=self.params["n_orders"]) for path in paths])


        all_stock = []
        all_fx = []
        wealth = []
        all_pos_fx = []
        all_pos_eq = []
        dt = 1. / self.params["n_orders"]

        for path in tqdm(paths):
            stock = [self.params["q0"] / self.params["s0"]]
            fx = [0.]
            _pos_fx = []
            _pos_eq = []

            w = 0.
            for i in range(1, len(path)):
                if self.params["N"] > 0:
                    sig = np.r_[1., iisignature.sig(path[:i + 1], self.params["N"])]
                else:
                    sig = np.array([1.])

                pos_stock = np.dot(self.l1.flatten(), sig)
                pos_fx = np.dot(self.l2.flatten(), sig)

                _pos_eq.append(pos_stock)
                _pos_fx.append(pos_fx)

                q = stock[-1] - pos_stock * dt
                q_fx = fx[-1] - pos_fx * dt + pos_stock * (path[i, 1] - self.params["Lambda1"] * pos_stock) * dt
                w += pos_fx * (path[i, 2] - self.params["Lambda2"] * pos_fx) * dt

                stock.append(q)
                fx.append(q_fx)

            all_stock.append(self.params["s0"] * np.array(stock))
            all_fx.append(self.params["s0"] * np.array(fx))
            all_pos_fx.append(_pos_fx)
            all_pos_eq.append(_pos_eq)

            w += fx[-1] * (path[i, 2] - self.params["alpha2"] * fx[-1])
            wealth.append(self.params["s0"] ** 2 * w)

        timeline = np.linspace(0, 1, len(all_stock[0]))
        for stock in all_stock:
            plt.plot(timeline, stock, "C0", alpha=0.1, linewidth=4)

        for fx in all_fx:
            plt.plot(timeline, fx, "C1", linestyle="--", alpha=0.1, linewidth=4)




        legend_elements = [Line2D([0], [0], color='C0', lw=4, label='{} inventory'.format(name1)),
                           Line2D([0], [0], linestyle="--", color='C1', lw=4, label='{} inventory'.format(name2))]

        plt.legend(handles=legend_elements)


        plt.xlabel("Time")
        plt.ylabel("Inventory")

        if show:
            plt.show()

        if return_all:
            return wealth, np.array(all_pos_eq), np.array(all_pos_fx)

