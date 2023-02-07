import numpy as np
import torch
from torch.optim import Adam
from kernels2 import KernelARD2
import random
import matplotlib.pyplot as plt
import pandas as pd

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

torch.set_default_tensor_type(torch.DoubleTensor)


class PIGP:

    def __init__(self, X, y, X_col, device, jitter=1e-6):
        self.device = device
        y = y.reshape([y.size, 1])
        self.y = torch.tensor(y, device=self.device, requires_grad=False)
        self.X = torch.tensor(X, device=self.device, requires_grad=False)
        self.N = y.size
        self.d = X.shape[1]
        self.X_col = torch.tensor(X_col, device=self.device, requires_grad=False)
        self.N_col = self.X_col.shape[0]
        self.X = torch.cat((self.X, self.X_col), 0)
        self.N_con = self.X.shape[0]
        self.jitter = jitter
        self.log_ls_h = torch.tensor(torch.zeros(1), device=self.device, requires_grad=True)
        self.log_v = torch.tensor(0.0, device=self.device, requires_grad=True)
        self.log_tau = torch.tensor(0.0, device=self.device, requires_grad=True)
        self.M = self.N_con + 2 * self.N_col
        self.mu = torch.tensor(torch.randn([self.M, 1]), device=self.device, requires_grad=True)
        self.L1 = torch.tensor(torch.zeros(self.M, 1), device=self.device, requires_grad=True)
        self.L2 = torch.tensor(torch.zeros(self.M, self.M), device=self.device, requires_grad=True)
        self.kernel = KernelARD2(self.jitter)
        self.yte = None
        self.best_RMSE = None

    def KL_term(self, mu, L, Kmm):
        Ltril = torch.tril(L)
        hh_expt = torch.matmul(Ltril, Ltril.T) + torch.matmul(mu, mu.T)
        kl = (0.5 * torch.trace(torch.linalg.solve(Kmm, hh_expt)) + 0.5 * torch.logdet(Kmm) - 0.5 * torch.sum(torch.log(torch.square(torch.diag(Ltril)))))
        return kl

    def negELBO(self):
        K_u_u = self.kernel.matrix(self.X, torch.exp(self.log_ls_h))
        K_u1 = self.kernel.cross(self.X_col, self.X, torch.exp(self.log_ls_h))
        K_u2 = self.kernel.cross(self.X_col, self.X_col, torch.exp(self.log_ls_h))
        K_u_tt_u_tt = self.kernel.matrix_pp(K_u2, self.X_col, 0, torch.exp(self.log_ls_h))
        K_u_tt = self.kernel.cross_pp_0(K_u1, self.X_col, self.X, 0, torch.exp(self.log_ls_h))
        K1 = torch.cat((torch.cat((K_u_u, K_u_tt.T), 1), torch.cat((K_u_tt, K_u_tt_u_tt), 1)), 0)
        K_gg = self.kernel.matrix(self.X_col, torch.exp(self.log_ls_h))
        K = torch.block_diag(K1, K_gg)
        A = torch.cholesky(K)
        self.L = torch.diag(self.L1.exp().view(-1)) + torch.matmul(self.L2, self.L2.T)
        self.L = torch.cholesky(self.L)
        Ltril = self.L
        kl = self.KL_term(self.mu.clone().detach(), Ltril, torch.eye(self.M, device=self.device))
        eta = self.mu + torch.matmul(Ltril, torch.randn(self.M, 1, device=self.device))
        f = torch.matmul(A, eta)
        elbo = (-kl + 0.5 * self.N * self.log_tau - 0.5 * torch.exp(self.log_tau) * torch.sum(torch.square(f[0:self.N].view(-1, 1) - self.y)) + 0.5 * self.N_col * self.log_v -
                0.5 * torch.exp(self.log_v) * torch.sum(torch.square(f[self.N_con:self.N_con + self.N_col].view(-1, 1) - f[self.N_con + self.N_col:self.N_con + 2 * self.N_col].view(-1, 1))))
        return -elbo

    def train(self, Xte, yte, lr, max_epochs=100):
        Xte = torch.tensor(Xte, device=self.device)
        yte = torch.tensor(yte.reshape([yte.size, 1]), device=self.device)
        self.yte = yte
        for epoch in range(max_epochs):
            self.epoch = epoch
            paras = [self.mu, self.L1, self.L2, self.log_ls_h, self.log_tau, self.log_v]
            self.minimizer = Adam(paras, lr=lr)
            self.minimizer.zero_grad()
            loss = self.negELBO()
            loss.backward(retain_graph=True)
            self.minimizer.step()
            if epoch % 1 == 0:
                self._callback(Xte, yte, epoch)

    def _callback(self, Xte, yte, epoch):
        with torch.no_grad():
            pred_mean, std = self.pred(Xte)
            err_te = torch.sqrt(torch.mean(torch.square(pred_mean - yte)))
            if self.best_RMSE is None or self.best_RMSE > err_te:
                print("Epoch", self.epoch, "RMSE: ", err_te)
                self.best_RMSE = err_te
                plt.plot(Xte.cpu(), yte.cpu(), label="Ground-truth")
                plt.plot(Xte.cpu(), pred_mean.mean(axis=1).cpu(), label="PIGP")
                plt.plot(Xte.cpu(), (pred_mean.mean(axis=1)).view(-1, 1).cpu() - std.cpu(), linewidth=.3)
                plt.plot(Xte.cpu(), (pred_mean.mean(axis=1)).view(-1, 1).cpu() + std.cpu(), linewidth=.3)
                ti = "RMSE " + str(err_te.cpu().numpy())
                plt.title(ti)
                plt.legend()
                plt.savefig("latent_no_damping")
                plt.cla()

    def pred(self, Xte):
        K_u_u = self.kernel.matrix(self.X, torch.exp(self.log_ls_h))
        K_u1 = self.kernel.cross(self.X_col, self.X, torch.exp(self.log_ls_h))
        K_u2 = self.kernel.cross(self.X_col, self.X_col, torch.exp(self.log_ls_h))
        K_utt_utt = self.kernel.matrix_pp(K_u2, self.X_col, 0, torch.exp(self.log_ls_h))
        K_utt = self.kernel.cross_pp_0(K_u1, self.X_col, self.X, 0, torch.exp(self.log_ls_h))
        K1 = torch.cat((torch.cat((K_u_u, K_utt.T), 1), torch.cat((K_utt, K_utt_utt), 1)), 0)
        K_gg = self.kernel.matrix(self.X_col, torch.exp(self.log_ls_h))
        K = torch.block_diag(K1, K_gg)
        A = torch.cholesky(K)
        Ltril = torch.tril(self.L)
        eta = self.mu
        f = torch.matmul(A, eta)
        K_u1_te = self.kernel.cross(Xte, self.X, torch.exp(self.log_ls_h))
        K_u2_te = self.kernel.cross(Xte, self.X_col, torch.exp(self.log_ls_h))
        K_utt_te = self.kernel.cross_pp_0(K_u2_te.T, self.X_col, Xte, 0, torch.exp(self.log_ls_h))
        K_te = torch.cat((K_u1_te, K_utt_te.T), 1)
        pred_mean = torch.matmul(K_te, torch.linalg.solve(K1, f[0:self.N_con + self.N_col]))
        A1 = torch.cholesky(K1)
        t1 = torch.linalg.solve(A1.T, Ltril[0:self.N_con + self.N_col])
        t2 = torch.matmul(t1, t1.T)
        pred_stds = (1 + self.jitter - (K_te * torch.linalg.solve(K1, K_te.T).T).sum(1) + (K_te * torch.matmul(t2, K_te.T).T).sum(1))
        pred_stds = pred_stds.view(pred_mean.shape)
        return pred_mean, pred_stds**0.5


device = "cpu"

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)


def test(M):
    infile1 = "./Data/no_damping_train.csv"
    infile2 = "./Data/no_damping_test.csv"
    X_train = ((pd.read_csv(infile1)).to_numpy()[:, :-1]).reshape(-1, 1)
    Y_train = (pd.read_csv(infile1)).to_numpy()[:, -1].reshape(-1, 1)
    X_test0 = (pd.read_csv(infile2)).to_numpy()
    X_test0 = X_test0[np.argsort(X_test0[:, 0])]
    X_test = (X_test0[:, :-1]).reshape(-1, 1)
    Y_test = X_test0[:, -1].reshape(-1, 1)
    X_col = np.min(X_test, 0) + (np.max(X_test, 0) - np.min(X_test, 0)) * (np.linspace(0, 1, num=M)).reshape((-1, 1))
    model_PIGP = PIGP(X_train, Y_train, X_col, torch.device(device))
    lr = 0.05
    nepoch = 2000
    model_PIGP.train(X_test, Y_test, lr, nepoch)


test(20)