import numpy as np
import torch
from torch.optim import Adam
from kernels import KernelARD2
import random
import pandas as pd
import matplotlib.pyplot as plt
from pyDOE import lhs

torch.set_default_tensor_type(torch.DoubleTensor)


class PIGP:

    def __init__(self, X, y, X_col, device, jitter=1e-6):
        self.device = device
        self.jitter = jitter
        self.kernel = KernelARD2(jitter)
        y = y.reshape([y.size, 1])
        self.X = torch.tensor(X, device=self.device, requires_grad=False)
        self.y = torch.tensor(y, device=self.device, requires_grad=False)
        self.X_col = torch.tensor(X_col, device=self.device, requires_grad=False)
        self.N = y.size
        self.d = X.shape[1]
        self.N_col = self.X_col.shape[0]
        self.X_concat = torch.cat((self.X, self.X_col), 0)
        self.N_concat = (self.N + self.N_col)
        self.log_ls_h = torch.tensor([0.0, 0.0], device=self.device, requires_grad=True)
        self.log_ls_g = torch.tensor([0.0, 0.0], device=self.device, requires_grad=True)
        self.log_v = torch.tensor(1.0, device=self.device, requires_grad=False)
        self.log_tau = torch.tensor(10.0, device=self.device, requires_grad=False)
        self.M = self.N_concat + 3 * self.N_col
        self.mu = torch.tensor(torch.zeros(self.M, 1), device=self.device, requires_grad=True)
        self.L1 = torch.tensor(torch.zeros(self.M, 1), device=self.device, requires_grad=True)
        self.L2 = torch.tensor(torch.zeros(self.M, self.M), device=self.device, requires_grad=True)
        self.best_RMSE = None

    def KL_term(self, mu, cov):
        kl = (0.5 * torch.trace(cov) + 0.5 * torch.square(mu).sum() - 0.5 * torch.logdet(cov))
        return kl

    def negELBO(self):
        K_u_u = self.kernel.matrix(self.X_concat, torch.exp(self.log_ls_h))
        K_u1 = self.kernel.cross(self.X_col, self.X_concat, torch.exp(self.log_ls_h))
        K_u2 = self.kernel.cross(self.X_col, self.X_col, torch.exp(self.log_ls_h))
        K_uxx_uxx = self.kernel.matrix_pp_pp(K_u2, self.X_col, 0, torch.exp(self.log_ls_h))
        K_uxx = self.kernel.cross_pp_0(K_u1, self.X_col, self.X_concat, 0, torch.exp(self.log_ls_h))
        K_ux_ux = self.kernel.matrix_p_p(K_u2, self.X_col, 1, torch.exp(self.log_ls_h))
        K_ux = self.kernel.cross_p_0(K_u1, self.X_col, self.X_concat, 1, torch.exp(self.log_ls_h))
        K_ux_uxx = self.kernel.cross_p1_pp0(K_u2, self.X_col, self.X_col, torch.exp(self.log_ls_h))
        K = torch.cat(
            (
                torch.cat((K_u_u, K_uxx.T, K_ux.T), 1),
                torch.cat((K_uxx, K_uxx_uxx, K_ux_uxx.T), 1),
                torch.cat((K_ux, K_ux_uxx, K_ux_ux), 1),
            ),
            0,
        )
        K_g = self.kernel.matrix(self.X_col, torch.exp(self.log_ls_g))
        K = torch.block_diag(K, K_g)
        A = torch.cholesky(K)
        cov = torch.diag(self.L1.exp().view(-1)) + torch.matmul(self.L2, self.L2.T)
        self.L = torch.cholesky(cov)
        Ltril = torch.tril(self.L)
        kl = self.KL_term(self.mu, cov)
        eta = self.mu + torch.matmul(Ltril, torch.randn(self.M, 1, device=self.device))
        f = torch.matmul(A, eta)
        elbo = (-kl + 0.5 * self.N * self.log_tau - 0.5 * torch.exp(self.log_tau) * torch.sum(torch.square(f[0:self.N].view(-1, 1) - self.y)) + 0.5 * self.N_col * self.log_v - 0.5 * torch.exp(self.log_v) *
                torch.sum(torch.square(f[self.N_concat + self.N_col:self.N_concat + 2 * self.N_col].view(-1, 1) - 0.0001 * f[self.N_concat:self.N_concat + self.N_col].view(-1, 1) - f[self.N_concat + 2 * self.N_col:self.N_concat + 3 * self.N_col])))
        return -elbo

    def train(self, Xte, yte, lr, max_epochs):
        Xte = torch.tensor(Xte, device=self.device)
        yte = torch.tensor(yte.reshape([yte.size, 1]), device=self.device)
        paras = [self.mu, self.L1, self.L2, self.log_ls_h, self.log_ls_g, self.log_tau, self.log_v]
        minimizer = Adam(paras, lr=lr)
        for epoch in range(max_epochs):
            if epoch % 1000 == 0:
                print("Epoch: ", epoch)
            self.epoch = epoch
            minimizer.zero_grad()
            loss = self.negELBO()
            loss.backward(retain_graph=True)
            minimizer.step()
            if epoch % 10 == 0:
                self._callback(Xte, yte, epoch)

    def _callback(self, Xte, yte, epoch):
        with torch.no_grad():
            K_u_u = self.kernel.matrix(self.X_concat, torch.exp(self.log_ls_h))
            K_u1 = self.kernel.cross(self.X_col, self.X_concat, torch.exp(self.log_ls_h))
            K_u2 = self.kernel.cross(self.X_col, self.X_col, torch.exp(self.log_ls_h))
            K_uxx_uxx = self.kernel.matrix_pp_pp(K_u2, self.X_col, 0, torch.exp(self.log_ls_h))
            K_uxx = self.kernel.cross_pp_0(K_u1, self.X_col, self.X_concat, 0, torch.exp(self.log_ls_h))
            K_ux_ux = self.kernel.matrix_p_p(K_u2, self.X_col, 1, torch.exp(self.log_ls_h))
            K_ux = self.kernel.cross_p_0(K_u1, self.X_col, self.X_concat, 1, torch.exp(self.log_ls_h))
            K_ux_uxx = self.kernel.cross_p1_pp0(K_u2, self.X_col, self.X_col, torch.exp(self.log_ls_h))
            K1 = torch.cat(
                (
                    torch.cat((K_u_u, K_uxx.T, K_ux.T), 1),
                    torch.cat((K_uxx, K_uxx_uxx, K_ux_uxx.T), 1),
                    torch.cat((K_ux, K_ux_uxx, K_ux_ux), 1),
                ),
                0,
            )
            K_g = self.kernel.matrix(self.X_col, torch.exp(self.log_ls_g))
            K = torch.block_diag(K1, K_g)
            A = torch.cholesky(K)
            eta = self.mu
            f = torch.matmul(A, eta)
            K_u1_te = self.kernel.cross(Xte, self.X_concat, torch.exp(self.log_ls_h))
            K_u2_te = self.kernel.cross(Xte, self.X_col, torch.exp(self.log_ls_h))
            K_utt_te = self.kernel.cross_pp_0(K_u2_te.T, self.X_col, Xte, 0, torch.exp(self.log_ls_h))
            K_ut_te = self.kernel.cross_p_0(K_u2_te.T, self.X_col, Xte, 1, torch.exp(self.log_ls_h))
            K_te = torch.cat((K_u1_te, K_utt_te.T, K_ut_te.T), 1)
            pred_mean = torch.matmul(K_te, torch.linalg.solve(K1, f[0:self.N_concat + 2 * self.N_col]))
            err_te = torch.sqrt(torch.mean(torch.square(pred_mean - yte)))
            if self.best_RMSE is None or self.best_RMSE > err_te:
                self.best_RMSE = err_te
                cov = torch.diag(self.L1.exp().view(-1)) + torch.matmul(self.L2, self.L2.T)
                self.L = torch.cholesky(cov)
                Ltril = torch.tril(self.L)
                A1 = torch.cholesky(K1)
                t1 = torch.linalg.solve(A1.T, Ltril[0:self.N_concat + 2 * self.N_col])
                t2 = torch.matmul(t1, t1.T)
                pred_stds = (1 + self.jitter - (K_te * torch.linalg.solve(K1, K_te.T).T).sum(1) + (K_te * torch.matmul(t2, K_te.T).T).sum(1))
                pred_stds = pred_stds.view(pred_mean.shape)
                print("Epoch", self.epoch, "RMSE: ", err_te)
                pred_mean = (pred_mean.cpu().reshape((67, 128)))
                pred_stds = pred_stds.cpu().reshape((67, 128))
                plt.imshow(pred_mean, cmap="hot")
                ti = "RMSE " + str(err_te.cpu().numpy())
                plt.title(ti)
                plt.savefig("latent_allen_cahn")
                plt.clf()
                plt.imshow(pred_stds, cmap="hot")
                ti = "RMSE " + str(err_te.cpu().numpy())
                plt.title(ti)
                plt.savefig("latent_allen_cahn_var")
                plt.clf()


def test(M):
    infile1 = "./Data/AllenCahnTrain.csv"
    infile2 = "./Data/AllenCahnTest.csv"
    X_train = (pd.read_csv(infile1)).to_numpy()[:, :-1]
    Y_train = (pd.read_csv(infile1)).to_numpy()[:, -1].reshape([-1, 1])
    X_test = (pd.read_csv(infile2)).to_numpy()[:, :-1]
    Y_test = (pd.read_csv(infile2)).to_numpy()[:, -1].reshape([-1, 1])
    lb = X_test.min(axis=0)
    ub = X_test.max(axis=0)
    X_col = lb + (ub - lb) * lhs(2, M)
    model_PIGP = PIGP(X_train, Y_train, X_col, torch.device("cuda"))
    lr = 0.05
    nepoch = 25000
    model_PIGP.train(X_test, Y_test, lr, nepoch)


torch.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)

test(100)