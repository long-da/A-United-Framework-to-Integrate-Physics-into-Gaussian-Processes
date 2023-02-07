import torch


class KernelARD2:

    def __init__(self, jitter):
        self.jitter = jitter

    def matrix(self, X, ls):
        K = self.cross(X, X, ls)
        Ijit = self.jitter * torch.eye(X.shape[0]).to(K.device)
        K = K + Ijit
        return K

    def matrix_p_p(self, Kmm, X, d, ls):
        K = self.cross_p_p(Kmm, X, X, d, ls)
        Ijit = self.jitter * torch.eye(X.shape[0]).to(K.device)
        K = K + Ijit
        return K

    def matrix_pp_pp(self, Kmm, X, d, ls):
        K = self.cross_pp_pp(Kmm, X, X, d, ls)
        Ijit = self.jitter * torch.eye(X.shape[0]).to(K.device)
        K = K + Ijit
        return K

    def cross(self, X1, X2, ls):
        ls_sqrt = torch.sqrt(ls)
        X1 = X1 / ls_sqrt
        X2 = X2 / ls_sqrt
        norm1 = torch.reshape(torch.sum(torch.square(X1), dim=1), [-1, 1])
        norm2 = torch.reshape(torch.sum(torch.square(X2), dim=1), [1, -1])
        K = norm1 - 2.0 * torch.matmul(X1, X2.T) + norm2
        K = torch.exp(-1.0 * K)
        return K

    def cross_p_0(self, Kmn, X1, X2, d, ls):
        s = ls[d]
        D = X1[:, d:d + 1] - X2[:, d:d + 1].T
        res = Kmn * (-2 / s * D)
        return res

    def cross_p_p(self, Kmm, X1, X2, d, ls):
        s = ls[d]
        D = X1[:, d:d + 1] - X2[:, d:d + 1].T
        res = Kmm * (-4 / s**2 * D**2 + 2 / s)
        return res

    def cross_p0_p1(self, Kmn, X1, X2, ls):
        s1 = ls[0]
        s2 = ls[1]
        D1 = X1[:, 0:1] - X2[:, 0:1].T
        D2 = X1[:, 1:2] - X2[:, 1:2].T
        res = Kmn * (-4 / s1 / s2 * D1 * D2)
        return res

    def cross_p_pp(self, Kmn, X1, X2, d, ls):
        s = ls[d]
        D = X1[:, d:d + 1] - X2[:, d:d + 1].T
        res = Kmn * (-8 * D**3 / s**3 + 12 * D / s**2)
        return res

    def cross_p0_pp1(self, Kmn, X1, X2, ls):
        s1 = ls[0]
        s2 = ls[1]
        D1 = X1[:, 0:1] - X2[:, 0:1].T
        D2 = X1[:, 1:2] - X2[:, 1:2].T
        res = Kmn * 4 * D1 / s1 / s2 * (-2 / s2 * D2**2 + 1)
        return res

    def cross_p1_pp0(self, Kmn, X1, X2, ls):
        s1 = ls[0]
        s2 = ls[1]
        D1 = X1[:, 0:1] - X2[:, 0:1].T
        D2 = X1[:, 1:2] - X2[:, 1:2].T
        res = Kmn * 4 * D2 / s1 / s2 * (-2 / s1 * D1**2 + 1)
        return res

    def cross_pp_pp(self, Kmm, X, Xp, d, ls):
        s = ls[d]
        D = X[:, d:d + 1] - Xp[:, d:d + 1].T
        res = Kmm * (16 / torch.pow(s, 4) * torch.pow(D, 4) - 48 / torch.pow(s, 3) * torch.pow(D, 2) + 12 / torch.square(s))
        return res

    def cross_pp_0(self, Kmn, X, Xp, d, ls):
        s = ls[d]
        D = X[:, d:d + 1] - Xp[:, d:d + 1].T
        res = Kmn * (4 / torch.pow(s, 2) * torch.pow(D, 2) - 2.0 / s)
        return res

    def cross_pp0_pp1(self, Kmn, X, Xp, ls):
        s1 = ls[0]
        D1 = X[:, 0:1] - Xp[:, 0:1].T
        s2 = ls[1]
        D2 = X[:, 1:2] - Xp[:, 1:2].T
        res = (Kmn * (4 / torch.pow(s1, 2) * torch.square(D1) - 2.0 / s1) * (4 / torch.pow(s2, 2) * torch.square(D2) - 2.0 / s2))
        return res
