# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 23:04:48 2020

@author: Sparse
"""
import os
import csv
import math

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.nn.utils.parametrizations import orthogonal
from torch.optim.lr_scheduler import StepLR

from scipy import stats
import numpy as np
import pandas as pd
import math


# SparsePCA hinge GAN
class ITSPCA:
    def __init__(self, config):
        self.config = config
        # Set universal random seed for reproducibility
        torch.manual_seed(self.config.seed)

        # Generate real data X under config model setting
        self.real, self.V, self.D = self.sampler_real()

    # Sample real data
    def sampler_real(self):
        p = self.config.p
        r = self.config.r
        k = self.config.k
        n = self.config.n
        eps = self.config.eps
        n_Q = round(eps * n)

        V = torch.Tensor(p, r).normal_()
        for i in range(k):
            V[i, :] = V[i, :] * (i + 1) ** 2
        V[list(range(k, p)), :] = 0  # k-sparse, k=5

        u, s, vh = torch.linalg.svd(V, full_matrices=False)
        V = u @ vh
        D = (torch.linspace(start=20, end=10, steps=r) ** (1 / 2))
        if r == 1:
            D = torch.linspace(start=20 ** (1 / 2), end=20 ** (1 / 2), steps=r)

        P_0 = torch.Tensor(n - n_Q, r).normal_() @ torch.diag(D) @ V.T + \
              torch.Tensor(n - n_Q, p).normal_()

        Q = torch.empty(n_Q, p)
        if self.config.Q == 'far_cluster':
            # The second type
            Q.normal_(mean=0, std=5 ** .5)
            chi2 = torch.empty(n_Q, 1).normal_() ** 2
            Q /= chi2 ** .5
            Q += 5 * torch.ones(p)
        elif self.config.Q == 'far_point':
            # Single point
            Q.fill_(10)
        elif self.config.Q == 'close_cluster':
            # The first type
            Q.normal_(mean=0, std=10 ** .5)
            chi2 = torch.empty(n_Q, 1).normal_() ** 2
            Q /= chi2 ** .5
        else:
            Q.normal_(median=5, sigma=5 ** .5)

        self.V = V
        self.D = D
        return torch.cat([P_0, Q], dim=0), V, D

    def evaluate1(self, write_to_csv=False):
        p = torch.Tensor([self.config.p])
        n = torch.Tensor([self.config.n])
        m = torch.Tensor([self.config.r])

        Z = torch.zeros([int(n), int(p)]).normal_()
        X = self.real.cpu()
        X0 = X + Z  # n x p
        X1 = X - Z
        S0 = X0.T.cov()
        S1 = X1.T.cov()

        alpha = 3
        beta = 2.1
        delta = 0.05
        lambda_0 = torch.log(p) / n
        ## Algorithm 2 for initial V_0
        alpha_n = alpha * torch.sqrt(lambda_0)
        J = (torch.diag(S0) >= (2 + 2 * alpha_n)).nonzero(as_tuple=True)[0]
        S_JJ = (S0[J].T)[J]
        eigval, eigvec = torch.linalg.eigh(S_JJ)  # eigvec is cardB x cardB
        eigval = eigval.flip(0)
        eigvec = eigvec.flip(1)
        eigvec = eigvec.T
        Q_B = torch.zeros([int(m), int(p)])  # cardB x p
        for i in range(int(m)):
            Q_B[i][J] = eigvec[i]
        V0 = Q_B.T  # p x m

        B = X0 @ V0  # n x p x p x m
        L, C, Rt = torch.linalg.svd(B)  # n x m x m x m x m x m
        Y = X1.T @ B @ Rt.T @ torch.diag(1 / C) / torch.sqrt(torch.ones(1) * 2)

        def t(k):
            return int(m) + math.sqrt(2 * int(m) * beta * math.log(math.exp(1) * int(p) / k)) + beta * math.log(
                math.exp(1) * int(p) / k)

        res = torch.ones([int(p)]) * 1e22
        Y_norm = torch.ones(int(p))
        for i in range(int(p)):
            Y_norm[i] = Y[i].norm(2)
        Y_norm = torch.sort(Y_norm, descending=True).values

        for k in range(0, int(p)):
            res[k] = torch.tensor((1 + delta) ** 2 * sum([t(x) for x in range(1, k + 2)])) + \
                     sum([(Y_norm[j]) ** 2 for j in range(k + 1, int(p))])

        khat = (res == min(res)).nonzero(as_tuple=True)[0] + 1
        print(khat)
        Theta = torch.empty([int(p), int(m)])
        for i in range(int(p)):
            Theta[i] = Y[i] * (Y[i].norm(2) ** 2 > (1 + delta) ** 2 * t(khat))
        model.Theta = Theta
        V_hat0, _ = torch.linalg.qr(Theta)

        J = (torch.diag(S1) >= (2 + 2 * alpha_n)).nonzero(as_tuple=True)[0]
        S_JJ = (S1[J].T)[J]
        eigval, eigvec = torch.linalg.eigh(S_JJ)  # eigvec is cardB x cardB
        eigval = eigval.flip(0)
        eigvec = eigvec.flip(1)
        eigvec = eigvec.T
        Q_B = torch.zeros([int(m), int(p)])  # cardB x p
        for i in range(int(m)):
            Q_B[i][J] = eigvec[i]
        V1 = Q_B.T  # p x m

        B = X1 @ V1  # n x p x p x m
        L, C, Rt = torch.linalg.svd(B)  # n x m x m x m x m x m
        Y = X0.T @ B @ Rt.T @ torch.diag(1 / C) / torch.sqrt(torch.ones(1) * 2)

        def t(k):
            return int(m) + math.sqrt(2 * int(m) * beta * math.log(math.exp(1) * int(p) / k)) + beta * math.log(
                math.exp(1) * int(p) / k)

        res = torch.ones([int(p)]) * 1e22
        Y_norm = torch.ones(int(p))
        for i in range(int(p)):
            Y_norm[i] = Y[i].norm(2)
        Y_norm = torch.sort(Y_norm, descending=True).values

        for k in range(0, int(p)):
            res[k] = torch.tensor((1 + delta) ** 2 * sum([t(x) for x in range(1, k+2)])) + \
                     sum([(Y_norm[j]) ** 2 for j in range(k+1, int(p))])

        khat = (res == min(res)).nonzero(as_tuple=True)[0] + 1
        print(khat)
        Theta = torch.empty([int(p), int(m)])
        for i in range(int(p)):
            Theta[i] = Y[i] * (Y[i].norm(2) ** 2 > (1 + delta) ** 2 * t(khat))
        model.Theta = Theta
        V_hat1, _ = torch.linalg.qr(Theta)

        V = self.V.cpu()

        _, vec = torch.linalg.eigh(V_hat0.mm(V_hat0.T) + V_hat1.mm(V_hat1.T))
        vec = vec.flip(1)
        vec = (vec.T)[:int(m)].T
        print((V_hat0.mm(V_hat0.T) - V.mm(V.T)).detach().norm(2) ** 2)
        print((V_hat1.mm(V_hat1.T) - V.mm(V.T)).detach().norm(2) ** 2)
        print((vec @ vec.T - V.mm(V.T)).detach().norm(2) ** 2)
        val = (vec @ vec.T - V.mm(V.T)).detach().norm(2)
        if write_to_csv:
            # writing to csv file
            filename = 'metrics2.csv'
            with open(filename, 'a+', newline='') as csvfile:
                # writing the data rows
                csvwriter = csv.writer(csvfile)
                new_row = [str(x) for x in [val.item() ** 2]]
                csvwriter.writerow(new_row)
    def evaluate(self, write_to_csv=False):
        V = self.V.cpu()
        # ITSPCA, Ma13
        S = self.real.T.cpu().cov()
        alpha = 3
        p = torch.Tensor([self.config.p])
        n = torch.Tensor([self.config.n])
        m = torch.Tensor([self.config.r])
        lambda_0 = torch.log(p) / n
        # eigval, _ = torch.linalg.eigh(S)

        ## Algorithm 2 for initial Q_B
        alpha_n = alpha * torch.sqrt(lambda_0)
        B = (torch.diag(S) >= 1 + alpha_n).nonzero(as_tuple=True)[0]
        # B = torch.linspace(1,39, 39).int()
        S_BB = (S[B].T)[B]
        eigval, eigvec = torch.linalg.eigh(S_BB)  # eigvec is cardB x cardB
        cardB = (torch.tensor(B.size()))

        Q_B = torch.zeros([int(cardB), int(p)])  # cardB x p
        for i in range(int(cardB)):
            Q_B[i][B] = eigvec[i]

        ## Algorithm 1, the actual ITSPCA
        model.Q_B = Q_B
        model.B = B
        model.eigvec = eigvec
        model.eigval = eigval

        Q = Q_B[0:int(m)]
        # Q = self.V.T.cpu()
        gamma = 1
        gamma_n = gamma * torch.sqrt(lambda_0)
        gamma_nj = torch.ones(int(m)) * gamma_n
        eig = eigval.sort(descending=True).values
        for i, x in enumerate(gamma_nj):
            gamma_nj[i] = x * torch.sqrt(max(torch.tensor(1), eig[i]))
        K_s = (1.1 * eig[0] / (eig[int(m) - 1] - eig[int(m)])) * (
                (1 + 1 / torch.log(torch.tensor(2))) * torch.log(n) +
                max(0, (eig[0] - 1) ** 2 / (eig[0]))
        )

        K_s = min(K_s, 200)  # save time
        for i in range(int(K_s)):
            T = Q @ S  # cardB x p x p x p
            for j in range(int(m)):
                T[j] = T[j] * (abs(T[j]) >= gamma_nj[j])
            Q, _ = torch.linalg.qr(T.T)
            print((Q.mm(Q.T) - V.mm(V.T)).detach().norm(2) ** 2)
            Q = Q.T

        V_hat = Q.T
        self.V_hat = V_hat
        # V = self.V.cpu()
        # print(V_hat.size())

        val = (V_hat.mm(V_hat.T) - V.mm(V.T)).detach().norm(2)
        # print('\n')

        # Append value of errors to a csv file as a new row, create file if it doesn't exist.
        if write_to_csv:
            # writing to csv file
            filename = 'metrics1.csv'
            with open(filename, 'a+', newline='') as csvfile:
                # writing the data rows
                csvwriter = csv.writer(csvfile)
                new_row = [str(x) for x in [val.item() ** 2]]
                csvwriter.writerow(new_row)


if __name__ == "__main__":
    from config import Config

    config = Config().parse()
    model = ITSPCA(config)
    model.evaluate(write_to_csv=True)
    model.evaluate1(write_to_csv=True)