    # -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 23:04:48 2020

@author: Sparse
"""
import os
import csv

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


# Loss
def rKL_loss(hx, hgx):
    return 1 - (-hx).exp().mean() - hgx.mean()


def JS_loss(hx, hgx):
    return 2 * math.log(2) - \
           torch.log(1 + torch.exp(-hx)).mean() - \
           torch.log(1 + torch.exp(hgx)).mean()


def H2_loss(hx, hgx):
    return -(-hx / 2).exp().mean() - (hgx / 2).exp().mean() + 2


def hinge_loss(hx, hgx):
    # E_real min(1, h(x)) + E_fake min(1, -h(x))
    return -F.relu(1 - hx).mean() - \
           F.relu(1 + hgx).mean() + 2


def loss_select(name):
    dic = {
        'rKL': (rKL_loss, rKL_loss),
        'JS': (JS_loss, JS_loss),
        'hinge_cal': (hinge_loss, rKL_loss),
        'hinge': (hinge_loss, hinge_loss),
        'H2': (H2_loss, H2_loss)
    }
    return dic.get(name, 'Not a supported loss function.')


# Models
class Generator(nn.Module):
    def __init__(self, p, r):
        super(Generator, self).__init__()
        self.p = p
        self.r = r
        self.D = torch.nn.Parameter(torch.ones(r).cuda())
        self.V = orthogonal(nn.Linear(r, p, bias=False)).cuda()
        # register parametrization for nn.linear
        # https://pytorch.org/docs/stable/generated/torch.nn.utils.parametrize.register_parametrization.html#torch.nn.utils.parametrize.register_parametrization
        # https://pytorch.org/docs/stable/generated/torch.nn.utils.parametrizations.orthogonal.html
        # See above docs for how this is implemented and how to get the parametrized model's
        #   original parameters with grad.

        # self.cov = self.V.weight.data.mm(torch.diag(self.D**2)).mm(self.V.weight.data.T) + \
        #   torch.eye(self.p).cuda()

    def forward(self, x):
        x = self.V(x[0].mm(torch.diag((self.D)))) + x[1]
        #x = x[0] @ self.V(torch.diag(self.D)) + x[1]
        return x


class Discriminator(nn.Module):
    def __init__(self, p, node_size):
        super(Discriminator, self).__init__()
        self.p = p
        self.node_size = node_size
        self.main = nn.Sequential(
            nn.Linear(p, self.node_size, bias=True),
            nn.LeakyReLU(),
            nn.Linear(self.node_size, 1, bias=True)
        )

    def forward(self, x):
        return self.main(x)


# Utils
def init_weights(m):
    if type(m) == torch.nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


# SparsePCA hinge GAN
class SpcaGAN:
    def __init__(self, config):
        self.config = config

        # Set universal random seed for reproducibility
        torch.manual_seed(self.config.seed)

        # Generate real data X under config model setting
        self.real, self.V, self.D = self.sampler_real()

        # Setup networks and loss
        self.netG = Generator(self.config.p, self.config.r).cuda()
        self.netD = Discriminator(self.config.p, self.config.node_size).cuda()
        self.loss_d, self.loss_g = loss_select(self.config.loss)

        # Setup optimizers
        self.optimizerD = optim.Adam(self.netD.parameters(), lr=self.config.lr_d)
        if self.config.no_eigval:
            self.optimizerG = optim.Adam([{'params': list(self.netG.parameters())[1]}],
                                         lr=self.config.lr_g)
        else:
            #self.optimizerG = optim.Adam(self.netG.parameters(), lr=self.config.lr_g)
            self.optimizerG = optim.Adam([{'params': list(self.netG.parameters())[1], 'lr': self.config.lr_g},
                      {'params': self.netG.D, 'lr': 1 * self.config.lr_g}],
                     lr=self.config.lr_g)
        # Setup schedulers for learning rate decay
        self.schedulerD = StepLR(optimizer=self.optimizerD,
                                 step_size=self.config.decay_step,
                                 gamma=1)
        self.schedulerG = StepLR(optimizer=self.optimizerG,
                                 step_size=self.config.decay_step,
                                 gamma=self.config.decay_gamma)

        # Initialize networks
        self.netD.apply(init_weights)

        if self.config.use_sample_cov:
            sample_cov = self.real.T.cov()
            val, V0 = torch.linalg.eigh(sample_cov)
            idx = val.argsort(descending=True)
            val = val[idx]
            V0 = V0[:, idx]
            self.netG.V.weight.data = V0[:, :self.config.r]
            self.netG.D.data = torch.sqrt(val[:self.config.r])
        else:
            nn.init.xavier_uniform_(self.netG.V.weight)
            # Coordinate-wise robust scale estimation, use scale 'normal' for unbiased estimation.
            MAD = np.square(stats.median_abs_deviation(self.real.cpu(), scale='normal'))
            sigma_square_hat = np.median(MAD)
            # We can estimate the trace, but not each individual of eigenvalues unless the entire covariance matrix is
            # estimated, which is costly and not stable under contamination.
            # With a compromised method, we take average and use equal values for init estimate of the eigenvalues.
            D_hat = np.sqrt((MAD.sum() - self.config.p * sigma_square_hat) / self.config.r)
            if self.config.no_eigval:
                # Estimate only eigenvectors, assuming eigenvalues are known
                self.netG.D.data = self.D
            else:
                # Estimate both eigenvalues and eigenvectors
                self.netG.D.data = (torch.ones(self.config.r) * D_hat).cuda() + \
                        torch.cuda.FloatTensor(self.config.r).normal_(0, .1)

                # Small noises are added to prevent initial eigenvalues to be all equal.

    # Sample real data
    def sampler_real(self):
        p = self.config.p
        r = self.config.r
        k = self.config.k
        n = self.config.n
        eps = self.config.eps
        n_Q = round(eps * n)

        V = torch.cuda.FloatTensor(p, r).normal_()
        for i in range(k):
            V[i, :] = V[i, :] * (i + 1) ** 2
        V[list(range(k, p)), :] = 0  # k-sparse, k=5

        u, s, vh = torch.linalg.svd(V, full_matrices=False)
        V = u @ vh
        D = (torch.linspace(start=20, end=10, steps=r) ** (1 / 2)).cuda()
        if r == 1:
            D = torch.linspace(start=20 ** (1 / 2), end=20 ** (1 / 2), steps=r).cuda()

        P_0 = torch.cuda.FloatTensor(n - n_Q, r).normal_() @ torch.diag(D) @ V.T + \
            torch.cuda.FloatTensor(n - n_Q, p).normal_()

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
        return torch.cat([P_0, Q.cuda()], dim=0), V, D

    # Sample noise for generating fake data
    def sampler_fake(self):
        return torch.cuda.FloatTensor(self.config.fake_size, self.config.r).normal_(), \
               torch.cuda.FloatTensor(self.config.fake_size, self.config.p).normal_()

    # Train
    def train(self):
        _n_iter_d = self.config.n_iter_d

        digit_show = str(len(str(self.config.n_iter)))
        formatted = '0' + digit_show + 'd'

        tracker = pd.DataFrame(index=range(self.config.n_iter + 1),
                               columns=['dloss', 'gloss', 'dpen', 'gpen'])
        tracker[:1] = 0

        # Training
        for i in range(self.config.n_iter):

            # 8/20/2023 Keep D lr free from decay since it is fixed by def
            #self.optimizerG.param_groups[1]['lr'] = .1 * self.config.lr_g

            # Warm-up setting for the discriminator
            if i == 0 and not self.config.no_warm_up:
                print('\nDiscriminator warm-up training...\n')
                n_iter_d = int(1 / self.config.lr_d)  # min(int(5 / lr_d), int(n_iter/2))
            else:
                n_iter_d = _n_iter_d
            #n_iter_d = _n_iter_d
            # Update the discriminator
            for i_d in range(n_iter_d):
                self.netD.zero_grad()

                fake = self.netG(self.sampler_fake())
                h_real = self.netD(self.real)
                h_fake = self.netD(fake.detach())
                # pen = torch.sum(abs(self.netD.main[0].weight), dim=1).max() * \
                #      abs(self.netD.main[2].weight).max()
                pen = sum(self.netD.main[0].weight.norm(1, dim=1) *
                          self.netD.main[2].weight.norm(1, dim=0)) # Just for abs value
                D_loss_ = -self.loss_d(h_real, h_fake)
                # Object 'fake' is 'detached' so we don't calculate gradient for netG's parameters.
                D_loss = D_loss_ + self.config.lambda_d * pen

                D_loss.backward()
                if not self.config.no_norm_clip:
                    torch.nn.utils.clip_grad_norm_(self.netD.parameters(), .5)
                self.optimizerD.step()


            # update the generator
            for i_g in range(self.config.n_iter_g):
                self.netG.zero_grad()

                fake = self.netG(self.sampler_fake())
                h_real = self.netD(self.real).detach()
                h_fake = self.netD(fake)
                # group lasso penalty \|A\|_G = sum_{i=1,...,p} \|A_{i, *}\|_{2}
                pen_V = self.netG.V(torch.diag(self.netG.D)).norm(2, dim=0).norm(1)

                gloss_ = self.loss_g(h_real, h_fake)
                gloss = gloss_ + self.config.lambda_g * pen_V  # minimize

                gloss.backward()
                # netG.D.grad *= 10
                if not self.config.no_norm_clip:
                    torch.nn.utils.clip_grad_norm_(self.netG.parameters(), .5)
                self.optimizerG.step()


            tracker.dloss[i + 1] = -D_loss_.item()  # L(D_{t-1}, G_{t-1})
            tracker.dpen[i + 1] = self.config.lambda_d * pen.item()  # pen(D_{t-1})
            tracker.gloss[i + 1] = gloss_.item()  # L(D_{t}, G_{t-1})
            tracker.gpen[i + 1] = self.config.lambda_g * pen_V.item()  # pen(G_{t-1})

            # Learning rate decay
            self.schedulerG.step()
            self.schedulerD.step()

            # Estimation error after one iteration
            D_err = (self.netG.D.detach() - self.D).norm(2)
            V_hat = self.netG.V.weight
            V_err = (V_hat.mm(V_hat.T) - self.V.mm(self.V.T)).detach().norm(2)

            # Display result and output
            if i % self.config.display_gap == 0:
                print('Epoch', format(i, formatted),
                      '| V_err:', round(V_err.item() ** 2, 5),
                      '| D_err:', round(D_err.item(), 5),
                      '| D_loss:', round(tracker.dloss[i + 1], 5),
                      '| D_pen:', round(tracker.dpen[i + 1], 5),
                      '| G_loss:', round(tracker.gloss[i + 1], 5),
                      '| G_pen:', round(tracker.gpen[i + 1], 5)
                      )

    def evaluate(self, write_to_csv=False):
        V_hat = self.netG.V.weight.detach()
        D_hat = (self.netG.D.sort(descending=True).values.detach())**2

        D = self.D**2
        V = self.V
        idx = D.argsort(descending=True)
        D = D[idx]
        V = V[:, idx]

        print('\nD_est: ' + str(D_hat))
        print('D_true: ' + str(D))

        val = (V_hat.mm(V_hat.T) - self.V.mm(self.V.T)).detach().norm(2)

        print('||V_hat V_hat^T - VV^T||_F squared: ' + str(val.item()**2))
        print('\n')


        # Append value of errors to a csv file as a new row, create file if it doesn't exist.
        if write_to_csv:
            # writing to csv file
            filename = 'metrics.csv'
            with open(filename, 'a+', newline='') as csvfile:
                # writing the data rows
                csvwriter = csv.writer(csvfile)
                new_row = [str(x) for x in [val.item()**2]]
                csvwriter.writerow(new_row)
