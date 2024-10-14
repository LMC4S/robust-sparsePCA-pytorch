import csv
import math

import torch

# Run experiments for ITSPCA and RegSPCA as described in the CMW13 paper
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
                
    def evaluate_ITSPCA(self, write_to_csv=False):
        V = self.V.cpu()
        # ITSPCA, Ma13
        # SPARSE PRINCIPAL COMPONENT ANALYSIS AND
        # ITERATIVE THRESHOLDING ALGORITHM, ZONGMING MA, 2013, AoS

        # Implementation of Algorithm 1 in the paper, with Algorithm 2 for initialization
        S = self.real.T.cpu().cov()
        alpha = 3 # the recommended value in Ma13   
        p = torch.Tensor([self.config.p])
        n = torch.Tensor([self.config.n])
        m = torch.Tensor([self.config.r])
        lambda_0 = torch.log(p) / n
        # eigval, _ = torch.linalg.eigh(S)

        ## Algorithm 2 DTSPCA in Ma13 for initial Q_B (see paragraph after Remark 2.1 for details)
        ## Input: S, the sample covariance matrix, diagonal thresholding parameter alpha_n
        ## Output: Q_B, the initial orthogonal matrix
        alpha_n = alpha * torch.sqrt(lambda_0) # Diagonal thresholding parameter, specified in eq (3.2)

        ### Step 1: Variance selection: select the set B of coordinates with large variances
        B = (torch.diag(S) >= 1 + alpha_n).nonzero(as_tuple=True)[0] # sigma^2 is set to be 1
        S_BB = (S[B].T)[B] # submatrix of S indexed by B

        ### Step 2: Reduced PCA
        eigval, eigvec = torch.linalg.eigh(S_BB)  # eigvec is cardB x cardB
        cardB = (torch.tensor(B.size()))

        ### Step 3: Zero-padding, construct Q_B
        Q_B = torch.zeros([int(cardB), int(p)])  # cardB x p
        for i in range(int(cardB)):
            Q_B[i][B] = eigvec[i]

        ## Algorithm 1, the actual "ITSPCA" in Ma13, 
        ## with orthonormal Q_B initialized by DTSPCA
        ## Input: S, the sample covariance matrix, orthonormal Q_0 = Q_B initialized by DTSPCA
        ## Output: subspace estimator P_m = ran(Q_converge)
        self.Q_B = Q_B
        self.B = B
        self.eigvec = eigvec
        self.eigval = eigval

        Q = Q_B[0:int(m)]
        # Q = self.V.T.cpu()
        gamma = 1 # threshold multiplier
        gamma_n = gamma * torch.sqrt(lambda_0)
        gamma_nj = torch.ones(int(m)) * gamma_n
        eig = eigval.sort(descending=True).values

        for i, x in enumerate(gamma_nj):
            gamma_nj[i] = x * torch.sqrt(max(torch.tensor(1), eig[i])) # specified in eq (3.3)

        K_s = (1.1 * eig[0] / (eig[int(m) - 1] - eig[int(m)])) * (
                (1 + 1 / torch.log(torch.tensor(2))) * torch.log(n) +
                max(0, (eig[0] - 1) ** 2 / (eig[0]))
        ) # iteration times specified in eq (3.4)

        K_s = min(K_s, 200)  # save time, will converge within 200 iterations anyway
        for i in range(int(K_s)):
            T = Q @ S  # cardB x p x p x p
            for j in range(int(m)):
                T[j] = T[j] * (abs(T[j]) >= gamma_nj[j]) # hard thresholding
            Q, _ = torch.linalg.qr(T.T)
            #print((Q.mm(Q.T) - V.mm(V.T)).detach().norm(2) ** 2)
            Q = Q.T

        V_hat = Q.T
        self.V_hat_ITSPCA = V_hat.detach().clone()
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
                
    def estimate_V(self, X0, X1, S, alpha_n, m, p, beta, delta):
        # Initial estimation for V, specified in Section 3.2 of CMW13
        # Diagonal thresholding method in JOHNSTONE, I. M. and LU, A. Y. (2009)
        J = (torch.diag(S) >= (2 + 2 * alpha_n)).nonzero(as_tuple=True)[0]  # eq (37)
        S_JJ = (S[J].T)[J]  # submatrix of S indexed by J
        eigval, eigvec = torch.linalg.eigh(S_JJ)  # eigvec is cardB x cardB
        eigval = eigval.flip(0)  # sort eigenvalues in descending order
        eigvec = eigvec.flip(1).T  # sort eigenvectors in descending order
        Q_B = torch.zeros([int(m), int(p)])  # cardB x p
        for i in range(int(m)):
            Q_B[i][J] = eigvec[i]  # eq (38)
        V = Q_B.T  # p x m
        
        # Reduction to regression
        B = X0 @ V  # n x p x p x m
        L, C, Rt = torch.linalg.svd(B)  # n x m x m x m x m x m
        Y = X1.T @ B @ Rt.T @ torch.diag(1 / C) / torch.sqrt(torch.ones(1) * 2)  # defined after eq (32)

        # Given Y, propose the following method for computing Theta_hat 
        def t(k):  # eq (33)
            return int(m) + math.sqrt(2 * int(m) * beta * math.log(math.exp(1) * int(p) / k)) + beta * math.log(
                math.exp(1) * int(p) / k)

        res = torch.ones([int(p)]) * 1e22
        Y_norm = torch.linalg.norm(Y, dim=1)
        Y_norm, _ = torch.sort(Y_norm, descending=True)  # get the sorted Y_norm

        for k in range(0, int(p)):  # grid search over 1 to p for khat following eq (36)
            res[k] = torch.tensor((1 + delta) ** 2 * sum([t(x) for x in range(1, k + 2)])) + \
                    torch.sum(Y_norm[k+1:]**2)

        khat = (res == min(res)).nonzero(as_tuple=True)[0] + 1
        # print(khat)

        Theta = torch.empty([int(p), int(m)])
        for i in range(int(p)):  # recover Theta_hat with t_khat
            Theta[i] = Y[i] * (torch.linalg.norm(Y[i])**2 > (1 + delta) ** 2 * t(khat))
        
        # Final estimation by orthonormalizing the cols of Theta_hat
        V_hat, _ = torch.linalg.qr(Theta)
        return V_hat

    def evaluate_RegSPCA(self, write_to_csv=False):
        # Adaptive estimation RegSPCA in CMW13 
        # SPARSE PCA: OPTIMAL RATES AND ADAPTIVE ESTIMATION, Cai, MA and Wu, 2013, AoS
        # Adaptive estimation in Section 3 of CMW13
        p = torch.Tensor([self.config.p])
        n = torch.Tensor([self.config.n])
        m = torch.Tensor([self.config.r])

        # Step 1: Sample generation
        # Given data matrix X, generate independent Z^tilde, then 
        # X0 = X + Z^tilde, X1 = X - Z^tilde, then calculate their covariance matrix S0, S1
        Z = torch.zeros([int(n), int(p)]).normal_()
        X = self.real.cpu()
        X0 = X + Z  # n x p
        X1 = X - Z
        S0 = X0.T.cov()
        S1 = X1.T.cov()

        # Step 2: Initial estimation for V^0, specified in Section 3.2 of CMW13
        alpha = 3
        beta = 2.1
        delta = 0.05  # specified in Section 4 of CMW13
        lambda_0 = torch.log(p) / n
        alpha_n = alpha * torch.sqrt(lambda_0)

        # Step 3 to 4, estimate V_hat0
        V_hat0 = self.estimate_V(X0, X1, S0, alpha_n, m, p, beta, delta)

        # Step 5: switch X0 and X1, run the whole thing again for V2 hat
        # This is described in Section 4, as a modification of Section 3.2
        V_hat1 = self.estimate_V(X1, X0, S1, alpha_n, m, p, beta, delta)

        # Final estimation of Vhat will be r leading eigenvectors of V1@V1.T + V2@V2.T
        V = self.V.cpu()

        _, vec = torch.linalg.eigh(V_hat0.mm(V_hat0.T) + V_hat1.mm(V_hat1.T))
        vec = vec.flip(1)
        vec = (vec.T)[:int(m)].T
        self.V_hat_RegSPCA = vec.detach().clone()

        val = (vec @ vec.T - V.mm(V.T)).detach().norm(2)

        if write_to_csv:
            # writing to csv file
            filename = 'metrics2.csv'
            with open(filename, 'a+', newline='') as csvfile:
                # writing the data rows
                csvwriter = csv.writer(csvfile)
                new_row = [str(x) for x in [val.item() ** 2]]
                csvwriter.writerow(new_row)

if __name__ == "__main__":
    from config import Config

    config = Config().parse()
    model = ITSPCA(config)
    model.evaluate_ITSPCA(write_to_csv=True)
    model.evaluate_RegSPCA(write_to_csv=True)
