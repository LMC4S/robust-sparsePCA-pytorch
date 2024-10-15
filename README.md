# Robust Sparse PCA with hinge GAN

Hinge GAN with orthonormal generator for robust sparse PCA under Huber's contamination model. 

## Example usages
To run the hinge GAN, a cuda gpu is required. 
```
python train.py --r 20 --k 200 --p 2000 --n 1000 --eps 0.1
```

For the ITSPCA and RegSPCA, run the following script on cpu:
```
python spca_reg_ita.py --r 20 --k 200 --p 2000 --n 1000 --eps 0
```

For both scripts, the output is the estimation error measured in squared Frobenius norm of the principal subspace projection matrix. To obtain the estimated parameters, access the `model` object after evaluation, as it stores the estimated V_hat as attributes V_hat, V_hat_ITSPCA, and V_hat_RegSPCA. 
