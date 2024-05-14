import torch
from metrics.wasserstein import *
import time

def OBSW(Xs,X,L=10,lam=1,p=2,device="cpu"):
    dim = X.size(1)
    theta = rand_projections(dim=dim, num_projections=L, device=device)
    Xs_tensor = torch.cat(Xs, dim=0)
    Xs_prod = torch.matmul(Xs_tensor, theta.transpose(0, 1))
    X_prod = torch.matmul(X, theta.transpose(0, 1))
    Xs_prod = Xs_prod.reshape(len(Xs), Xs_tensor.shape[0] // len(Xs), L)
    Xs_prod_sorted = torch.sort(Xs_prod,dim=1)[0]
    X_prod_sorted = torch.sort(X_prod, dim=0)[0]
    wasserstein_distance = torch.abs(Xs_prod_sorted - X_prod_sorted)
    sw = torch.mean(torch.pow(wasserstein_distance, p), dim=1)
    sw = torch.mean(sw,dim=1)
    return torch.mean(sw) + lam * torch.cdist(sw.view(-1,1), sw.view(-1,1), p=1).sum() / (sw.shape[0]*sw.shape[0] - sw.shape[0])


def OBSW_list(Xs,X,L=10,lam=1,p=2,device="cpu"):
    dim = X.size(1)
    theta = rand_projections(dim=dim, num_projections=L, device=device)
    sw = [torch.mean(one_dimensional_Wasserstein(Xs[i], X, theta, p=p))for i in range(len(Xs))]
    sw = torch.tensor(sw)
    return torch.mean(sw) + lam * torch.cdist(sw.view(-1,1), sw.view(-1,1), p=1).sum() / (sw.shape[0]*sw.shape[0] - sw.shape[0])



num_proj = 3

list_x = list()
for i in range(10):
    x_marginal = torch.randn(2, 4)
    list_x.append(x_marginal)

x_barycenter = torch.randn(2, 4)
theta = rand_projections(dim=4, num_projections=num_proj, device="cpu")

start_time = time.time()

o1 = OBSW_list(Xs=list_x, X=x_barycenter, L=num_proj, lam=1, p=2, theta=theta)

# Calculate elapsed time for OBSW_list
elapsed_time_o1 = time.time() - start_time

# Start timing for OBSW
start_time = time.time()

o2 = OBSW(Xs=list_x, X=x_barycenter, L=num_proj, lam=1, p=2, theta=theta)

# Calculate elapsed time for OBSW
elapsed_time_o2 = time.time() - start_time

print("Execution time for OBSW_list:", elapsed_time_o1)
print("Execution time for OBSW:", elapsed_time_o2)
print(o1, o2)