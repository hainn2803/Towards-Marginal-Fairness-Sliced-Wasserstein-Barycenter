import torch
from metrics.wasserstein import *
import torch.nn as nn

class SWAEBatchTrainer:
    """ Sliced Wasserstein Autoencoder Batch Trainer.

        Args:
            autoencoder (torch.nn.Module): module which implements autoencoder framework
            optimizer (torch.optim.Optimizer): torch optimizer
            distribution_fn (callable): callable to draw random samples
            num_projections (int): number of projections to approximate sliced metrics distance
            p (int): power of distance metric
            weight (float): weight of divergence metric compared to reconstruction in loss
            device (torch.Device): torch device
    """

    def __init__(self, autoencoder, optimizer, distribution_fn, num_classes=10,
                 num_projections=200, p=2, weight_swd=1, weight_fsw=1, device=None, method="FEFBSW", lambda_obsw=1.):
        self.model_ = autoencoder
        self.optimizer = optimizer
        self._distribution_fn = distribution_fn
        self.embedding_dim_ = self.model_.encoder.embedding_dim_
        self.num_projections_ = num_projections
        self.p_ = p
        self._device = device if device else torch.device('cpu')
        self.num_classes = num_classes

        self.weight = weight_swd
        self.weight_fsw = weight_fsw

        self.method = method
        self.lambda_obsw = lambda_obsw

        self.criterion = nn.CrossEntropyLoss().to(self._device)

    def train_on_batch(self, x, y):
        # reset gradients
        self.optimizer.zero_grad()
        # autoencoder forward pass and loss
        evals = self.eval_on_batch(x, y)
        # backpropagate loss
        evals['loss'].backward()
        # update encoder and decoder parameters
        self.optimizer.step()
        return evals

    def test_on_batch(self, x, y):
        with torch.no_grad():
            self.optimizer.zero_grad()
            x = x.to(self._device)
            y = y.to(self._device)
            logits, z_latent = self.model_(x)
            pred_loss = self.criterion(logits, y)
            y_pred = torch.argmax(logits, dim=1)

            # acc = torch.sum(y_pred == y) / y.shape[0]
            batch_size = x.size(0)
            z_prior = self._distribution_fn(batch_size).to(self._device)

            swd = sliced_wasserstein_distance(encoded_samples=z_latent, distribution_samples=z_prior,
                                              num_projections=self.num_projections_, p=self.p_,
                                              device=self._device)

            loss = pred_loss + float(self.weight) * swd

            return {
            'loss': loss,
            'pred_loss': pred_loss,
            'swd': swd,
            'z_latent': z_latent,
            'y_pred': y_pred,
            'logits': logits
            }

    def eval_on_batch(self, x, y):

        x = x.to(self._device)
        y = y.to(self._device)

        logits, z_latent = self.model_(x)
        pred_loss = self.criterion(logits, y)

        batch_size = x.size(0)
        z_prior = self._distribution_fn(batch_size).to(self._device)

        swd = sliced_wasserstein_distance(encoded_samples=z_latent, distribution_samples=z_prior,
                                          num_projections=self.num_projections_, p=self.p_,
                                          device=self._device)

        list_z_posterior = list()
        for cls in range(self.num_classes):
            list_z_posterior.append(z_latent[y == cls])

        if self.method == "FEFBSW":
            fsw = FEFBSW_list(Xs=list_z_posterior, X=z_prior, L=self.num_projections_, device=self._device)
        elif self.method == "lowerbound_FEFBSW":
            fsw = lowerbound_FEFBSW_list(Xs=list_z_posterior, X=z_prior, L=self.num_projections_, device=self._device)

        elif self.method == "EFBSW":
            fsw = EFBSW_list(Xs=list_z_posterior, X=z_prior, L=self.num_projections_, device=self._device)
        elif self.method == "lowerbound_EFBSW":
            fsw = lowerbound_EFBSW_list(Xs=list_z_posterior, X=z_prior, L=self.num_projections_, device=self._device)

        elif self.method == "FBSW":
            fsw = FBSW_list(Xs=list_z_posterior, X=z_prior, L=self.num_projections_, device=self._device)
        elif self.method == "lowerboundFBSW":
            fsw = lowerboundFBSW_list(Xs=list_z_posterior, X=z_prior, L=self.num_projections_, device=self._device)

        elif self.method == "BSW":
            fsw = BSW_list(Xs=list_z_posterior, X=z_prior, L=self.num_projections_, device=self._device)

        elif self.method == "OBSW":
            fsw = OBSW_list(Xs=list_z_posterior, X=z_prior, L=self.num_projections_, lam=self.lambda_obsw, device=self._device)
        else:
            fsw = 0

        loss = pred_loss + float(self.weight_fsw) * fsw + float(self.weight) * swd

        return {
            'loss': loss,
            'pred_loss': pred_loss,
            'fsw': fsw,
            'swd': swd,
            'z_latent': z_latent
        }

    def forward(self, x):
        x = x.to(self._device)
        logits, z_latent = self.model_(x)
        return {
            'logits': logits,
            'z_latent': z_latent
        }
