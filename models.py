import torch
from torch import nn
from nearest_embed import NearestEmbed,NearestEmbedEMA
class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, activation):
        super(MLP, self).__init__()
        self.l1 = nn.Linear(in_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, out_dim)
        self.act = activation()

    def forward(self, x):
        out1 = self.act(self.l1(x))
        out2 = self.act(self.l2(out1))
        return self.l3(out2)


class IntroGRN(nn.Module):
    def __init__(
            self, n_gene, hidden_dim=128, z_dim=1, A_dim=1,
            activation=nn.Tanh, train_on_non_zero=False,
            dropout_augmentation_p=0.1, dropout_augmentation_type='all',
            pretrained_A=None,
    ):
        super(IntroGRN, self).__init__()
        self.n_gene = n_gene
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim
        self.A_dim = A_dim
        self.train_on_non_zero = train_on_non_zero

        if pretrained_A is None:
            adj_A = torch.ones(A_dim, n_gene, n_gene) / (n_gene - 1)
            adj_A += torch.rand_like(adj_A) * 0.0002
        else:
            adj_A = pretrained_A
        self.adj_A = nn.Parameter(adj_A, requires_grad=True)

        self.inference_zposterior = MLP(1, hidden_dim, z_dim * 2, activation)
        self.generative_pxz = MLP(z_dim, hidden_dim, 1, activation)
        self.da_p = dropout_augmentation_p
        self.da_type = dropout_augmentation_type
        # classifier_pos_weight = torch.FloatTensor([1.0])
        # if self.da_p != 0:
        #     classifier_pos_weight *= (1 - self.da_p) / self.da_p
        # self.classifier_pos_weight = nn.Parameter(
        #     classifier_pos_weight, requires_grad=False
        # )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()

    def get_adj_(self):
        eye_tensor = torch.eye(
            self.n_gene, device=self.adj_A.device
        ).repeat(self.A_dim, 1, 1)
        mask = torch.ones_like(self.adj_A) - eye_tensor
        return (self.adj_A * mask).mean(0)

    def get_adj(self):
        return self.get_adj_().cpu().detach().numpy()

    def I_minus_A(self):
        eye_tensor = torch.eye(
            self.n_gene, device=self.adj_A.device
        ).repeat(self.A_dim, 1, 1)
        # clean up A along diagnal line
        mask = torch.ones_like(self.adj_A) - eye_tensor
        clean_A = self.adj_A * mask
        return eye_tensor - clean_A

    def reparameterization(self, z_mu, z_sigma):
        return z_mu + z_sigma * torch.randn_like(z_sigma)

    @torch.no_grad()

    def add_gaussian_noise(self,matrix, mean=0, std=1):
        device=matrix.device
        noise = torch.randn(matrix.shape,device=device) * std + mean
        noisy_matrix = matrix + noise
        return noisy_matrix
    def dropout_augmentation2(self, x, global_mean,da_p):
        da_mask = (torch.rand_like(x) < da_p)
        if self.da_type == 'belowmean':
            da_mask = da_mask * (x < global_mean)
        elif self.da_type == 'belowhalfmean':
            da_mask = da_mask * (x < (global_mean / 2))
        elif self.da_type == 'all':
            da_mask = da_mask
        noise =  x * da_mask  # change
        x = x - noise
        return x, noise, da_mask
    def forward(self, x, global_mean,global_std, normal='z-score',add_gaussian=False,use_dropout_augmentation=True):
        if self.train_on_non_zero:
            eval_mask = (x != 0)
        else:
            eval_mask = torch.ones_like(x)

        x_init = x
        if use_dropout_augmentation:
            x, noise, da_mask = self.dropout_augmentation2(x_init, global_mean,self.da_p)

        else:
            noise = torch.zeros_like(x)
            da_mask = (noise == 1)

        if normal=='z-score':
            x = (x - global_mean) / (global_std)
            # noise = (noise - global_mean) / (global_std)
            x[torch.isnan(x)]=0
            x[torch.isinf(x)]=0

        else:
            x=x
            # noise=noise

        # Encoder --------------------------------------------------------------
        I_minus_A = self.I_minus_A()

        z_posterior = self.inference_zposterior(x.unsqueeze(-1))
        z_posterior = torch.einsum('ogd,agh->ohd', z_posterior, I_minus_A)

        z_mu = z_posterior[:, :, :self.z_dim]
        z_logvar = z_posterior[:, :, self.z_dim:]
        z = self.reparameterization(z_mu, torch.exp(z_logvar * 0.5))

        # Decoder --------------------------------------------------------------
        z_inv = torch.einsum('ogd,agh->ohd', z, torch.inverse(I_minus_A))
        x_rec = self.generative_pxz(z_inv).squeeze(2)

        # Losses ---------------------------------------------------------------
        loss_rec_all = (x - x_rec).pow(2)
        loss_rec = torch.sum(loss_rec_all * eval_mask)
        loss_rec = loss_rec / torch.sum(eval_mask)

        loss_kl = -0.5 * torch.mean(1 + z_logvar - z_mu.pow(2) - torch.exp(z_logvar))

        out = {
            'loss_rec': loss_rec, 'loss_kl': loss_kl,
            'z_posterior': z_posterior, 'z': z, 'x_rec': x_rec,
             'da_mask': da_mask,
            'norm_x':x,'z_inv':z_inv,'z_mu':z_mu,'IA':I_minus_A,'z_logvar':z_logvar
        }
        return out
