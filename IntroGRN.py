import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.dataset import TensorDataset
from models import IntroGRN
from evaluate import get_metrics
from tqdm import tqdm
from logger import LightLogger
def runIntroGRN(exp_array, configs,
               ground_truth=None, logger=None, progress_bar=False):
    '''
    Initialize and Train a IntroGRN model with configs
    '''
    if configs['early_stopping'] != 0 and configs['train_split'] == 1.0:
        raise Exception(
            "You indicate early stopping but you have not specified any ",
            "validation data. Consider decrease your train_split. ")
    es = configs['early_stopping']

    n_obs, n_gene = exp_array.shape

    # Logger -------------------------------------------------------------------
    if logger is None:
        logger = LightLogger()
    logger.set_configs(configs)
    note_id = logger.start()

    # Global Mean/Std ----------------------------------------------------------
    global_mean = torch.FloatTensor(exp_array.mean(0))
    global_std = torch.FloatTensor(exp_array.std(0))

    # Train/Test split if requested --------------------------------------------
    assert configs['train_split'] > 0 and configs['train_split'] <= 1, \
        f'Expect 0<configs["train_split"]<=1'
    has_train_test_split = (configs['train_split'] != 1.0)

    if configs['train_split_seed'] is None:
        train_mask = np.random.rand(n_obs)
    else:
        rs = np.random.RandomState(seed=configs['train_split_seed'])
        train_mask = rs.rand(n_obs)

    train_dt = TensorDataset(
        torch.FloatTensor(exp_array[train_mask <= configs['train_split'],]),
    )
    train_loader = DataLoader(
        train_dt, batch_size=configs['batch_size'], shuffle=True)

    if has_train_test_split:
        val_dt = TensorDataset(
            torch.FloatTensor(exp_array[train_mask > configs['train_split'],]),
        )
        val_loader = DataLoader(
            val_dt, batch_size=configs['batch_size'], shuffle=True)


    vae = IntroGRN(
        n_gene=n_gene,
        hidden_dim=configs['hidden_dim'], z_dim=configs['z_dim'],
        A_dim=configs['A_dim'],
        train_on_non_zero=configs['train_on_non_zero'],
        dropout_augmentation_p=configs['dropout_augmentation_p'],
        dropout_augmentation_type=configs['dropout_augmentation_type']
        # A_dim=configs['A_dim']
    )

    if configs['cuda']:
        global_mean = global_mean.cuda()
        global_std = global_std.cuda()
        vae = vae.cuda()

    if configs['number_of_opt'] == 2:
        opt_nn_e = torch.optim.Adam(vae.inference_zposterior.parameters(), lr=configs['lr_nn'])
        opt_nn_d = torch.optim.Adam(vae.generative_pxz.parameters(), lr=configs['lr_nn'])
        opt_adj = torch.optim.Adam([vae.adj_A], lr=configs['lr_adj'], weight_decay=0.00, betas=[0.9, 0.9])

    es_tracks = []
    adjs = []
    disable_tqdm = (progress_bar == False)
    for epoch in tqdm(range(configs['n_epochs']), disable=disable_tqdm):
        if configs['number_of_opt'] == 2:
            vae.train(True)
            # iteration_for_A = epoch % (configs['K1'] + configs['K2']) >= configs['K1']
            # vae.adj_A.requires_grad = iteration_for_A
            evaluation_turn = (epoch % configs['eval_on_n_steps'] == 0)

            # go through training samples
            eval_log = {
                'epoch': 0, 'loss_rec': 0, 'loss_kl': 0, 'loss_sparse': 0,
                 'loss': 0, 'loss_ma': 0, 'lossG': 0
            }
            for i, batch in enumerate(train_loader):
                exp = batch[0]
                noise = torch.randn_like(exp)
                if configs['cuda']:
                    exp = exp.cuda()
                    noise = noise.cuda()

                # (1)train VAE
                opt_adj.zero_grad()
                opt_nn_e.zero_grad()

                out_real = vae(exp, global_mean, global_std)

                z_posterior = vae.inference_zposterior(out_real['x_rec'].detach().unsqueeze(-1))
                z_posterior = torch.einsum('ogd,agh->ohd', z_posterior, out_real['IA'])

                z_mu = z_posterior[:, :, :vae.z_dim]
                z_logvar = z_posterior[:, :, vae.z_dim:]
                lossE_kl = -0.5 * torch.mean(1 + z_logvar - z_mu.pow(2) - torch.exp(z_logvar))

                lossE_rec_kl = lossE_kl
                lossE_real_kl = out_real['loss_kl']

                m_plus = configs['marginal']
                weight_neg = 0.5

                loss_margin = lossE_real_kl + (F.relu(m_plus - lossE_rec_kl)) * 0.5 * weight_neg

                alpha_2 = 1

                lossE = loss_margin + alpha_2 * (out_real['loss_rec'])

                adj_m = vae.get_adj_()
                loss_sparse = torch.norm(adj_m, 1) / n_gene / n_gene
                if epoch >= configs['delayed_steps_on_sparse']:
                    lossE += configs['alpha'] * loss_sparse

                lossE.backward(retain_graph=True)

                opt_adj.step()
                opt_nn_e.step()
                opt_nn_d.step()

                # -------lry updata G------

                opt_adj.zero_grad()
                opt_nn_d.zero_grad()

                z_posterior_G = vae.inference_zposterior(out_real['x_rec'].detach().unsqueeze(-1))
                z_posterior_G = torch.einsum('ogd,agh->ohd', z_posterior_G, out_real['IA'])

                z_mu_G = z_posterior_G[:, :, :vae.z_dim]
                z_logvar_G = z_posterior_G[:, :, vae.z_dim:]
                lossG_kl = -0.5 * torch.mean(1 + z_logvar_G - z_mu_G.pow(2) - torch.exp(z_logvar_G))

                lossG_rec_kl = lossG_kl

                lossG = configs['beta'] * lossG_rec_kl

                lossG = lossG + out_real['loss_rec'].detach()

                lossG.backward()

                opt_adj.step()
                opt_nn_d.step()
                # opt_nn_c.step()

                eval_log['epoch'] += epoch  # lry add
                eval_log['loss_rec'] += round(out_real['loss_rec'].detach().cpu().item(), 5)
                eval_log['loss_kl'] += round(out_real['loss_kl'].detach().cpu().item(), 5)
                eval_log['loss_sparse'] += round(loss_sparse.detach().cpu().item(), 5)
                eval_log['loss_ma'] += round(loss_margin.detach().cpu().item(), 5)
                eval_log['loss'] += lossE.detach().cpu().item()
                eval_log['lossG'] += lossG.detach().cpu().item()
                # eval_log['lossG'] += 0


        for log_item in eval_log.keys():
            # eval_log[log_item] /= (i+1)#lry change
            eval_log[log_item] = round(eval_log[log_item] / (i + 1), 5)

        # go through val samples
        if evaluation_turn:
            adj_matrix = adj_m.cpu().detach().numpy()
            adjs.append(adj_matrix)
            eval_log['negative_adj'] = int(np.sum(adj_matrix < -1e-5))

            # adjs.append(adj_m2)
            if ground_truth is not None:
                epoch_perf = get_metrics(adj_matrix, ground_truth)
                for k in epoch_perf.keys():
                    eval_log[k] = epoch_perf[k]

            if has_train_test_split:
                eval_log['val_loss_rec'] = 0
                eval_log['val_loss_kl'] = 0
                eval_log['val_loss_sparse'] = 0
                vae.train(False)
                for batch in val_loader:
                    x = batch[0]
                    if configs['cuda']:
                        x = x.cuda()
                    out = vae(x, global_mean, global_std)
                    eval_log['val_loss_rec'] += out['loss_rec'].detach().cpu().item()
                    eval_log['val_loss_kl'] += out['loss_kl'].detach().cpu().item()
                    eval_log['val_loss_sparse'] += out['loss_sparse'].detach().cpu().item()
                if epoch >= configs['delayed_steps_on_sparse']:
                    es_tracks.append(eval_log['val_loss_rec'])

            # logger.log(eval_log)
            # lry change
            print(eval_log)
            # early stopping
            if (es > 0) and (len(es_tracks) > (es + 2)):
                if min(es_tracks[(-es - 1):]) < min(es_tracks[(-es):]):
                    print('Early stopping triggered')
                    break

    rec_xs = []

    logger.finish()
    vae = vae.cpu()
    # vae.classifier_pos_weight = vae.classifier_pos_weight.cpu()
    return vae, adjs, rec_xs

