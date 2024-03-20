import numpy as np
from data import load_beeline
from logger import LightLogger
from IntroGRN import runIntroGRN
from evaluate import extract_edges, get_metrics

DEFAULT_GRNVAE_CONFIGS = {
    # Train/Test split
    'train_split': 1.0,
    'train_split_seed': None,

    # Neural Net Definition
    'hidden_dim': 128,
    'z_dim': 1,
    'A_dim': 1,
    'train_on_non_zero': True,
    'dropout_augmentation_p': 0.1,
    'dropout_augmentation_type': 'all',
    'cuda': True,

    # Loss
    'alpha': 100,
    'beta': 1,
    'chi': 0.5,
    'h_scale': 0,
    'delayed_steps_on_sparse': 30,

    # Neural Net Training
    'number_of_opt': 2,
    'batch_size': 64,
    'n_epochs': 120,
    'marginal':10,
    # 'schedule': [120, 240],
    'eval_on_n_steps': 10,
    'early_stopping': 0,
    'lr_nn': 1e-4,
    'lr_adj': 2e-5,
    'K1': 1,
    'K2': 1
}

for jj in range(1):
    import pandas as pd
    import time
    list_result=[]
    data, ground_truth = load_beeline(
        data_dir='data',
        benchmark_data='hESC', #hESC,hHep,mDC,mESC,mHSC-E,mHSC-GM,mHSC-L
        benchmark_setting='500_STRING'#500_STRING,1000_STRING,1000_ChIP-seq,500_ChIP-seq,1000_Non-ChIP,500_Non-ChIP
    )
    print(data)
    logger = LightLogger()
    start_time = time.time()

    vae, adjs, result_rec = runIntroGRN(
        data.X, DEFAULT_GRNVAE_CONFIGS, ground_truth=ground_truth, logger=logger)

    A = vae.get_adj()
    end_time = time.time()
    total_time = end_time - start_time  # seconds
    ppi_auc = get_metrics(A, ground_truth)
    print(ppi_auc)

