# Standard library imports
import os
import sys
import json
from itertools import combinations

# Third-party library imports
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader 

# Lightning imports
from lightning.pytorch.trainer import Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor

# Local application imports
from models import MODEL_CLASSES
from .data import AbstractionsDataset


def load_model(model_type, dataset, checkpoint_path=None):
    print('loading model from checkpoint path', checkpoint_path)
    model_class = MODEL_CLASSES[model_type]

    model = model_class.load_from_checkpoint(checkpoint_path, strict=False) if checkpoint_path is not None else model_class(None)
    return model


## Few Shot Learning
def few_shot_learning(args, model, num_data=2, val=True, total_epoch=20, checkpoint_name='few_shot', cwd='phier'):
    model.args.total_epoch = total_epoch
    model.configure_optimizers()

    print('---Creating Dataloaders---')
    reg_type = args.reg_type if hasattr(args, 'reg_type') else None
    few_dataset = AbstractionsDataset(args.data_dir, args.dataset, 'train', state_type='ood', num_data=num_data, similarity=args.similarity, reg_type=reg_type)
    few_dataloader = DataLoader(few_dataset, batch_size=8, shuffle=True)
    test_dataset = AbstractionsDataset(args.data_dir, args.dataset,  'test', state_type='ood', similarity=args.similarity, reg_type=reg_type)
    test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    #initialize logger + callbacks
    print('---Initializing Logger + Callback---') 
    wandb_logger = WandbLogger(
            project="abstractions_few_shot", 
            name=checkpoint_name,
            config=args,
            dir=os.path.join(cwd, 'wandb')
    )
    
    limit_val_batches = len(test_dataloader) if val else 0

    print('---Initializing Trainer---')
    trainer = Trainer(
        accelerator="auto", 
        strategy="auto", 
        callbacks=[LearningRateMonitor(logging_interval='step')],
        enable_checkpointing=False,
        max_epochs=total_epoch,
        log_every_n_steps=1,
        logger=wandb_logger,
        limit_val_batches=limit_val_batches
    )

    # evaluate on few shot
    print('---Begin Training---')
    trainer.fit(model, few_dataloader, val_dataloaders=[test_dataloader])
    
    return model

def few_shot_learning_notebook(data_dir, dataset, model, checkpoint_name, num_data=3, acc_grad=1, lr=1e-4, total_epoch=20, similarity='llm'):
    model.args.lr = lr
    model.args.total_epoch = total_epoch
    model.configure_optimizers()

    few_dataset = AbstractionsDataset(data_dir, dataset, 'train', state_type='ood', num_data=num_data, similarity=similarity)
    few_dataloader = DataLoader(few_dataset, batch_size=8, shuffle=True)
    test_dataset = AbstractionsDataset(data_dir,dataset,  'test', state_type='ood', similarity=similarity)
    test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    wandb_logger = WandbLogger(
        project=f"abstractions-{model.args.dataset}-few", 
        name=checkpoint_name,
        config=model.args,
    )
    
    print('---Initializing Trainer---')
    trainer = Trainer(
        accelerator="auto", 
        strategy="auto", 
        callbacks=[LearningRateMonitor(logging_interval='step')],
        enable_checkpointing=False,
        max_epochs=model.args.total_epoch,
        log_every_n_steps=1,
        accumulate_grad_batches=acc_grad,
        logger=wandb_logger,
    )

    # evaluate on few shot
    print('---Begin Training---')
    trainer.fit(model, few_dataloader, val_dataloaders=[test_dataloader])
    
    return model
