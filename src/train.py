import os  
import argparse
import shutil
import torch 
import torch.optim as optim
from torch.utils.data import DataLoader
from lightning import Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

from utils import AbstractionsDataset
from models import MODEL_CLASSES

def main(args, run=None):
    args.reg_type = None if args.reg_type == 'None' else args.reg_type 

    print('---Creating Dataloaders---')
    train_dataset = AbstractionsDataset(args.data_dir, args.dataset, 'train', triplet_selector=args.triplet_selector, similarity=args.similarity, reg_type=args.reg_type) 
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataset = AbstractionsDataset(args.data_dir, args.dataset, 'test', triplet_selector=args.triplet_selector, similarity=args.similarity, reg_type=args.reg_type) 
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_dataset = AbstractionsDataset(args.data_dir, args.dataset, 'test', state_type='ood', triplet_selector=args.triplet_selector, similarity=args.similarity, reg_type=args.reg_type) 
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
 
    print('---Initializing WandB---')
    run_name = f"lr={args.lr}-total_epoch={args.total_epoch}"
    save_dir = os.path.join(args.cwd, 'checkpoints', args.dataset, run_name)

    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir, exist_ok=True)

    wandb_logger = WandbLogger(
        project=f"abstractions-{args.dataset}", 
        name=run_name,
        config=args,
        dir=os.path.join(args.cwd, 'wandb'),
        group=run_name.split('/')[0]
    ) 

    print('---Initializing Callbacks---') 
    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint_callback = ModelCheckpoint(dirpath=save_dir, save_top_k=1, save_last=True, every_n_epochs=1, monitor='accuracy/val', mode='max')
    early_stopping = EarlyStopping(monitor="accuracy/val", mode="max", patience=args.patience, min_delta=args.min_delta)

    print('---Initializing Model---')
    model = MODEL_CLASSES[args.model_type](args)

    print('---Initializing Optimizers + Trainer---')
    trainer = Trainer(
        default_root_dir=args.cwd,
        accelerator="auto", 
        devices="auto", 
        strategy="auto",
        accumulate_grad_batches=args.accumulate_grad_batches, 
        logger=wandb_logger, 
        callbacks=[lr_monitor, checkpoint_callback, early_stopping],
        profiler="simple",
        max_epochs=args.total_epoch,
        log_every_n_steps=2,
        check_val_every_n_epoch=1#args.total_epoch // 20
    )
    # breakpoint()
    print('---Begin Training---')
    trainer.fit(model, train_dataloader, val_dataloaders=[val_dataloader, test_dataloader]) 


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train model') 
    parser.add_argument('--model_type', type=str, default='ObjectCentric', help='Name of the run')

    parser.add_argument('--dataset', type=str, default='calvin', help='Name of the dataset')
    parser.add_argument('--data_dir', type=str, default='data', help='Directory of the dataset')
    parser.add_argument('--cwd', type=str, default='', help='Path to the model checkpoint')
    
    parser.add_argument('--reg_type', type=str, default='relative', help='Total epochs to train')
    parser.add_argument('--reg_coeff', type=float, default=0, help='Total epochs to train')
    parser.add_argument('--reg_margin', type=float, default=0.0, help='Total epochs to train')

    # model hyperparams
    parser.add_argument('--num_fc_layers', type=int, default=1, help='Number of fully connected layers')
    parser.add_argument('--num_hyp_layers', type=int, default=0, help='Number of hyperbolic fully connected layers')
    parser.add_argument('--hyp_hidden_dim', type=int, default=256, help='Hidden dimension of the first fully connected layer')
    parser.add_argument('--hyp_output_dim', type=int, default=2, help='Output dimension of the hyperbolic encoder')
    parser.add_argument('--hyp_to_dim', type=int, default=0, help='Hidden dimension of the first fully connected layer')
    parser.add_argument('--fc_hidden_dim', type=int, default=256, help='Hidden dimension of the first fully connected layer')

    # training hyperparams
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for dataloader')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--multiplier', type=int, default=1, help='Multiplier for warmup scheduler')
    parser.add_argument('--total_epoch', type=int, default=100, help='Total epochs to train')
    parser.add_argument('--optimizer', type=str, default='adam', help='Hyperbolic representation space')
    parser.add_argument('--accumulate_grad_batches', type=int, default=15, help='Hyperbolic representation space')
    parser.add_argument('--patience', type=int, default=25, help='Hyperbolic representation space')
    parser.add_argument('--min_delta', type=int, default=0.05, help='Hyperbolic representation space')

    # loss hyperparams
    parser.add_argument('--similarity', type=str, default='llm', help='Distance metric')
    parser.add_argument('--sup_coeff', type=float, default=1.0, help='Supervised loss coefficient')
    parser.add_argument('--ssl_coeff', type=float, default=1.0, help='Self supervised loss coefficient')
    parser.add_argument('--margin', type=float, default=1.0, help='Margin for triplet loss')
    parser.add_argument('--triplet_selector', type=str, default='random', help='Triplet selector')

    parser.add_argument('--is_hyperbolic', type=bool, default=True, help='Hyperbolic representation space')
    parser.add_argument('--hyp_nonlin', type=str, default=None, help='Hyperbolic representation space')
    args = parser.parse_args()

    main(args)
