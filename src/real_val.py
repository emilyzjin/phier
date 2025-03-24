import os
import argparse
import json

from datetime import datetime
from lightning import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor

from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader

# Local application imports
from utils import load_model, compute_accuracy, AbstractionsDataset

def main(args):
    date = datetime.today().strftime('%m-%d')

    print('---Initializing Model---')
    checkpoint_dir = os.path.join(args.checkpoint_dir, args.dataset, args.checkpoint_name)
    checkpoint = sorted([f for f in os.listdir(checkpoint_dir) if f[:4] != 'last'])[-1:][0]

    checkpoint_path = os.path.join(checkpoint_dir, checkpoint)

    params = {p.split('=')[0]: p.split('=')[1] for p in args.checkpoint_name.split('/')[0].split('-')}
    print(args.checkpoint_name.split('/')[-1])
    hyperparams = {p.split('=')[0]: p.split('=')[1] for p in args.checkpoint_name.split('/')[-1].split('-')}
    args.model_type = 'EndToEnd' if params['prior_mode'] == 'None' else 'ObjectCentric'

    model = load_model(args.model_type, args.dataset, checkpoint_path)

    print('---Creating Dataloaders---')
    model.args.total_epoch = args.total_epoch
    model.configure_optimizers()

    test_dataset = AbstractionsDataset(args.data_dir, 'real_behavior',  'test', state_type='ood', test_only=True)
    test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    #initialize logger + callbacks
    print('---Initializing Logger + Callback---') 
    wandb_logger = WandbLogger(
            project="abstractions-real_behavior", 
            name=args.checkpoint_name,
            config=args,
            dir=os.path.join('/vision/u/emilyjin/abstractions', 'wandb')
    )

    print('---Initializing Trainer---')
    trainer = Trainer(
        accelerator="auto", 
        strategy="auto", 
        callbacks=[LearningRateMonitor(logging_interval='step')],
        enable_checkpointing=False,
        max_epochs=args.total_epoch,
        log_every_n_steps=1,
        logger=wandb_logger,
        limit_val_batches=len(test_dataloader)
    )

    if args.num_data > 0:
        few_dataset = AbstractionsDataset(args.data_dir, 'behavior', 'test', state_type='ood', num_data=args.num_data, similarity=model.args.similarity, reg_type=model.args.reg_type)
        few_dataloader = DataLoader(few_dataset, batch_size=8, shuffle=True)
        trainer.fit(model, few_dataloader, val_dataloaders=[test_dataloader])
    
    results = compute_accuracy(model, test_dataloader)
    
    print('---Saving Results---')
    print(results)

    overall_accs = {}
    gen_types = json.load(open('/vision/u/emilyjin/abstractions/data/real_behavior/generalization_type.json', 'r'))
    for gen_type, states in gen_types.items():
        accs = [results[state] for state in states if state in results.keys()]
        acc = sum(accs) / len(accs)

        overall_accs[gen_type] = acc
 
    overall_accs['overall'] = results['overall']
    
    for gen_type, acc in overall_accs.items():
        accuracy_file = os.path.join(args.results_dir, f"real_behavior-{gen_type}.txt")
        
        with open(accuracy_file, 'a+') as f:
            f.write(f'{args.checkpoint_name}-{args.num_data}_shot: {str(acc)}\n')

    # save_path = os.path.join(args.results_dir, date, args.dataset, f"{args.checkpoint_name.replace('/', '--')}.json")
    # os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # with open(save_path, 'w') as f:
    #     json.dump(results, f, indent=4)
    
    print('---Done---') 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test model') 
    parser.add_argument('--model_type', type=str, default='EndToEnd', help='Type of model to test')

    parser.add_argument('--dataset', type=str, default='behavior', help='Directory of the dataset')
    parser.add_argument('--data_dir', type=str, default='/viscam/u/emilyjin/abstractions/data', help='Directory of the dataset')
    parser.add_argument('--checkpoint_dir', type=str, default='/vision/u/emilyjin/abstractions/checkpoints', help='Path to directory of model checkpoints')
    parser.add_argument('--checkpoint_name', type=str, default='is_hyperbolic=True-prior_mode=bottleneck-reg_type=absolute-similarity=llm_predicate//num_hyp_layers=2-hyp_hidden_dim=512-hyp_output_dim=2-num_fc_layers=1-fc_hidden_dim=0-ssl_coeff=0.05-margin=10.0-reg_coeff=1.0-reg_margin=0.0-lr=0.0001-total_epoch=50', help='Name of the model checkpoint')
    parser.add_argument('--num_data', type=int, default=5, help='Name of the model checkpoint')
    parser.add_argument('--total_epoch', type=int, default=50, help='Name of the model checkpoint')
    parser.add_argument('--results_dir', type=str, default='/vision/u/emilyjin/abstractions/accuracies/real_behavior', help='Directory to save visual')
    args = parser.parse_args()

    main(args)
