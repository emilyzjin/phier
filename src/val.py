import os
import argparse
import json

from lightning import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from datetime import datetime

from utils import load_model, few_shot_learning, create_dataloaders, compute_accuracy, visualize_embeddings

def main(args):
    try:
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
        in_dataloader, ood_dataloader = create_dataloaders(args.data_dir, args.dataset, 'test', num_data=None, similarity=model.args.similarity, reg_type=model.args.reg_type)

        if args.get_accuracy:
            print('---Getting Accuracy---')
            results = {}
            results['in'] = compute_accuracy(model, in_dataloader)
            results['ood-zero_shot'] = compute_accuracy(model, ood_dataloader)

        if args.visualize_embeddings: 
            print('---Visualizing Embeddings---') 
            visualize_embeddings(model, in_dataloader, output_path=f"{args.vis_dir}/embeddings/{args.dataset}/{args.checkpoint_name}--in.png", by_pred=args.by_pred)
 
        if args.few_shot:
            lr_monitor = LearningRateMonitor(logging_interval='step')
            early_stopping = EarlyStopping(monitor="accuracy/val", mode="max", patience=5)

            print('---Initializing Trainer---')
            trainer = Trainer(
                accelerator="auto",
                devices="auto",
                strategy="auto", 
                logger=True,
                log_every_n_steps=50,
                callbacks=[lr_monitor, early_stopping]
            )
            
            print('---Doing few shot learning now---')
            model = few_shot_learning(model.args, model, num_data=args.num_data, total_epoch=args.total_epoch, val=False)

            if args.get_accuracy:
                print('---Getting Accuracy---') 
                results[f'ood-{args.num_data}_shot'] = compute_accuracy(model, ood_dataloader)

            if args.visualize_embeddings: 
                print('---Visualizing Embeddings---')
                visualize_embeddings(model, ood_dataloader, output_path=f"{args.vis_dir}/embeddings/{args.dataset}/{args.checkpoint_name}-ood_{args.num_data}_shot.png", by_pred=args.by_pred)

        if args.get_accuracy:
            print('---Results---')
            print(results)

            print('---Saving Results---')
            save_path = os.path.join(args.results_dir, args.dataset, f"{args.checkpoint_name}-{args.num_data}_shot.json")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            if os.path.exists(save_path):
                with open(save_path, 'r') as f:
                    old_results = json.load(f)
                    old_results.update(results)
                    results = old_results

            with open(save_path, 'w') as f:
                json.dump(results, f, indent=4)
        
        print('---Done---')
        
        with open(os.path.join('/vision/u/emilyjin/abstractions', 'completed_val.txt'), 'a') as f:
            # command = '--checkpoint_name {} {}'.format(args.checkpoint_name, ' '.join(['--' + k + ' ' + v for k, v in vars(args).items() if v]))
            f.write(f'{args.checkpoint_name} \n')
    except:
        with open(os.path.join('/vision/u/emilyjin/abstractions', 'failed_val.txt'), 'a') as f:
            f.write(f'{args.checkpoint_name} \n')
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test model') 
    parser.add_argument('--model_type', type=str, default='EndToEnd', help='Type of model to test')

    parser.add_argument('--dataset', type=str, default='behavior', help='Directory of the dataset')
    parser.add_argument('--data_dir', type=str, default='/vision/u/emilyjin/abstractions/data', help='Directory of the dataset')
    parser.add_argument('--checkpoint_dir', type=str, default='/vision/u/emilyjin/abstractions/checkpoints', help='Path to directory of model checkpoints')
    parser.add_argument('--checkpoint_name', type=str, default='checkpoints/calvin/is_hyperbolic=True-prior_mode=bottleneck-reg_type=None-similarity=llm_predicate', help='Name of the model checkpoint')

    parser.add_argument('--num_data', type=int, default=5, help='Name of the model checkpoint')
    parser.add_argument('--total_epoch', type=int, default=20, help='Name of the model checkpoint')

    parser.add_argument('--get_accuracy', action='store_true', help='Get accuracy of model on test set')
    parser.add_argument('--visualize_embeddings', action='store_true', help='Get accuracy of model on test set') 

    parser.add_argument('--few_shot', action='store_true', help='Get accuracy of model on test set')


    parser.add_argument('--by_pred', action='store_true', help='Get accuracy of model on test set') 

    parser.add_argument('--vis_dir', type=str, default='/viscam/u/emilyjin/abstractions/visualizations', help='Directory to save visual')
    parser.add_argument('--results_dir', type=str, default='/vision/u/emilyjin/abstractions/accuracies', help='Directory to save visual')
    args = parser.parse_args()

    main(args)
