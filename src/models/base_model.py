import os  
import copy
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.init as init
import lightning as L
from geoopt.manifolds import PoincareBall 
from transformers import BertModel, BertTokenizer, ViTModel, ViTConfig
from warmup_scheduler import GradualWarmupScheduler

from lightning import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor
import math

from utils import AbstractionsDataset
from torch.utils.data import DataLoader

from transformers import AutoTokenizer, AutoProcessor, CLIPModel
from torchvision.transforms.functional import to_pil_image 

class EndToEndModel(L.LightningModule):
    def __init__(self, args=None): 
        super(EndToEndModel, self).__init__()
        self.data_dir = args.data_dir  
        self.dataset = args.dataset if args is not None else 'behavior'
        self.QUERY_TO_SENTENCE = json.load(open(os.path.join(self.data_dir, self.dataset, 'query_to_sentence.json'), 'r'))
        self.SENTENCE_TO_QUERY = {v: k for k, v in self.QUERY_TO_SENTENCE.items()}

        self.args = args  

        self.is_hyperbolic = args.is_hyperbolic if args is not None else False
        self.hyp_nonlin = nn.ReLU() if args is not None and args.hyp_nonlin == 'relu' else None

        num_fc_layers = args.num_fc_layers if args is not None else 1
        num_hyp_layers = args.num_hyp_layers if (args is not None and hasattr(args, "num_hyp_layers")) else 2

        # initialize model
        # vision encoder: ViT-B with sine-cosine positional embeddings
        vit_config = ViTConfig.from_pretrained('google/vit-base-patch16-224-in21k')
        self.vision_encoder = ViTModel(vit_config)
        
        # Replace positional embeddings with sine-cosine positional embeddings
        num_patches = (224 // vit_config.patch_size) ** 2
        sine_cosine_embeddings = self.get_sine_cosine_positional_embeddings(num_patches, vit_config.hidden_size)
        self.vision_encoder.embeddings.position_embeddings = nn.Parameter(sine_cosine_embeddings, requires_grad=False)
    
        # text encoder: BERT
        self.text_encoder = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        input_dim = self.vision_encoder.config.hidden_size + self.text_encoder.config.hidden_size
        hyp_hidden_dim = args.hyp_hidden_dim if (args is not None and hasattr(args, "hyp_hidden_dim")) else 512
        hyp_output_dim = args.hyp_output_dim if (args is not None and hasattr(args, "hyp_output_dim")) else 2
        fc_hidden_dim = args.fc_hidden_dim if (args is not None and hasattr(args, "fc_hidden_dim")) else 512
        
        # hyperbolic layers
        if self.is_hyperbolic:
            self.manifold = PoincareBall()
            self.hyp_layers = nn.ModuleList()
             
            self.hyp_to_ball = None
            
            if num_hyp_layers > 0:
                for _ in range(num_hyp_layers - 1):
                    self.hyp_layers.append(MobiusLinear(input_dim, hyp_hidden_dim, self.manifold)) 
                    input_dim = hyp_hidden_dim
                    hyp_hidden_dim = int(hyp_hidden_dim / 2)
                
                self.hyp_layers.append(MobiusLinear(input_dim, hyp_output_dim, self.manifold)) 

                input_dim = hyp_output_dim
        else:
            self.manifold = None
            self.hyp_layers = None
            self.hyp_to_ball = None

        # fc layers
        self.fc_layers = nn.ModuleList()

        for _ in range(num_fc_layers - 1):
            self.fc_layers.append(nn.Linear(input_dim, fc_hidden_dim))
            input_dim = fc_hidden_dim
            fc_hidden_dim = int(fc_hidden_dim / 2)
        
        self.fc_layers.append(nn.Linear(input_dim, 2))

        # set loss fns
        self.similarity = args.similarity if args is not None else 'llm'
        self.sup_coeff = args.sup_coeff if args is not None else 1
        self.ssl_coeff = args.ssl_coeff if args is not None else 1
        self.margin = args.margin if args is not None else 1
        self.reg_coeff = args.reg_coeff if args is not None and hasattr(args, 'reg_coeff') else 0 
        self.reg_type = args.reg_type if args is not None and hasattr(args, 'reg_type') else 'relative'
        self.reg_margin = args.reg_margin if args is not None and hasattr(args, 'reg_margin') else 0.0

        self.sup_loss_fn = nn.CrossEntropyLoss() 
        if self.is_hyperbolic: 
            self.ssl_loss_fn = nn.TripletMarginWithDistanceLoss(distance_function=self.manifold.dist, margin=self.margin)
        else:
            self.ssl_loss_fn = nn.TripletMarginLoss(margin=self.margin) 

        self.save_hyperparameters()

        self.similarities = {}

        self.log_mode = 'val'
        
        if self.similarity is not None:
            self.load_similarities(filepath=f'/vision/u/emilyjin/abstractions/src/assets/similarities/{self.dataset}/{self.similarity}_similarities.json')

    def parse_query_states(self, states):
        # Given a unary state pred(obj1) or binary state pred(obj1, obj2), parse the state into the predicate and the objects
        preds = []
        objss = []
        for state in states:
            pred = state.split('(')[0]
            if ',' in state:
                obj1 = state.split('(')[1].split(', ')[0]
                obj2 = state.split('(')[1].split(', ')[1].split(')')[0]
                objs = [obj1, obj2]
            else:
                objs = [state.split('(')[1].split(',')[0][:-1], '']
            
            objs = [obj.replace('_', ' ') for obj in objs]

            preds.append(pred)
            objss += objs
         
        return preds, objss

    def change_input_dim(self, input_dim):
        if self.hyp_layers is not None and len(self.hyp_layers) > 0:
            self.hyp_layers[0] = nn.Linear(in_features=input_dim, out_features=self.hyp_layers[0].out_features)
        else:
            self.fc_layers[0] = nn.Linear(in_features=input_dim, out_features=self.fc_layers[0].out_features)


    def forward(self, image, state, save_masks=False):
        x = self.get_conditioned_embedding(image, state, save_masks=save_masks)
        
        if self.is_hyperbolic:
            x = self.get_hyperbolic_embedding(x) 
            x = self.manifold.logmap0(x)

        for layer in self.fc_layers[:-1]:
            x = layer(x)
            x = nn.ReLU()(x)

        output = self.fc_layers[-1](x) 
        return output

    def get_image_embedding(self, image):
        outputs = self.vision_encoder(image)
        image_embedding = outputs.last_hidden_state[:, 0, :]  # CLS token embedding
        return image_embedding

    def get_query_embedding(self, queries, mode='sentence'):
        # Convert states to sentences
        if mode == 'sentence':
            sentences = [self.QUERY_TO_SENTENCE[query] for query in queries]
        else:
            sentences = queries

        # Tokenize the sentence
        tokens = self.tokenizer(sentences, return_tensors='pt', padding=True)
        tokens = {k: v.to('cuda') for k, v in tokens.items()} if torch.cuda.is_available() else tokens
        
        # Pass through BERT model
        outputs = self.text_encoder(**tokens)

        hidden_states = outputs.hidden_states[-2] # Second to last hidden layer
        query_embedding = hidden_states.mean(dim=1) # Average the hidden states of each token
        return query_embedding

    def get_conditioned_embedding(self, image, state, save_masks=False):
        image_embedding = self.get_image_embedding(image)
        query_embedding = self.get_query_embedding(state)
        
        # Concatenate scene and state embeddings
        conditioned_embedding = torch.cat((image_embedding, query_embedding), dim=1)
        return conditioned_embedding

    def get_hyperbolic_embedding(self, x):
        x = self.manifold.expmap0(x)
        for layer in self.hyp_layers:
            x = layer(x)
        return x

    def get_representation(self, image, state):
        x = self.get_conditioned_embedding(image, state)
        
        if self.is_hyperbolic:
            x = self.get_hyperbolic_embedding(x) 
        
        return x

    def get_sine_cosine_positional_embeddings(self, num_patches, hidden_size):
        position_embeddings = torch.zeros(num_patches + 1, hidden_size)
        position = torch.arange(0, num_patches + 1, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_size, 2).float() * (-torch.log(torch.tensor(10000.0)) / hidden_size))
        position_embeddings[:, 0::2] = torch.sin(position * div_term)
        position_embeddings[:, 1::2] = torch.cos(position * div_term)
        return position_embeddings

    def load_similarities(self, filepath=None):
        """
        compute distances offline for all queries
        store it so that the distance can be loaded online
        """  
        
        if filepath is not None:
            print(f'loading similarities from {filepath}') 
            self.similarities = json.load(open(filepath, 'r')) 
        else:
            print('no similarity file', filepath)
            
    def get_distances(self, triplets): 
        # determine similarity w/ llm
        distances = []
        for anchor, query1, query2 in triplets:
            pair = '//'.join(sorted([query1, query2]))

            if query1 == query2:
                distance = 0
            else:
                positive = pair.split('//')[0] if self.similarities[anchor][pair] == 1 else pair.split('//')[1]
                distance = 1 if positive == query1 else -1 

            distances.append(distance)
        distances = torch.tensor(distances)
  
        distances = distances.unsqueeze(1)
        return distances

    def get_triplet_embeddings(self, triplets, triplet_imgs):
        # compute distance of anchor-query pairs
        triplet_imgs = torch.hstack([img.unsqueeze(1) for img in triplet_imgs])
        a, b, c, d, e = triplet_imgs.shape
        images = triplet_imgs.reshape(a*b, c, d, e)

        # determine anchor, positive, negative. each is size batch_size x embedding_dim 
        queries = [query for triplet in triplets for query in triplet]  
        
        conditioned_embedding = self.get_conditioned_embedding(images, queries)
        a, b = conditioned_embedding.shape
        conditioned_embedding = conditioned_embedding.reshape(a // 3, 3, b) # batch_size x 3 x embed_dim

        if self.is_hyperbolic:
            conditioned_embedding = self.get_hyperbolic_embedding(conditioned_embedding)
            if self.hyp_to_ball is not None:
                conditioned_embedding = self.hyp_to_ball(conditioned_embedding)

        return conditioned_embedding

    def compute_triplet_loss(self, triplets, triplet_embeddings):  
        distances = self.get_distances(triplets)
        distances = distances.to('cuda') if torch.cuda.is_available() else distances.to('cpu')

        anchor = triplet_embeddings[:, 0, :]
        positive = torch.where(distances >= 0, triplet_embeddings[:, 1, :], triplet_embeddings[:, 2, :])
        negative = torch.where(distances < 0, triplet_embeddings[:, 1, :], triplet_embeddings[:, 2, :])

        loss = self.ssl_loss_fn(anchor, positive, negative)
        return loss

    def training_step(self, batch, batch_idx):
        images, queries, labels, triplets, triplet_imgs, triplet_hierarchies = batch

        outputs = self.forward(images, queries, save_masks=False)
        loss, loss_dict = self.compute_loss(outputs, labels, triplets, triplet_imgs, triplet_hierarchies)

        accuracy = self.compute_accuracy(outputs, labels)

        self.log_dict({f'{loss_name}/train': loss_val.item() for loss_name, loss_val in loss_dict.items()}) 
        self.log("accuracy/train", accuracy)

        return loss
    
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        images, queries, labels, triplets, triplet_imgs, triplet_hierarchies = batch 
        
        if dataloader_idx == 0:
            save_mask = batch_idx % 500 == 0
            save_mask = False
            outputs = self.forward(images, queries, save_masks=save_mask)
            if len(triplets) > 0:
                loss, loss_dict = self.compute_loss(outputs, labels, triplets, triplet_imgs, triplet_hierarchies)
            else:
                loss = 0
                loss_dict = {}
            accuracy = self.compute_accuracy(outputs, labels) 

            self.log_dict({f'{loss_name}/{self.log_mode}': loss_val.item() for loss_name, loss_val in loss_dict.items()}, add_dataloader_idx=False)
            self.log(f"accuracy/{self.log_mode}", accuracy, add_dataloader_idx=False) 
        elif dataloader_idx == 1:
            if batch_idx == 0:
                self = self.to('cpu')

                temp_model = copy.deepcopy(self)
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                temp_model = temp_model.to(device)
                temp_model.log_mode = 'few_shot'

                few_dataset = AbstractionsDataset(self.data_dir, self.dataset, 'train', state_type='ood', num_data=2, similarity=self.args.similarity, reg_type=self.args.reg_type) 
                few_dataloader = DataLoader(few_dataset, batch_size=4, shuffle=True) 

                ood_dataset = AbstractionsDataset(self.data_dir, self.dataset, 'test', state_type='ood', similarity=self.args.similarity, reg_type=self.args.reg_type)
                ood_dataloader = DataLoader(ood_dataset, batch_size=4, shuffle=False)

                print('\n\n ----------------- run few shot validation ----------------- \n\n')
                # Initialize Trainer
                trainer = Trainer(
                    accelerator="auto",
                    strategy="auto",
                    callbacks=[LearningRateMonitor(logging_interval='step')],
                    enable_checkpointing=False,
                    max_epochs=20, 
                    check_val_every_n_epoch=20,
                    enable_progress_bar=False
                )

                trainer.fit(temp_model, few_dataloader, val_dataloaders=ood_dataloader)
                loss = trainer.callback_metrics['loss/few_shot']
                accuracy = trainer.callback_metrics['accuracy/few_shot']
                
                log_dict = {metric_name: metric_val.item() for metric_name, metric_val in trainer.callback_metrics.items() if 'few_shot' in metric_name}
                self.log_dict(log_dict, add_dataloader_idx=False) 
                
                temp_model = temp_model.to('cpu')
                self = self.to(device)  
            else:
                loss = 0
                accuracy = 0
            
        return loss, accuracy

    def test_step(self, batch, batch_idx):
        images, queries, labels, _, _, _ = batch

        outputs = self.forward(images, queries)
        overall_accuracy = self.compute_accuracy(outputs, labels) 
        accuracies = self.compute_accuracy_by_state(outputs, labels, queries)
        accuracies['overall'] = overall_accuracy

        self.log_dict(accuracies, on_step=False, on_epoch=True, prog_bar=True) 
        return accuracies
    

    def compute_norms(self, hyp_embeds, norm_type='euc'):
        if norm_type == 'euc':
            norms = hyp_embeds.norm(dim=-1)
        elif norm_type == 'hyp':
            norms = self.manifold.norm(torch.tensor(0), hyp_embeds, dim=-1)
        else:
            raise ValueError(f"Norm type {norm_type} not implemented")
        return norms


    def compute_regularization_loss(self, triplets, hyp_embeds, triplet_hierarchies):
        norms = self.compute_norms(hyp_embeds)

        if self.reg_type == 'relative':
            reordered_norms = torch.gather(norms, 1, triplet_hierarchies) # order from what should be lowest to highest norm. 0, 1, 2
            shifted_norms = torch.roll(reordered_norms, shifts=-1, dims=1) # shift left. order of 1, 2, 0 norm

            diffs = reordered_norms[:, :2] - shifted_norms[:, :2] # (0-1, 1-2, 2-0)
            diffs += torch.ones_like(diffs) * self.reg_margin # add margin 
            scores = torch.max(torch.zeros_like(diffs), diffs)
        elif self.reg_type == 'absolute':
            diffs = norms - triplet_hierarchies
            scores = torch.max(torch.zeros_like(diffs), diffs) # only keep positive diffs, meaning that the norm of something lower in the hierarchy is smaller

        scores = torch.sum(scores, dim=1) 
        loss = scores.mean()
        return loss
        
    def compute_loss(self, outputs, labels, triplets, triplet_imgs, triplet_hierarchies):#, split='train'):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        losses = {}

        if self.sup_coeff > 0:
            sup_loss = self.sup_loss_fn(outputs, labels)
            losses['sup_loss'] = sup_loss
        else:
            sup_loss = torch.tensor(0).to(device)

        if self.ssl_coeff == 0 and self.reg_coeff == 0:
            loss = self.sup_coeff * sup_loss
        else:
            triplets = [list(row) for row in zip(*triplets)]  
            triplet_embeddings = self.get_triplet_embeddings(triplets, triplet_imgs)

            if self.ssl_coeff > 0:
                ssl_loss = self.compute_triplet_loss(triplets, triplet_embeddings)
                losses['ssl_loss'] = ssl_loss
            else:
                ssl_loss = torch.tensor(0).to(device)

            if self.reg_coeff > 0:
                reg_loss = self.compute_regularization_loss(triplets, triplet_embeddings, triplet_hierarchies)
                losses['reg_loss'] = reg_loss
            else:
                reg_loss = torch.tensor(0).to(device)
            loss = self.sup_coeff * sup_loss + self.ssl_coeff * ssl_loss + self.reg_coeff * reg_loss

        losses['loss'] = loss

        return loss, losses

    def compute_accuracy(self, outputs, labels):
        _, predicted = torch.max(outputs, 1)
        total = labels.size(0)
        correct = (predicted == labels).sum().item()
        return correct / total

    def compute_accuracy_by_state(self, outputs, labels, states):
        state_accuracies = {}
        _, predicted = torch.max(outputs, 1)
        correct = (predicted == labels)

        for state, correct in zip(states, correct):
            state = self.SENTENCE_TO_QUERY[state]
            state_accuracies[state] = state_accuracies.get(state, []) + [correct.item()]
        for state, correct in state_accuracies.items():
            state_accuracies[state] = sum(correct) / len(correct)
        
        return state_accuracies
    
    def configure_optimizers(self, total_epoch=100, lr=0.0001):
        lr = self.args.lr if self.args is not None else lr
        total_epoch = self.args.total_epoch if self.args is not None else total_epoch

        optimizer = optim.AdamW(self.parameters(), lr=lr)

        after_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max = int(total_epoch * 1)
        )

        scheduler = GradualWarmupScheduler(
            optimizer, 
            multiplier=1, 
            total_epoch=max(total_epoch // 5, 1),
            after_scheduler=after_scheduler
        ) 

        return {
            "optimizer": optimizer, 
            "lr_scheduler": {
                "scheduler": scheduler
            }
        } 


class MobiusLinear(nn.Module):
    """
    Hyperbolic linear layer.
    https://github.com/HazyResearch/hgcn/blob/master/layers/hyp_layers.py
    """

    def __init__(self, in_features, out_features, manifold, c=1., dropout=None, use_bias=True):
        super(MobiusLinear, self).__init__()
        self.manifold = manifold
        self.in_features = in_features
        self.out_features = out_features
        self.c = c
        self.dropout = dropout
        self.use_bias = use_bias
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform_(self.weight, gain=math.sqrt(2))
        init.constant_(self.bias, 0)

    def forward(self, x):
        if self.dropout is not None:
            drop_weight = F.dropout(self.weight, self.dropout, training=self.training)

        mv = self.manifold.mobius_matvec(self.weight, x)
        output = self.manifold.projx(mv)
        if self.use_bias:
            hyp_bias = self.manifold.expmap0(self.bias)
            hyp_bias = self.manifold.projx(hyp_bias)
            output = self.manifold.mobius_add(output, hyp_bias)
            output = self.manifold.projx(output)
        return output

    def extra_repr(self):
        return 'in_features={}, out_features={}, c={}'.format(
            self.in_features, self.out_features, self.c
        )
