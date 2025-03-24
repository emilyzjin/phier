import json
import torch
import numpy as np
from PIL import Image
from scipy.stats import poisson
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from itertools import combinations, permutations
import os

def create_dataloaders(data_dir, dataset, split='test', num_data=10, batch_size=16, similarity='llm_predicate', reg_type=None, test_only=False):
    print('creating dataloaders')
    in_dataset = AbstractionsDataset(data_dir, dataset, split, state_type='train', num_data=num_data, similarity=similarity, reg_type=reg_type, test_only=test_only)
    in_dataloader = DataLoader(in_dataset, batch_size=batch_size, shuffle=False) 

    ood_dataset = AbstractionsDataset(data_dir, dataset, split, state_type='ood', num_data=num_data, similarity=similarity, reg_type=reg_type, test_only=test_only)
    ood_dataloader = DataLoader(ood_dataset, batch_size=batch_size, shuffle=False)

    return in_dataloader, ood_dataloader


def get_possible_triplets(queries):
    similarities = {anchor: {} for anchor in queries}

    for anchor in queries:
        other_queries = [query for query in queries if query != anchor]

        for query1, query2 in tqdm(combinations(other_queries, 2), total=len(other_queries) * (len(other_queries)) // 2):
            similarities[anchor]['//'.join(sorted([query1, query2]))] = []
    os.makedirs(f'end_to_end/similarities/', exist_ok=True)
    json.dump(similarities, open(f'end_to_end/similarities/all.json', 'w'), indent=4)    
    triplets = [triplet for triplet in combinations(queries, 3)]
    return triplets


def get_pred(state):
    return state.split('(')[0]


def get_objs(state):
    if ',' in state:
        obj1 = state.split('(')[1].split(',')[0]
        obj2 = state.split(', ')[1].split(')')[0]
        objs = [obj1, obj2]
    else:
        objs = [state.split('(')[1].split(')')[0]]
    
    return objs


class AbstractionsDataset(Dataset):
    def __init__(
            self, 
            data_dir, 
            dataset, 
            split, 
            state_type='train', 
            state=None, 
            num_data=None, 
            triplet_selector='random', 
            test_only=False, 
            to_tensor=True, 
            similarity='llm',
            reg_type=None
        ):
        TRAIN_STATES = json.load(open(os.path.join(data_dir, dataset, 'states.json'), 'r'))['train']
        OOD_STATES = json.load(open(os.path.join(data_dir, dataset, 'states.json'), 'r'))['ood']

        self.dataset = dataset
        self.data_dir = os.path.join(data_dir, dataset)
        self.split = split 
        self.test_only = test_only

        if state_type == 'train':
            self.all_states = TRAIN_STATES
        elif state_type == 'ood':
            self.all_states = OOD_STATES
        elif state_type == 'all':
            self.all_states = TRAIN_STATES + OOD_STATES

        # if to_tensor:
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize the image to 224x224 pixels
            transforms.ToTensor()  # Convert the image to a tensor
        ]) 

        # load in the filename, labels
        with open(os.path.join(self.data_dir, f'all_samples.json'), 'r') as f:
            self.data = json.load(f)[split]

        if state is not None:
            self.data = {k: v for k, v in self.data.items() if k.split('=')[0] == state}
        else:
            self.data = {k: v for k, v in self.data.items() if k.split('=')[0] in self.all_states}

        self.states = []
        self.labels = []
        self.filepaths = []

        for state, files in self.data.items():
            cur_num = len(files) if (num_data is None or num_data > len(files)) else num_data

            states = [self.get_state(state) for _ in range(cur_num)]
            labels = [self.get_label(state) for _ in range(cur_num)]
            filepaths = [os.path.join(self.data_dir, state.split('=')[0], file) for file in files[:num_data]]

            assert len(states) == len(filepaths), f"num data {cur_num}, state {state}, len states {len(states)}, len filepaths {len(filepaths)}"
            self.states += states
            self.labels += labels 
            self.filepaths += filepaths

        self.reg_type = reg_type
        self.hierarchy = None

        if not self.test_only:
            if self.reg_type in ['absolute', 'relative']:
                self.hierarchy = self.load_hierarchy(filepath=f'src/assets/hierarchies/{self.dataset}_{self.reg_type}.json')

            self.get_triplets(triplet_selector, similarity=similarity)


    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        image = Image.open(self.filepaths[idx]).convert("RGB")
        state = self.states[idx]
        label = self.labels[idx] # convert to int for cross entropy loss
        if self.transform:
            image = self.transform(image)

        if self.test_only:
            return image, state, label, [], [], []
        else:
            triplet = self.triplets[idx]
            triplet_imgs = []
            for i in triplet: 
                img = Image.open(self.filepaths[i]).convert("RGB")
                img = self.transform(img)
                triplet_imgs.append(img)
            
            triplet = (self.states[triplet[0]], self.states[triplet[1]], self.states[triplet[2]])

            pair = '//'.join(sorted(triplet[1:]))

            triplet_hierarchy = torch.tensor(self.hierarchy[triplet[0]][pair]) if self.hierarchy is not None else 0
     
            return image, state, label, triplet, triplet_imgs, triplet_hierarchy

    def load_hierarchy(self, filepath=None):
        """
        compute hierarchy offline for all queries
        store it so that the distance can be loaded online
        """  
        if filepath is not None:
            print(f'loading hierarchy from {filepath}')
            hierarchy = json.load(open(filepath, 'r')) 
        else:
            print('no hierarchy file', filepath)

        return hierarchy 

    def get_state(self, state_value):
        return state_value.split('=')[0]
    
    def get_label(self, state_value):
        return int((eval(state_value.split('=')[1])))

    def get_triplets(self, triplet_selector, similarity='llm'):  
        num_triplets = len(self.states)

        if triplet_selector == 'random':
            if similarity == 'llm':
                vocab = {self.all_states[i]: i for i in range(len(self.all_states))}
                queries = [vocab[query] for query in self.states]
            elif similarity == 'llm_predicate':
                all_preds = list(set([get_pred(state) for state in self.all_states]))
                vocab = {all_preds[i]: i for i in range(len(all_preds))}
                queries = [vocab[get_pred(query)] for query in self.states]
            else:
                raise ValueError('similarity must be either state or predicate')
            
            queries = torch.Tensor(queries).unsqueeze(1)
            mask = torch.ne(queries, queries.squeeze().unsqueeze(0)).int()
            samples = [] 

            for i in range(num_triplets):
                idxs = torch.nonzero(mask[i]).flatten()

                while True:
                    sample = idxs[torch.randperm(len(idxs))[:2]]
                    if int(queries[sample[0]]) != int(queries[sample[1]]):
                        if similarity == 'llm':
                            break
                        elif similarity == 'llm_predicate':                       
                            if self.reg_type == 'relative':
                                anchor = self.states[i]
                                sample0 = self.states[sample[0]]
                                sample1 = self.states[sample[1]]
                                pair = sorted([sample0, sample1])
                                triplet_norms = self.hierarchy[anchor]['//'.join(pair)]

                                if max(triplet_norms) == 2:
                                    break
                            else:
                                break

                query_pair = sorted([self.states[sample[0]], self.states[sample[1]]])
                if query_pair[0] == self.states[sample[0]]:
                    sample = sample
                else:
                    sample = sample.flip(0)

                samples.append(sample)

            anchors = torch.arange(0, num_triplets).unsqueeze(1) # num x 1
            samples = torch.vstack(samples) # num x 2
            triplets = torch.cat((anchors, samples), dim=1) # num x 3   

        elif triplet_selector == 'llm':
            queries = self.states
            with open(os.path.join('end_to_end/anchor_to_triplets.json'), 'r') as f:
                anchor_to_triplets = json.load(f)
            
            triplets = []
            for anchor_idx in range(len(queries)):
                temp = anchor_to_triplets[queries[anchor_idx]]
                p = np.array([1/(5 * i + 15) for i in range(len(temp))])
                p = np.array([round(poisson.pmf(x, len(temp) / 2, - len(temp) // 4), 2) for x in range(len(temp))])
                p = p / p.sum()
                sample_idx = np.random.choice([i for i in range(len(temp))], p=p)
                anchor, pos, neg = temp[sample_idx]

                pos_idxs = [j for j in range(num_triplets) if queries[j] == pos]
                neg_idxs = [j for j in range(num_triplets) if queries[j] == neg]
                pos_idx = np.random.choice(pos_idxs)
                neg_idx = np.random.choice(neg_idxs)

                triplet = [anchor_idx, pos_idx, neg_idx]
                triplets.append(triplet)
                
            triplets = torch.tensor(triplets)
            
        self.triplets = triplets
