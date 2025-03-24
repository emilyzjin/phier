# Predicate Hierarchies Improve Few-Shot State Classification
![Pull Figure](imgs/pull.png)

Predicate Hierarchies Improve Few-Shot State Classification
[Emily Jin*](https://emilyzjin.github.io/), [Joy Hsu*](https://stanford.edu/~joycj/), [Jiajun Wu](https://jiajunwu.com/)

In International Conference on Learning Representations (ICLR) 2025

[[project page](https://emilyzjin.github.io/projects/phier.html)] [[arXiv](https://www.arxiv.org/abs/2502.12481)] [[paper](https://www.arxiv.org/pdf/2502.12481)]

## Setup
Run the following commands to install necessary dependencies.

```bash
  conda create -n phier 
  conda activate phierg
  pip install -r requirements.txt
```

## Training
```bash
    python train.py --dataset ${dataset} 
```

## Evaluation
```bash
    python -u val.py --checkpoint_name ${checkpoint_name} --dataset ${dataset}  
```

Please feel free to email me at emilyjin@stanford.edu if any problems arise.
