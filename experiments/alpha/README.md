# QM9

Experiments on the QM9 dataset. Download a preprocessed version of the dataset [here](https://drive.google.com/file/d/1EpJG0Bo2RPK30bMKK6IUdsR5r0pTBEP0/view?usp=sharing) and place it in `experiments/qm9/`

## Training

To train the model in the paper, run this command:

```train
python train.py --model SE3Transformer --num_epochs 100 --num_degrees 4 --num_layers 7 --num_channels 32 --name qm9-homo --num_workers 4 --batch_size 32 --task homo --div 2 --pooling max --head 8
```

## Evaluation

Untested

```eval
python eval.py --model SE3Transformer --num_degrees 4 --num_layers 7 --num_channels 32 --name qm9-homo --num_workers 4 --batch_size 32 --task homo --div 2 --pooling max --head 8 --restore <path-to-model>
```

## Pre-trained Models

- [ ] TODO


## Results

Our model achieves the following performance on (latest results, may not be in paper):

| Dataset         | Mean        | Standard deviation |
| --------------- | ------------| ------------------ |
| Alpha (bohr^3)) |     .142    |      .002          |
| Gap (meV)       |     53.0    |      .3            |
| Lumo (meV)      |     33.0    |      .7            |
| Homo (meV)      |     35.0    |      .9            |
| Mu (D)          |     .051    |      .001          |
| Cv (cal/mol K)  |     .054    |      .002          |

