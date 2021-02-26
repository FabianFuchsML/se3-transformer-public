
These scripts are adapted from https://github.com/ethanfetaya/NRI which is under the MIT license.

When using these scripts, please cite

@article{kipf2018neural,
  title={Neural Relational Inference for Interacting Systems},
  author={Kipf, Thomas and Fetaya, Ethan and Wang, Kuan-Chieh and Welling, Max and Zemel, Richard},
  journal={arXiv preprint arXiv:1802.04687},
  year={2018}
}

The following is a suggested configuration for creating a dataset
of 20 bodies with random charges

```
python3 generate_dataset.py --num-train 30000 --num-test 5000 --n-balls 20
```
