This is an implementation of an iterative SE(3)-Transformer where gradients are backpropagated through the spherical harmonics.

The toy dataset is created on the fly.

This is an example command to run an the experiment:
``` python opt_run.py --name 0226f_sa0211h_st10a_iterated_10pt_ --model SE3TransformerIterative --num_iter 3```


Find the blog post describing these experiments [here](https://edwag.github.io/se3iterative/) and the paper [here](https://arxiv.org/pdf/2102.13419.pdf).

If you found this helpful for your research, please cite us as:
@inproceedings{se3iterative,
    title={Iterative SE(3)-Transformers},
    author={Fabian B. Fuchs and Ed Wagstaff and Justas Dauparas and Ingmar Posner},
    year={2021},
    booktitle = {arXiv},
}

