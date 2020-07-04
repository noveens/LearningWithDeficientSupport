# Off-Policy Learning with Deficient Support

This repository contains various off-policy learning algorithms under deficient support. The code accompanies the paper ***"Off-policy Bandits with Deficient Support"*** [[ACM]](https://doi.org/10.1145/3394486.3403139) [[arXiv]](https://arxiv.org/pdf/2006.09438.pdf) where we firstly define the impact of deficient support on existing algorithms and secondly propose different estimators to tackle this problem.

If you find any module of this repository helpful for your own research, please consider citing the below KDD'20 paper. Thanks!
```
@inproceedings{SachdevaJoachims20,
  author = {Noveen Sachdeva, Yi Su, and Thorsten Joachims},
  title = {Off-policy Bandits with Deficient Support},
  booktitle = {ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD)},
  year = {2020}
}
```

**Code Author**: Noveen Sachdeva (ernoveen@gmail.com)

---
### Environents
- Python3 
- Pytorch >= 0.4.0
- tensorboardX

---
### Data Setup
There are six hyper-parameters for pre-processing the data which you *might* need to edit in the `preprocess.sh` file.

```bash
$ ./preprocess.sh
```

The above command will:
- Download the CIFAR-10 dataset
- Train a logging policy (Default: Train ResNet20 on 35K out of 50K images for 2 epochs)
- Create bandit feedback data with train/test/val splits (Default: temperature (t=4) softmax on logging policy, and clip actions with prob < 0.01)
- Train a regression function (Default: ResNet20 until convergence on the bandit feedback dataset using all features)
- Create an auxillary data file (Only used while running the efficient approximation of the `conservative_extrapolation` method)

---
### Run Instructions
- Edit the `hyper_params.py` file which lists all config parameters, including what type of off-policy learning algorithm to run. Currently supported models:
  - IPS
  - IPS w/ prune_unsupported = True
  - IPS w/ imputed = negative
  - RegressionExtrapolation
  - PolicyRestriction

- Finally, type the following command to run:
```bash
$ cd code;
$ CUDA_VISIBLE_DEVICES=<SOME_GPU_ID> python main.py
```
---
### Contribution
If you have a new proposition about off-policy learning in the support deficient scenario, please feel free to send a pull request with your algorithm and I'll be happy to merge it into this repository.


### License
----

MIT
