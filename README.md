# BIGFormer

source code of BIGFormer. <br>
The data used in this study can be openly available at http://adni.loni.usc.edu/.

## Dependencies
The script has been tested running under Python 3.9.0, with the following packages installed (along with their dependencies): <br>

* numpy==1.21.2 <br>
* networkx==2.6.3 <br>
* torch==1.11.0 <br>
* torch-geometric==2.0.4 <br>
In addition, CUDA 11.3 have been used on NVIDIA GeForce RTX 3080. <br>

## Overview
The repository is organised as follows: <br>
* `dataset.py`: contains the implementation of **F**actors **I**nteraction **G**graph (FIG); <br>
* `transform.py`: include the implementation of batching operation and graph related feature engineering; <br>
* `layers.py`: implements the **p**erception **l**ocal **s**tructural **a**wareness (PLSA), **g**lobal **r**eliance **i**nference **c**omponent (GRIC), and other related layers; <br>
* `models.py`: contains the implementation of the BIGFormer and other comparative models; <br>
* `untils.py`: contains the HGCCN related operation; <br>
* `parameters.py`: including all the parameters involoved in model; <br>
* `train_eval_helper.py`: contains the cross-validation related helper functions. <br>
* Finally, `main.py` puts all of the above together and be used to execute a full training run.
