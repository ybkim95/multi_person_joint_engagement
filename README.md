# Joint Engagement Classification using Video Augmentation Techniques for Multi-person HRI in the wild

[arXiv preprint arXiv:2212.14128](https://arxiv.org/abs/2212.14128)


<p align="center">
  <img width="700" height="400" src="https://user-images.githubusercontent.com/45308022/221431152-f07c152e-1b4a-4466-8c51-3a1a3e13d713.png">
</p>

Accepted at AAMAS 2023!

Repository contains:

* the code to conduct all experiments reported in the paper
* fine-tuned model weights
* data access link

<br>

## Get Started

1. Create an environment:

```
conda create python=3.9 -y -n multi_person_joint_eng
conda activate multi_person_joint_eng
pip3 install -r requirements.txt
```

2. Download dataset:

If needed, submit a agreement to download the dataset used in the paper. 

3. If needed, create pretrained_models folder and download model weights [here](https://drive.google.com/drive/folders/1ltV9r7PEQE2KOsW9geDQbibN8_4mEAUq?usp=sharing).

<br>

## Datasets

It is recommended to symlink the dataset root to `$ROOT/data`.
If your folder structure is different, you may need to change the corresponding paths in config files.

```
multi_person_joint_engagement
├── mmaction
├── tools
├── configs
├── data
│   ├── triadic
│   │   ├── rawframes_train
│   │   ├── rawframes_val
│   │   ├── triadic_train_list.txt
│   │   ├── triadic_val_list.txt
│   ├── augtriadic
│   │   ├── rawframes_train
│   │   ├── rawframes_val
│   │   ├── augtriadic_train_list.txt
│   │   ├── augtriadic_val_list.txt
│   ├── ...
```

For more information on data preparation, please see [data_preparation.md](data_preparation.md)


<br>


## Train & Evaluate


### Train with multiple GPUs

```shell
./tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM} [optional arguments]
```

Optional arguments are:

- `--validate` (**strongly recommended**): Perform evaluation at every k (default value is 5, which can be modified by changing the `interval` value in `evaluation` dict in each config file) epochs during the training.
- `--test-last`: Test the final checkpoint when training is over, save the prediction to `${WORK_DIR}/last_pred.pkl`.
- `--test-best`: Test the best checkpoint when training is over, save the prediction to `${WORK_DIR}/best_pred.pkl`.
- `--work-dir ${WORK_DIR}`: Override the working directory specified in the config file.
- `--resume-from ${CHECKPOINT_FILE}`: Resume from a previous checkpoint file.
- `--gpus ${GPU_NUM}`: Number of gpus to use, which is only applicable to non-distributed training.
- `--gpu-ids ${GPU_IDS}`: IDs of gpus to use, which is only applicable to non-distributed training.
- `--seed ${SEED}`: Seed id for random state in python, numpy and pytorch to generate random numbers.
- `--deterministic`: If specified, it will set deterministic options for CUDNN backend.
- `JOB_LAUNCHER`: Items for distributed job initialization launcher. Allowed choices are `none`, `pytorch`, `slurm`, `mpi`. Especially, if set to none, it will test in a non-distributed mode.
- `LOCAL_RANK`: ID for local rank. If not specified, it will be set to 0.

Difference between `resume-from` and `load-from`:
`resume-from` loads both the model weights and optimizer status, and the epoch is also inherited from the specified checkpoint. It is usually used for resuming the training process that is interrupted accidentally.
`load-from` only loads the model weights and the training epoch starts from 0. It is usually used for finetuning.

<br>


## Cite

If you use this code in your research, please cite:

```
@misc{https://doi.org/10.48550/arxiv.2212.14128,
  doi = {10.48550/ARXIV.2212.14128},
  url = {https://arxiv.org/abs/2212.14128},
  author = {Kim, Yubin and Chen, Huili and Alghowinem, Sharifa and Breazeal, Cynthia and Park, Hae Won},
  keywords = {Computer Vision and Pattern Recognition (cs.CV), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Joint Engagement Classification using Video Augmentation Techniques for Multi-person Human-robot Interaction},
  publisher = {arXiv},
  year = {2022},
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```

## Contact

If you have any problems with the code or have a question, please open an issue or send an email to ybkim95@media.mit.edu. I'll try to answer as soon as possible.


## Acknowledgments and Licenses

The main structure of the code is based on [mmaction2](https://github.com/open-mmlab/mmaction2), [pyskl](https://github.com/kennymckormick/pyskl) and [SimSwap](https://github.com/neuralchen/SimSwap). Thanks for sharing wonderful works!
