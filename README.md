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
pip install -r requirements.txt
```

2. Download dataset:

If needed, submit a agreement to download the dataset used in the paper. 

3. If needed, create pretrained_models folder and download model weights [here]().

<br>

## Evaluation





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

The main structure of the code is based on [mmaction2](https://github.com/open-mmlab/mmaction2). Thanks for sharing good practices!
