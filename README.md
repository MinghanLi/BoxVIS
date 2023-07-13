# [BoxVIS](https://arxiv.org/abs/2303.14618): Video Instance Segmentation with Box Annotation

[Minghan LI](https://scholar.google.com/citations?user=LhdBgMAAAAAJ) and [Lei ZHANG](https://www4.comp.polyu.edu.hk/~cslzhang/)

[[arXiv]](https://arxiv.org/abs/2303.14618)

<div align="center">
  <img src="imgs/BoxVIS_overview.jpg" width="80%" height="100%"/>
</div><br/>

## Updates
* **`July 13, 2023`:** Paper has been updated. 
* **`June 30, 2023`:** Code and trained models are available now.
* **`March 28, 2023`:** Paper is available now.


## Installation
See [installation instructions](INSTALL.md).

## Datasets
See [Datasets preparation](./datasets/README.md).

## Getting Started
We provide a script `train_net_boxvis.py`, that is made to train all the configs provided in BoxVIS.

Training: download [pretrained weights of Mask2Former](https://github.com/facebookresearch/Mask2Former/blob/main/MODEL_ZOO.md) and save it into the path 'pretrained/*.pth', then run:
```
sh run.sh
```

Inference: download [trained weights](https://drive.google.com/drive/folders/1xy2whEG-Dw40GOunhjqz2SzGviAug7qd?usp=sharing), and save it into the path 'pretrained/*.pth', then run:
```
sh test.sh
```

## Quantitative performance comparison 
<div align="center">
  <img src="imgs/sota_yt21_coco.jpg" width="80%" height="100%"/>
</div><br/>

## <a name="CitingBoxVIS"></a>Citing BoxVIS

If you use BoxVIS in your research or wish to refer to the baseline results published in the Model Zoo, please use the following BibTeX entry.

```BibTeX
@misc{li2023boxvis,
      title={BoxVIS: Video Instance Segmentation with Box Annotations}, 
      author={Minghan Li and Lei Zhang},
      year={2023},
      eprint={2303.14618},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Acknowledgement

Our code is largely based on [Detectron2](https://github.com/facebookresearch/detectron2), [Mask2Former](https://github.com/facebookresearch/Mask2Former), [MinVIS](https://github.com/NVlabs/MinVIS), and [VITA](https://github.com/sukjunhwang/VITA). We are truly grateful for their excellent work.
