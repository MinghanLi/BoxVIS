# Prepare Datasets for BoxVIS

The required datasets should be
```
$datasets/
  coco/
  ytvis_2019/
  ytvis_2021/
  ovis/
  bvisd/
```
`./datasets` relative to your current working directory.

## STEP-1: Prepare Image & VIS datasets
### Expected dataset structure for [COCO](https://cocodataset.org/#download):

```
coco/
  annotations/
    instances_{train,val}2017.json
    coco2bvisd_train.json (converted)
    {train,val}2017/
    # image files that are mentioned in the corresponding json
```

### Expected dataset structure for [YouTube-VIS 2021](https://competitions.codalab.org/competitions/28988) and [2022](https://codalab.lisn.upsaclay.fr/competitions/3410):
For evaluating on valid set of YouTubeVIS 2022, you just need to replace 'valid.json' of YouTubeVIS 2021 with 'valid.json' of YouTubeVIS 2022.

```
ytvis_2021/
  {train,valid,test}.json
  {train,valid,test}/
    JPEGImages/
```

### Expected dataset structure for [OVIS](https://competitions.codalab.org/competitions/32377):

```
ovis/
  {train,valid,test}.json
  {train,valid,test}/
    JPEGImages/
```

### Split train set for YouTube-VIS and OVIS
For convenient model evaluation, we split the training annotations train.json into two sets: 
train_sub.json and valid_sub.json. And train_sub.json is used for training, while valid_sub.json is used for validation.
```bash
python boxvis/data/data_utils/split_train_set.py
```

## STEP-2: Prepare Box-supervised VIS Dataset (BVISD) 
### Convert COCO to BVISD dataset
There are 25 overlapping categories between COCO and BVISD, including around 90k images and 450k objects. 
Please run the following command to extract these images:
```bash
python boxvis/data/data_utils/convert_coco2bvisd.py
```

Alternatively, you can download the converted annotation files from [here](https://drive.google.com/drive/folders/1nQWlCc7PXptRWqgFrovPCrKTF7jmozMn?usp=sharing).

During training, the static images from COCO will be augmented to pseudo video clips by the module
`build_pseudo_augmentation`, please refer to `boxvis/data/augmentation.py` for more details.

### Joint training on BVISD
We utilize `CombinedDataLoader` to simultaneously load video clips from YTVIS21, OVIS and COCO datasets,
where the sampling weights are 1/2, 1/4, and 1/4, respectively. Please refer to `boxvis/data/combined_loader.py` for more details.

For your easier understanding, we point out some core code of joint training on BVISD.
* The mapping ids of object categories between source and target datasets are provided in `boxvis/data/datasets/bvisd.py`.
* When loading video clips, the category ids of YTVIS21, OVIS and COCO are mapped to BVISD's category ids.
  Please refer to Lines 176-191 and 369-381 in `boxvis/data/dataset_mapper.py`.

* You can modify the sampling weights of BVISD datasets from the configure file:
    ```
    DATASETS:
      DATASET_RATIO: [2., 1., 1.]
      TRAIN: ("ytvis_2021_train", "ovis_train", "coco2bvisd_train")
      TEST: ("bvisd_dev", )
    ```

### Convert BVISD valid_sub set
Valid_sub set of BVISD consists of all videos in the valid_sub sets of YTVIS21 and OVIS, and 
category ids of YTVIS21 and OVIS are mapped to BVISD's category ids. Please run:
```bash
python boxvis/data/data_utils/convert_bvisd_valid_set.py
```
Alternatively, you can download the converted annotation files from [here](https://drive.google.com/drive/folders/1nQWlCc7PXptRWqgFrovPCrKTF7jmozMn?usp=sharing).

Compared with isolated YTVIS21 or OVIS valid_sub set, BVISD's valid_sub set covers more diverse variants of videos,
thereby can be used to evaluate the generalization ability of VIS models.




## Expected final dataset structure for all:
```
$datasets/
    +-- coco
    |   |
    |   +-- annotations
    |   |   |
    |   |   +-- instances_{train,val}2017.json
    |   |   +-- coco2bjvis_train.json
    |   |
    |   +-- {train,val}2017
    |
    +-- ytvis_2021
    |   | 
    |   +-- train.json
    |   +-- train_sub.json
    |   +-- valid.json
    |   +-- valid_sub.json
    |   +-- test.json
    |
    +-- ovis
    |   | 
    |   +-- train.json
    |   +-- train_sub.json
    |   +-- valid.json
    |   +-- valid_sub.json
    |   +-- test.json
    |
    +-- bvisd
    |   | 
    |   +-- valid_sub.json
    ```
