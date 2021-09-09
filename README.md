# Introduction 
This is Bidirectional Correct Attention Network, source code of Attend, Correct and Focus: [Bidirectional Correct Attention Network for Image-Text Matching (ICIP 2021)](https://ieeexplore.ieee.org/abstract/document/9506438) and BCAN++: Cross-Modal Retrieval with Improved Bidirectional Correct Attention Network.
It is built on top of the [SCAN](github.com/kuanghuei/SCAN) in Pytorch.
# Requirements and Installation
We recommended the following dependencies.
  - Python 3.7
  - Pytorch 1.6+
  - Numpy
  - nltk
# Download data
Download the dataset files. We use the image feature created by SCAN, downloaded [here](https://github.com/kuanghuei/SCAN). All the data needed for reproducing the experiments in the paper, including image features and vocabularies, can be downloaded from:
```bash
wget https://scanproject.blob.core.windows.net/scan-data/data.zip
wget https://scanproject.blob.core.windows.net/scan-data/vocab.zip 
```
# Training
- Train new BCAN models: Run `train.py`:
```bash
python train.py --data_path "$DATA_PATH" --data_name "$DATA_NAME" --logger_name "$LOGGER_NAME" --model_name "$MODEL_NAME"
```

- Train new BCAN++ models: Run `bcan++_train.py`:
```bash
python bcan++_trian.py --data_path "$DATA_PATH" --data_name "$DATA_NAME" --logger_name "$LOGGER_NAME" --model_name "$MODEL_NAME"
```

Argument used to train Flickr30K models and MSCOCO models are similar with those of SCAN:

For Flickr30K:
| Method | Arguments |
|:-:|:-:|
|BCAN-equal| `--num_epochs=20 --lr_update=15 --correct_type=equal`|
|BCAN-prob| `--num_epochs=20 --lr_update=15 --correct_type=prob`|

For MSCOCO:
| Method | Arguments |
|:-:|:-:|
|BCAN-equal| `--num_epochs=15 --lr_update=8 --correct_type=equal`|
|BCAN-prob| `--num_epochs=15 --lr_update=8 --correct_type=prob`|



# Evaluation
```python
from vocab import Vocabulary
import evaluation
evaluation.evalrank("$RUN_PATH/coco_scan/model_best.pth.tar", data_path="$DATA_PATH", split="test")
```
