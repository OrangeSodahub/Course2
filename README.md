# Couse2-Alexnet
《机器学习》大作业2

## Data
* Download the Animals-10 dataset from https://jbox.sjtu.edu.cn/1/X1L5Rp, password: hwqw.
* Rename the image files to `type_num.jpg`.
* Resize the images to (277,277).
* Merge all the folders to one in order to load dataset conveniently.
* Split one to `train` dataset and `test` dataset.
```python
python tools/process.py
```

## Train
The trained model will be saved to `/pkl`.
```python
python tools/train.py
```

## Predict
```python
python tools/pred.py
```

## Pretrained Model