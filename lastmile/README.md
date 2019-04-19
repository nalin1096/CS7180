# lastmile

This folder, `lastmile`, contains all the code necessary to generate our 
final report. We used this as a development repository. New ideas were tried
using an incremental file number scheme. Note that we expect the data in a 
specific directory format when training on simulated data; [RAISE Raw Images Dataset](http://loki.disi.unitn.it/RAISE/).

`raise/rgb/train`, `raise/rgb/test`, `raise/rgb/val`. 

## Training on Simulated Data

```
python train_model027.py
```

## Zero-Short Learning, Transfer Learning

You will need to comment out various review functions
depending on how you want to run this.

```
python freeze_model.py
```

## Creating images for Giph from Checkpoints

We created a visualization of a predicted image being trained
during every 5 epochs to qualitatively show prediction improvements.

```
python giphy.py
```

## Other notes

Our data pipeline is located in `image_preprocessing.py`. Utilities 
for testing are located in `test_model.py`. Model utilities, 
`model_utils.py` includes important functions for checkpointing.
