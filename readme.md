# Install script

```
./setup.sh
```

# Run training

```
python scripts/train.py --config configs/pf2.yaml --gpus 1 --wandb_project pf2 --name test
```

# TODO
* https://www.kaggle.com/c/petfinder-pawpularity-score/discussion/288896 
* SVR boost
* many heads. avg for prediction, min for mixup
* resize with crop
* smart blending of TTA predictions