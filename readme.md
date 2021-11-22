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
* stop gradient where there is no animal
* mixup / manifold mixup
* double model with softmax
* SVR boost
* pretrain on pf1
* lr scheduler
* SGD with momentum
* `aug_anneling = max_epochs / 2`
* top3 average
* freeze first layers for several steps
