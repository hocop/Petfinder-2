# Install script

```
./setup.sh
```

# Run training

```
python scripts/train.py --config configs/pf2.yaml --gpus 1 --wandb_project pf2 --name test
```

# TODO
* stop gradient where there is no animal
* train separately on different image aspect ratios
* mixup / manifold mixup
* double model with softmax
* dropout before linear
* classify dog breeds
* SVR boost
* freezeout
* pretrain on pf1
* global weight decay