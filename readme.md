# Install script

```
python3 setup.py bdist_wheel; pip3 install --force-reinstall --no-deps dist/*.whl
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