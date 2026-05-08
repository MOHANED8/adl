## Hydra multirun sweeps

### Architectures × datasets

```bash
python main.py -m dataset=cifar10,cifar100,tiny_imagenet model=resnet18,resnet34,efficientnet_b0,vit,deit_small training=base
```

### Optimizers

```bash
python main.py -m dataset=cifar10 model=cnn training=adam,base,sgd
```

### Batch size (small vs large)

```bash
python main.py -m dataset=cifar10 dataset.train.batch_size=32,256
```

### Augmentation policy

```bash
python main.py -m dataset=cifar10 dataset.aug=none,imagenet_basic,strong
```

### Label noise

```bash
python main.py -m dataset=cifar10 dataset.label_noise_p=0.0,0.1,0.2,0.4 dataset.label_noise_seed=0
```

### Domain shift (WILDS)

```bash
python main.py -m dataset=wilds_camelyon17,wilds_iwildcam model=resnet18 training=base
```

