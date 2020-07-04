## Training

for UCF-101, HEVC

# HEVC Motion vector model.

(no-accu)

baseline_1.1

hevc pretrain.

```bash
python train.py --lr 0.01 --batch-size 80 --arch resnet18 \
 	--data-name ucf101 --representation mv \
 	--model-prefix ucf101_hmv_model \
 	--lr-steps 150 270 390  --epochs 511 \
 	--gpus 0
```

Testing Results: Prec@1 73.302 Prec@5 91.197 Loss 1.16501

baseline_1.2

hevc no-pretrain

```bash
python train.py --lr 0.01 --batch-size 80 --arch resnet18 \
 	--data-name ucf101 --representation mv \
 	--model-prefix ucf101_hmv_nopre_model \
 	--lr-steps 150 270 390  --epochs 511 \
 	--gpus 7 --no-pretrained
```

Testing Results: Prec@1 69.707 Prec@5 90.589 Loss 1.15895

import torch
weight = 'ucf101_iframe_model_iframe_checkpoint.pth.tar'
checkpoint = torch.load(weight)
print("model epoch {} best prec@1: {}".format(checkpoint['epoch'], checkpoint['best_prec1']))
model epoch 476 best prec@1: 85.51414221618153
model epoch 146 best prec@1: 86.17499338342121

baseline_1.3

hevc no-pretrain no-precessing/ 256 /8x8 feature size. / 7x7 to 3 3x3 con1

```bash
python baseline_1_3_train.py --lr 0.01 --batch-size 80 --batch-size-val 28 \
 	--arch resnet18 --data-name ucf101 --representation mv \
 	--model-prefix ucf101_mv_baseline_1_3 \
 	--lr-steps 150 270 390  --epochs 511 \
 	--gpus 5

```

Testing Results: Prec@1 65.794 Prec@5 87.867 Loss 1.31846

```bash
python test.py --gpus 7 \
        --arch resnet18 \
        --data-name ucf101 --representation mv \
        --weights ucf101_hmv_model_mv_model_best.pth.tar \
        --save-scores hmv \
        --test-crops 1
```

```bash
python test.py --gpus 7 \
        --arch resnet18 \
        --data-name ucf101 --representation mv \
        --weights ucf101_hmv_nopre_model_mv_model_best.pth.tar \
        --save-scores hmv_nopre \
        --test-crops 1
```

```bash
python baseline_1_3_test.py --gpus 7 \
        --arch resnet18 \
        --data-name ucf101 --representation mv \
        --weights ucf101_mv_baseline_1_3_mv_model_best.pth.tar \
        --save-scores mv_baseline_1_3 \
        --test-crops 1
```
