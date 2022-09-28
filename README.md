# Tf2PyTorch
The tools for translate the pretrained TensorFlow model checkpoint to the PyTorch model, especially for the MobileNet v1 (the paper can be found in [here](https://arxiv.org/abs/1704.04861)) in TensorFlow Slim (the Mobilenet V1 code original code can be found in [here](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.py)). More models support will release in the future.
The translated pytorch checkpoint (stored in *.pth file) can be loaded and used in correspoding MobileNet v1 pytorch implementation (I find MobileNet V1 pytorch implementation in [here](https://github.com/osmr/imgclsmob/blob/956b4ebab0bbf98de4e1548287df5197a3c7154e/pytorch/pytorchcv/models/mobilenet.py), you can also find the pretrained mobilenet checkpoint in this repository, they convert from MXNet/Gluon or pretrained on different dataset [here](https://github.com/osmr/imgclsmob/releases)), but I modified some implementation details in this srcipt, especially the final pooling choice and final pooling kernel size adjust strategy, which can be found in tensorflow vesion.

# Environment Requiements
* TensorFlow 1.x (test passed on 1.14)
* pytorch 1.x (test passed on 1.8.1)
* numpy 1.x (test passed on 1.17.3)

# All MobileNet V1 TensorFlow Checkpoint File
All MobileNet V1 pretrained checkpoint in TensorFlow version can be found and downloaded in [here](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.md), but I do not test to translate the quant version. All these checkpoint are achieved by traininig MobileNet v1 on the [ILSVRC-2012-CLS](https://image-net.org/challenges/LSVRC/2012/) image classification dataset, which is abbreviated to ImageNet Dataset. Please carefully choose the version of checkpoint for you specified usage. For example, the file named "MobileNet_v1_0.75_192" is corresponding to the model trained with depth multiplier is set to 0.75 and the input trainning image size is 192x192.

# Usage

## Translate the tensorflow checkpoint to pytorch checkpoint
Run `translated.py` script like
```
$ python translated.py --tf_ck tensorflow-checkpoint-dir --pytorch_ck save-translated-result-dir --tf_np save-immediate-numpy-form-data-dir[optional]
```
Among these arguments, the tf_np is optional, if you do not offer this argument, the script will not save any numpy translated dict.
For example, you run script like
```
python translated.py --tf_ck mobilenet_v1_1.0_128/mobilenet_v1_1.0_128.ckpt --pytorch_ck mobilenet_v1_1.0_128_torch.pth --tf_np mobilenet_v1_1.0_128
```
After the translate procedure, you will find mobilenet_v1_1.0_128_torch.pth and mobilenet_v1_1.0_128.npy in working directory.
## Use translated checkpoint in PyTorch script
The pytorch implementation version of Mobilenet V1 is in `mobilenet.py`, you can get mobilenet model by use the function `get_mobilenet` to get MobileNet V1 with specified parameters, such as width_scale (the same meaning for depth_multiplier in TensorFlow implementation), in_size (input image size), dropout (wether use dropout before classification layer) or global pool (wether use global pooling as the final pooling.
So, you can create MobileNet with 0.75 depth multiplier with input size 128, with global pooling and dropout with 0.8 possibility, and load the translated checkpoint in your script like:
```
from mobilenet import get_mobilenet

# create specified mobilenet v1
mobilenet_v1_d75_128 = get_mobilenet(width_scale=0.75, in_size=128, global_pool=True, dropout=0.8)
# load the correspoing checkpoint which translated in previous step
mobilenet_v1_d75_128.load_state_dict(torch.load("mobilenet_v1_0.75_128.pth"))

# other op with mobilenet
...

```

Having fun in this repository, current only support for translating the mobilenet v1 model, may support more models in the furture. Any questions, please feel free to let me know.
