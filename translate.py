import tensorflow as tf
import numpy as np
import torch
import collections
import argparse

channel_list = [[32], [64], [128, 128], [256, 256], [512, 512, 512, 512, 512, 512], [1024, 1024]]
name_list = {}
name_list['Conv2d_0'] = 'init_block'
index = 1
num = 0

# stage name translate dict gen
for i, data in enumerate(channel_list[1:]):
	stage_name = "stage%d" % (i+1)
	# print(type(data))
	for j in range(len(data)):
		unit_name = "unit%d" % (j+1)
		tf_name = "Conv2d_%d_" % (index)
		index += 1
		name_list[tf_name] = "%s.%s." % (stage_name, unit_name)

# different layer name translate, order is important
layer_trans = collections.OrderedDict({
	# bn layer name map
	"BatchNorm/beta": "bn.bias",
	"BatchNorm/gamma": "bn.weight",
	"BatchNorm/moving_variance": "bn.running_var",
	"BatchNorm/moving_mean": "bn.running_mean",

	# depth conv layer map
	"depthwise_weights": "weights",
	"depthwise/": "dw_conv.",

	# conv layer name map
	"weights": "conv.weight",

	# pointwise conv layer map
	"pointwise": "pw_conv",

	#
})


tf_suffix = "MobilenetV1"
tf_split = "/"
torch_suffix = "features"
torch_split = "."


def convert_name(name):
	name = name.replace(tf_suffix, torch_suffix)
	for key in name_list:
		name = name.replace(key, name_list[key])
	for key in layer_trans:
		name = name.replace(key, layer_trans[key])
	name = name.replace(tf_split, torch_split)

	return name


def main():
	parser = argparse.ArgumentParser(description="Translator from TensorFlow checkpoint to Pytorch Config (MobilenetV1 Specified Version)")

	parser.add_argument("--tf_ck", required=True, default=None, help="the TensorFlow checkpoint file path, ckpt name like 'mobilenet_v1_1.0_128.ckpt'")
	parser.add_argument("--pytorch_ck", default="torch_translated_checkpoint.pth", help="the checkpoint filename after translated to torch (*.pth file)")
	parser.add_argument("--tf_np", default=None, help="filename for saving tensorflow model in numpy file (*.npy), set None if do not want to save")
	args = parser.parse_args()

	tensorflow_checkpoint_file = args.tf_ck
	torch_checkpoint_save_file = args.torch_ck
	numpy_save_file = args.tf_np

	print("load TensorFlow checkpoint...")
	init_vars = tf.train.list_variables(tensorflow_checkpoint_file)

	# print(type(init_vars))
	# print(init_vars)

	if numpy_save_file:
		tf_mobilenet_dict_np = {}
	torch_translated_dict = {}

	for name, shape in init_vars:
		var = tf.train.load_variable(tensorflow_checkpoint_file, name)
		# print(name)
		# print(var.shape)
		# print(name, shape)
		# print(name)
		if numpy_save_file:
			tf_mobilenet_dict_np[name] = var
		# filter out the MovingAverage Var or Optimizer Var
		if name.split('/')[-1] in ["ExponentialMovingAverage", "RMSProp", "RMSProp_1"]:
			continue
		translated_name = convert_name(name)
		print(name, "->", translated_name)
		if var.ndim == 4:
			if 'dw' in translated_name:
				var = var.transpose((2, 3, 0, 1))
			else:
				var = var.transpose((3, 2, 0, 1))

		elif var.ndim == 1 or var.ndim == 0:
			var = np.array(var, dtype=np.float32)
		else:
			raise Exception("Error in dim")

		torch_translated_dict[translated_name] = torch.from_numpy(var)

	if numpy_save_file:
		np.save(numpy_save_file, tf_mobilenet_dict_np)
		# load value in numpy array like this
		# npfile = np.load("mobilenet_v1_128.npy", allow_pickle=True).item()
	torch.save(torch_translated_dict, torch_checkpoint_save_file)

if __name__ == "__main__":
	main()
