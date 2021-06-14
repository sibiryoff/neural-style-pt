import torch
from os import path
from sys import version_info
from collections import OrderedDict
from torch.utils.model_zoo import load_url


# Download the VGG-19 model and fix the layer names
print("Downloading the VGG-19 model")
sd = load_url("https://download.pytorch.org/models/vgg19-dcbb9e9d.pth")
map = {'classifier.1.weight':u'classifier.0.weight', 'classifier.1.bias':u'classifier.0.bias', 'classifier.4.weight':u'classifier.3.weight', 'classifier.4.bias':u'classifier.3.bias'}
sd = OrderedDict([(map[k] if k in map else k,v) for k,v in sd.items()])
torch.save(sd, path.join("/", "content", "neural-style-pt", "models", "vgg19.pth"))

# Download the VGG-16 model and fix the layer names
print("Downloading the VGG-16 model")
sd = load_url("https://download.pytorch.org/models/vgg16-397923af.pth")
map = {'classifier.1.weight':u'classifier.0.weight', 'classifier.1.bias':u'classifier.0.bias', 'classifier.4.weight':u'classifier.3.weight', 'classifier.4.bias':u'classifier.3.bias'}
sd = OrderedDict([(map[k] if k in map else k,v) for k,v in sd.items()])
torch.save(sd, path.join("/", "content", "neural-style-pt", "models", "vgg16.pth"))

# Download the NIN model
import urllib.request
urllib.request.urlretrieve("https://github.com/ProGamerGov/pytorch-nin/raw/master/nin_imagenet.pth", 
                            path.join("/", "content", "neural-style-pt", "models", "nin_imagenet.pth"))

print("All models have been successfully downloaded")
