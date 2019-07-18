#
# Copyright 2018 Analytics Zoo Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import torch
import torch.nn as nn
import os
import tempfile
import numpy as np

from pyspark import RDD
from bigdl.nn.layer import Layer
from bigdl.nn.criterion import Criterion
from zoo import getOrCreateSparkContext
from zoo.feature.image import ImageSet
from bigdl.util.common import callBigDlFunc
from zoo.pipeline.api.net.tfnet import to_sample_rdd

class TempModule(nn.Module):
    def __init__(self, lossFunc):
        super(TempModule, self).__init__()
        self.func = lossFunc

    def forward(self, x, y):
        return self.func(x, y)

class TorchIdentityCriterion(Criterion):
    def __init__(self):
        super(TorchIdentityCriterion, self).__init__(None, "float")



class TorchNet(Layer):
    """
    TorchNet wraps a TorchScript model as a single layer, thus the Pytorch model can be used for
    distributed inference or training.
    :param path: path to the TorchScript model.
    """

    def __init__(self, path, lossPath=None, bigdl_type="float"):
        super(TorchNet, self).__init__(None, bigdl_type, path, lossPath)

    @staticmethod
    def from_pytorch(module, input_shape, lossFunc=None, pred_shape=None, label_shape=None,
                     pred_input=None, label_input=None):
        """
        Create a TorchNet directly from PyTorch model, e.g. model in torchvision.models
        :param module: a PyTorch model
        :param input_shape: list of integers. E.g. for ResNet, this may be [1, 3, 224, 224]
        """
        temp = tempfile.mkdtemp()

        # save model
        traced_script_module = torch.jit.trace(module, torch.rand(input_shape))
        path = os.path.join(temp, "model.pt")
        traced_script_module.save(path)

        if lossFunc:
            pred_input = pred_input if pred_input is not None else torch.rand(pred_shape)
            label_input = label_input if label_input is not None else torch.rand(label_shape)

            traced_script_loss = torch.jit.trace(TempModule(lossFunc).eval(),
                                                 (pred_input, label_input))
            lossPath = os.path.join(temp, "loss.pt")
            traced_script_loss.save(lossPath)
            net = TorchNet(path, lossPath)
        else:
            net = TorchNet(path)

        return net

    def predict(self, x, batch_per_thread=1, distributed=True):
        """
        Use a model to do prediction.
        """
        if isinstance(x, ImageSet):
            results = callBigDlFunc(self.bigdl_type, "zooPredict",
                                    self.value,
                                    x,
                                    batch_per_thread)
            return ImageSet(results)
        if distributed:
            if isinstance(x, np.ndarray):
                data_rdd = to_sample_rdd(x, np.zeros([x.shape[0]]), getOrCreateSparkContext())
            elif isinstance(x, RDD):
                data_rdd = x
            else:
                raise TypeError("Unsupported prediction data type: %s" % type(x))
            results = callBigDlFunc(self.bigdl_type, "zooPredict",
                                    self.value,
                                    data_rdd,
                                    batch_per_thread)
            return results.map(lambda result: Layer.convert_output(result))
        else:
            if isinstance(x, np.ndarray) or isinstance(x, list):
                results = callBigDlFunc(self.bigdl_type, "zooPredict",
                                        self.value,
                                        self._to_jtensors(x),
                                        batch_per_thread)
                return [Layer.convert_output(result) for result in results]
            else:
                raise TypeError("Unsupported prediction data type: %s" % type(x))
