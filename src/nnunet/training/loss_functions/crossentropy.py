# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""crossentropy module"""
import mindspore
import mindspore.nn as nn
from mindspore import dtype as mstype
from mindspore.nn.loss.loss import LossBase
from mindspore.ops import operations as P

import numpy as np

class nnUnet_SoftmaxCrossEntropyWithLogits(LossBase):
    """
    This loss function is used for 3D segmentation compute loss
    """

    def __init__(self):
        super(nnUnet_SoftmaxCrossEntropyWithLogits, self).__init__()
        self.transpose = P.Transpose()
        self.reshape = P.Reshape()
        self.loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=False)
        self.cast = P.Cast()
        self.reduce_mean = P.ReduceMean()
        self.num_classes = 3  # task04 3 classfication

    def construct(self, logits, label):
        y_onehot = mindspore.ops.stop_gradient(mindspore.Tensor(np.zeros(logits.shape)))
        y_onehot[:,0,:] = (label[:,0,:,:,:]==0)
        y_onehot[:,1,:] = (label[:,0,:,:,:]==1)
        y_onehot[:,2,:] = (label[:,0,:,:,:]==2)

        label = mindspore.ops.stop_gradient(y_onehot)
        logits = self.transpose(logits, (0, 2, 3, 4, 1))
        label = self.transpose(label, (0, 2, 3, 4, 1))
        label = self.cast(label, mstype.float32)
        loss = self.reduce_mean(self.loss_fn(self.reshape(logits, (-1, self.num_classes)), \
                                             self.reshape(label, (-1, self.num_classes))))
        return self.get_loss(loss)


class nnUnet_SoftmaxCrossEntropyWithLogits2d(LossBase):
    """
    This loss function is used for 3D segmentation compute loss
    """

    def __init__(self):
        super(nnUnet_SoftmaxCrossEntropyWithLogits2d, self).__init__()
        self.transpose = P.Transpose()
        self.reshape = P.Reshape()
        self.loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=False)
        self.cast = P.Cast()
        self.reduce_mean = P.ReduceMean()
        self.num_classes = 3  # task04 3 classfication

    def construct(self, logits, label):
        logits = self.transpose(logits, (0, 2, 3, 1))
        label = self.transpose(label, (0, 2, 3, 1))
        label = self.cast(label, mstype.float32)
        loss = self.reduce_mean(self.loss_fn(self.reshape(logits, (-1, self.num_classes)), \
                                             self.reshape(label, (-1, self.num_classes))))
        return self.get_loss(loss)





RobustCrossEntropyLoss = nnUnet_SoftmaxCrossEntropyWithLogits
RobustCrossEntropyLoss2d = nnUnet_SoftmaxCrossEntropyWithLogits2d
