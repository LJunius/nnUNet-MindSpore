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

"""dice loss module"""
import mindspore
import mindspore.ops as ops
from mindspore import nn
from mindspore.nn.loss.loss import LossBase

import numpy as np
from src.nnunet.training.loss_functions.crossentropy import RobustCrossEntropyLoss, RobustCrossEntropyLoss2d
from src.nnunet.utilities.nd_softmax import softmax_helper


# class SoftDiceLoss(nn.DiceLoss):
#     """soft dice loss module"""

#     def __init__(self, apply_nonlin=None, batch_dice=False, do_bg=True, smooth=1., loss_type='3d'):
#         super(SoftDiceLoss, self).__init__()
#         self.mean = ops.ReduceMean()
#         self.do_bg = do_bg
#         self.batch_dice = batch_dice
#         self.apply_nonlin = apply_nonlin
#         self.smooth = smooth
#         self.reshape = ops.Reshape()
#         self.zeros = ops.Zeros()

def softmax_helper(data):
    """mindspore softmax"""
    return ops.Softmax(1)(data)

def get_tp_fp_fn_tn(net_output, gt, axes=None, mask=None, square=False):
    """
    net_output must be (b, c, x, y(, z)))
    gt must be a label map (shape (b, 1, x, y(, z)) OR shape (b, x, y(, z))) or one hot encoding (b, c, x, y(, z))
    if mask is provided it must have shape (b, 1, x, y(, z)))
    :param net_output:
    :param gt:
    :param axes: can be (, ) = no summation
    :param mask: mask must be 1 for valid pixels and 0 for invalid pixels
    :param square: if True then fp, tp and fn will be squared before summation
    :return:
    """
    def sum_tensor(inp, axes, keepdim=False):
        axes = np.unique(axes).astype(int)
        if keepdim:
            for ax in axes:
                inp = inp.sum(int(ax), keepdim=True)
        else:
            for ax in sorted(axes, reverse=True):
                inp = inp.sum(int(ax))
        return inp


    if axes is None:
        axes = tuple(range(2, len(net_output.size())))

    shp_x = net_output.shape
    shp_y = gt.shape

    if len(shp_x) != len(shp_y):
        gt = ops.stop_gradient(gt.view((shp_y[0], 1, *shp_y[1:])))

    if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
        # if this is the case then gt is probably already a one hot encoding
        y_onehot = ops.stop_gradient(gt)
    else:
        y_onehot = ops.stop_gradient(mindspore.Tensor(np.zeros(shp_x)))
        y_onehot[:,0,:] = (gt[:,0,:,:,:]==0)
        y_onehot[:,1,:] = (gt[:,0,:,:,:]==1)
        y_onehot[:,2,:] = (gt[:,0,:,:,:]==2)
        mycast = ops.Cast()
        y_onehot = mycast(y_onehot, mindspore.dtype.float32)
    tp = net_output * y_onehot
    fp = net_output * (1 - y_onehot)
    fn = (1 - net_output) * y_onehot
    tn = (1 - net_output) * (1 - y_onehot)
    if len(axes) > 0:
        tp = sum_tensor(tp, axes, keepdim=False)
        fp = sum_tensor(fp, axes, keepdim=False)
        fn = sum_tensor(fn, axes, keepdim=False)
        tn = sum_tensor(tn, axes, keepdim=False)

    return tp, fp, fn, tn

class SoftDiceLoss(nn.Cell):
    """soft dice loss module"""
    def __init__(self, apply_nonlin=None, batch_dice=False, do_bg=True, smooth=1. ,**soft_dice_kwargs):
        """
        """
        super(SoftDiceLoss, self).__init__()

        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth

    def construct(self, x, y, loss_mask=None):
        shp_x = x.shape
        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))
        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)
        tp, fp, fn, _ = get_tp_fp_fn_tn(x, y, axes, loss_mask, False)
        nominator = 2 * tp + self.smooth
        denominator = 2 * tp + fp + fn + self.smooth

        dc = nominator / (denominator + 1e-8)

        if not self.do_bg:
            if self.batch_dice:
                dc = dc[1:]
            else:
                dc = dc[:, 1:]
        dc = dc.mean()

        return -dc


class DC_and_CE_loss(LossBase):
    """Dice and cross entrophy loss"""
    def __init__(self, soft_dice_kwargs, ce_kwargs, aggregate="sum", square_dice=False, weight_ce=1, weight_dice=1,
                 log_dice=False, ignore_label=None):

        super(DC_and_CE_loss, self).__init__()
        if ignore_label is not None:
            assert not square_dice, 'not implemented'
            ce_kwargs['reduction'] = 'none'
        self.log_dice = log_dice
        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.aggregate = aggregate

        # if soft_dice_kwargs["loss_type"] == '3d':
        #     self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        # else:
        #     self.ce = RobustCrossEntropyLoss2d(**ce_kwargs)
        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.transpose = ops.Transpose()
        self.ignore_label = ignore_label
        self.reshape = ops.Reshape()

        self.dc = SoftDiceLoss(apply_nonlin=softmax_helper, **soft_dice_kwargs)

    def construct(self, net_output, target):
        """construct network"""
        dc_loss = self.dc(net_output, target) if self.weight_dice != 0 else 0
        target = target
        ce_loss = self.ce(net_output, target) if self.weight_ce != 0 else 0
        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        return result
