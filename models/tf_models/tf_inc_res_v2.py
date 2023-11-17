import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

_weights_dict = dict()


def load_weights(weight_file):
    if weight_file == None:
        return

    try:
        weights_dict = np.load(weight_file, allow_pickle=True).item()
    except:
        weights_dict = np.load(weight_file, allow_pickle=True, encoding='bytes').item()

    return weights_dict


class IncResV2KitModel(nn.Module):

    def __init__(self, weight_file):
        super(IncResV2KitModel, self).__init__()
        global _weights_dict
        _weights_dict = load_weights(weight_file)

        self.InceptionResnetV2_InceptionResnetV2_Conv2d_1a_3x3_Conv2D = self.__conv(2,
                                                                                    name='InceptionResnetV2/InceptionResnetV2/Conv2d_1a_3x3/Conv2D',
                                                                                    in_channels=3, out_channels=32,
                                                                                    kernel_size=(3, 3), stride=(2, 2),
                                                                                    groups=1, bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Conv2d_1a_3x3_BatchNorm_FusedBatchNorm = self.__batch_normalization(2,
                                                                                                                     'InceptionResnetV2/InceptionResnetV2/Conv2d_1a_3x3/BatchNorm/FusedBatchNorm',
                                                                                                                     num_features=32,
                                                                                                                     eps=0.0010000000474974513,
                                                                                                                     momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Conv2d_2a_3x3_Conv2D = self.__conv(2,
                                                                                    name='InceptionResnetV2/InceptionResnetV2/Conv2d_2a_3x3/Conv2D',
                                                                                    in_channels=32, out_channels=32,
                                                                                    kernel_size=(3, 3), stride=(1, 1),
                                                                                    groups=1, bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Conv2d_2a_3x3_BatchNorm_FusedBatchNorm = self.__batch_normalization(2,
                                                                                                                     'InceptionResnetV2/InceptionResnetV2/Conv2d_2a_3x3/BatchNorm/FusedBatchNorm',
                                                                                                                     num_features=32,
                                                                                                                     eps=0.0010000000474974513,
                                                                                                                     momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Conv2d_2b_3x3_Conv2D = self.__conv(2,
                                                                                    name='InceptionResnetV2/InceptionResnetV2/Conv2d_2b_3x3/Conv2D',
                                                                                    in_channels=32, out_channels=64,
                                                                                    kernel_size=(3, 3), stride=(1, 1),
                                                                                    groups=1, bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Conv2d_2b_3x3_BatchNorm_FusedBatchNorm = self.__batch_normalization(2,
                                                                                                                     'InceptionResnetV2/InceptionResnetV2/Conv2d_2b_3x3/BatchNorm/FusedBatchNorm',
                                                                                                                     num_features=64,
                                                                                                                     eps=0.0010000000474974513,
                                                                                                                     momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Conv2d_3b_1x1_Conv2D = self.__conv(2,
                                                                                    name='InceptionResnetV2/InceptionResnetV2/Conv2d_3b_1x1/Conv2D',
                                                                                    in_channels=64, out_channels=80,
                                                                                    kernel_size=(1, 1), stride=(1, 1),
                                                                                    groups=1, bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Conv2d_3b_1x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(2,
                                                                                                                     'InceptionResnetV2/InceptionResnetV2/Conv2d_3b_1x1/BatchNorm/FusedBatchNorm',
                                                                                                                     num_features=80,
                                                                                                                     eps=0.0010000000474974513,
                                                                                                                     momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Conv2d_4a_3x3_Conv2D = self.__conv(2,
                                                                                    name='InceptionResnetV2/InceptionResnetV2/Conv2d_4a_3x3/Conv2D',
                                                                                    in_channels=80, out_channels=192,
                                                                                    kernel_size=(3, 3), stride=(1, 1),
                                                                                    groups=1, bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Conv2d_4a_3x3_BatchNorm_FusedBatchNorm = self.__batch_normalization(2,
                                                                                                                     'InceptionResnetV2/InceptionResnetV2/Conv2d_4a_3x3/BatchNorm/FusedBatchNorm',
                                                                                                                     num_features=192,
                                                                                                                     eps=0.0010000000474974513,
                                                                                                                     momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Mixed_5b_Branch_0_Conv2d_1x1_Conv2D = self.__conv(2,
                                                                                                   name='InceptionResnetV2/InceptionResnetV2/Mixed_5b/Branch_0/Conv2d_1x1/Conv2D',
                                                                                                   in_channels=192,
                                                                                                   out_channels=96,
                                                                                                   kernel_size=(1, 1),
                                                                                                   stride=(1, 1),
                                                                                                   groups=1, bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Mixed_5b_Branch_1_Conv2d_0a_1x1_Conv2D = self.__conv(2,
                                                                                                      name='InceptionResnetV2/InceptionResnetV2/Mixed_5b/Branch_1/Conv2d_0a_1x1/Conv2D',
                                                                                                      in_channels=192,
                                                                                                      out_channels=48,
                                                                                                      kernel_size=(
                                                                                                          1, 1),
                                                                                                      stride=(1, 1),
                                                                                                      groups=1,
                                                                                                      bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Mixed_5b_Branch_2_Conv2d_0a_1x1_Conv2D = self.__conv(2,
                                                                                                      name='InceptionResnetV2/InceptionResnetV2/Mixed_5b/Branch_2/Conv2d_0a_1x1/Conv2D',
                                                                                                      in_channels=192,
                                                                                                      out_channels=64,
                                                                                                      kernel_size=(
                                                                                                          1, 1),
                                                                                                      stride=(1, 1),
                                                                                                      groups=1,
                                                                                                      bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Mixed_5b_Branch_0_Conv2d_1x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionResnetV2/InceptionResnetV2/Mixed_5b/Branch_0/Conv2d_1x1/BatchNorm/FusedBatchNorm',
            num_features=96, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Mixed_5b_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionResnetV2/InceptionResnetV2/Mixed_5b/Branch_1/Conv2d_0a_1x1/BatchNorm/FusedBatchNorm',
            num_features=48, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Mixed_5b_Branch_2_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionResnetV2/InceptionResnetV2/Mixed_5b/Branch_2/Conv2d_0a_1x1/BatchNorm/FusedBatchNorm',
            num_features=64, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Mixed_5b_Branch_3_Conv2d_0b_1x1_Conv2D = self.__conv(2,
                                                                                                      name='InceptionResnetV2/InceptionResnetV2/Mixed_5b/Branch_3/Conv2d_0b_1x1/Conv2D',
                                                                                                      in_channels=192,
                                                                                                      out_channels=64,
                                                                                                      kernel_size=(
                                                                                                          1, 1),
                                                                                                      stride=(1, 1),
                                                                                                      groups=1,
                                                                                                      bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Mixed_5b_Branch_3_Conv2d_0b_1x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionResnetV2/InceptionResnetV2/Mixed_5b/Branch_3/Conv2d_0b_1x1/BatchNorm/FusedBatchNorm',
            num_features=64, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Mixed_5b_Branch_1_Conv2d_0b_5x5_Conv2D = self.__conv(2,
                                                                                                      name='InceptionResnetV2/InceptionResnetV2/Mixed_5b/Branch_1/Conv2d_0b_5x5/Conv2D',
                                                                                                      in_channels=48,
                                                                                                      out_channels=64,
                                                                                                      kernel_size=(
                                                                                                          5, 5),
                                                                                                      stride=(1, 1),
                                                                                                      groups=1,
                                                                                                      bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Mixed_5b_Branch_2_Conv2d_0b_3x3_Conv2D = self.__conv(2,
                                                                                                      name='InceptionResnetV2/InceptionResnetV2/Mixed_5b/Branch_2/Conv2d_0b_3x3/Conv2D',
                                                                                                      in_channels=64,
                                                                                                      out_channels=96,
                                                                                                      kernel_size=(
                                                                                                          3, 3),
                                                                                                      stride=(1, 1),
                                                                                                      groups=1,
                                                                                                      bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Mixed_5b_Branch_1_Conv2d_0b_5x5_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionResnetV2/InceptionResnetV2/Mixed_5b/Branch_1/Conv2d_0b_5x5/BatchNorm/FusedBatchNorm',
            num_features=64, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Mixed_5b_Branch_2_Conv2d_0b_3x3_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionResnetV2/InceptionResnetV2/Mixed_5b/Branch_2/Conv2d_0b_3x3/BatchNorm/FusedBatchNorm',
            num_features=96, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Mixed_5b_Branch_2_Conv2d_0c_3x3_Conv2D = self.__conv(2,
                                                                                                      name='InceptionResnetV2/InceptionResnetV2/Mixed_5b/Branch_2/Conv2d_0c_3x3/Conv2D',
                                                                                                      in_channels=96,
                                                                                                      out_channels=96,
                                                                                                      kernel_size=(
                                                                                                          3, 3),
                                                                                                      stride=(1, 1),
                                                                                                      groups=1,
                                                                                                      bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Mixed_5b_Branch_2_Conv2d_0c_3x3_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionResnetV2/InceptionResnetV2/Mixed_5b/Branch_2/Conv2d_0c_3x3/BatchNorm/FusedBatchNorm',
            num_features=96, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_1_Branch_0_Conv2d_1x1_Conv2D = self.__conv(2,
                                                                                                           name='InceptionResnetV2/InceptionResnetV2/Repeat/block35_1/Branch_0/Conv2d_1x1/Conv2D',
                                                                                                           in_channels=320,
                                                                                                           out_channels=32,
                                                                                                           kernel_size=(
                                                                                                               1, 1),
                                                                                                           stride=(
                                                                                                               1, 1),
                                                                                                           groups=1,
                                                                                                           bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_1_Branch_1_Conv2d_0a_1x1_Conv2D = self.__conv(2,
                                                                                                              name='InceptionResnetV2/InceptionResnetV2/Repeat/block35_1/Branch_1/Conv2d_0a_1x1/Conv2D',
                                                                                                              in_channels=320,
                                                                                                              out_channels=32,
                                                                                                              kernel_size=(
                                                                                                                  1, 1),
                                                                                                              stride=(
                                                                                                                  1, 1),
                                                                                                              groups=1,
                                                                                                              bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_1_Branch_2_Conv2d_0a_1x1_Conv2D = self.__conv(2,
                                                                                                              name='InceptionResnetV2/InceptionResnetV2/Repeat/block35_1/Branch_2/Conv2d_0a_1x1/Conv2D',
                                                                                                              in_channels=320,
                                                                                                              out_channels=32,
                                                                                                              kernel_size=(
                                                                                                                  1, 1),
                                                                                                              stride=(
                                                                                                                  1, 1),
                                                                                                              groups=1,
                                                                                                              bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_1_Branch_0_Conv2d_1x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionResnetV2/InceptionResnetV2/Repeat/block35_1/Branch_0/Conv2d_1x1/BatchNorm/FusedBatchNorm',
            num_features=32, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_1_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionResnetV2/InceptionResnetV2/Repeat/block35_1/Branch_1/Conv2d_0a_1x1/BatchNorm/FusedBatchNorm',
            num_features=32, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_1_Branch_2_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionResnetV2/InceptionResnetV2/Repeat/block35_1/Branch_2/Conv2d_0a_1x1/BatchNorm/FusedBatchNorm',
            num_features=32, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_1_Branch_1_Conv2d_0b_3x3_Conv2D = self.__conv(2,
                                                                                                              name='InceptionResnetV2/InceptionResnetV2/Repeat/block35_1/Branch_1/Conv2d_0b_3x3/Conv2D',
                                                                                                              in_channels=32,
                                                                                                              out_channels=32,
                                                                                                              kernel_size=(
                                                                                                                  3, 3),
                                                                                                              stride=(
                                                                                                                  1, 1),
                                                                                                              groups=1,
                                                                                                              bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_1_Branch_2_Conv2d_0b_3x3_Conv2D = self.__conv(2,
                                                                                                              name='InceptionResnetV2/InceptionResnetV2/Repeat/block35_1/Branch_2/Conv2d_0b_3x3/Conv2D',
                                                                                                              in_channels=32,
                                                                                                              out_channels=48,
                                                                                                              kernel_size=(
                                                                                                                  3, 3),
                                                                                                              stride=(
                                                                                                                  1, 1),
                                                                                                              groups=1,
                                                                                                              bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_1_Branch_1_Conv2d_0b_3x3_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionResnetV2/InceptionResnetV2/Repeat/block35_1/Branch_1/Conv2d_0b_3x3/BatchNorm/FusedBatchNorm',
            num_features=32, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_1_Branch_2_Conv2d_0b_3x3_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionResnetV2/InceptionResnetV2/Repeat/block35_1/Branch_2/Conv2d_0b_3x3/BatchNorm/FusedBatchNorm',
            num_features=48, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_1_Branch_2_Conv2d_0c_3x3_Conv2D = self.__conv(2,
                                                                                                              name='InceptionResnetV2/InceptionResnetV2/Repeat/block35_1/Branch_2/Conv2d_0c_3x3/Conv2D',
                                                                                                              in_channels=48,
                                                                                                              out_channels=64,
                                                                                                              kernel_size=(
                                                                                                                  3, 3),
                                                                                                              stride=(
                                                                                                                  1, 1),
                                                                                                              groups=1,
                                                                                                              bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_1_Branch_2_Conv2d_0c_3x3_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionResnetV2/InceptionResnetV2/Repeat/block35_1/Branch_2/Conv2d_0c_3x3/BatchNorm/FusedBatchNorm',
            num_features=64, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_1_Conv2d_1x1_Conv2D = self.__conv(2,
                                                                                                  name='InceptionResnetV2/InceptionResnetV2/Repeat/block35_1/Conv2d_1x1/Conv2D',
                                                                                                  in_channels=128,
                                                                                                  out_channels=320,
                                                                                                  kernel_size=(1, 1),
                                                                                                  stride=(1, 1),
                                                                                                  groups=1, bias=True)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_2_Branch_0_Conv2d_1x1_Conv2D = self.__conv(2,
                                                                                                           name='InceptionResnetV2/InceptionResnetV2/Repeat/block35_2/Branch_0/Conv2d_1x1/Conv2D',
                                                                                                           in_channels=320,
                                                                                                           out_channels=32,
                                                                                                           kernel_size=(
                                                                                                               1, 1),
                                                                                                           stride=(
                                                                                                               1, 1),
                                                                                                           groups=1,
                                                                                                           bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_2_Branch_1_Conv2d_0a_1x1_Conv2D = self.__conv(2,
                                                                                                              name='InceptionResnetV2/InceptionResnetV2/Repeat/block35_2/Branch_1/Conv2d_0a_1x1/Conv2D',
                                                                                                              in_channels=320,
                                                                                                              out_channels=32,
                                                                                                              kernel_size=(
                                                                                                                  1, 1),
                                                                                                              stride=(
                                                                                                                  1, 1),
                                                                                                              groups=1,
                                                                                                              bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_2_Branch_2_Conv2d_0a_1x1_Conv2D = self.__conv(2,
                                                                                                              name='InceptionResnetV2/InceptionResnetV2/Repeat/block35_2/Branch_2/Conv2d_0a_1x1/Conv2D',
                                                                                                              in_channels=320,
                                                                                                              out_channels=32,
                                                                                                              kernel_size=(
                                                                                                                  1, 1),
                                                                                                              stride=(
                                                                                                                  1, 1),
                                                                                                              groups=1,
                                                                                                              bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_2_Branch_0_Conv2d_1x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionResnetV2/InceptionResnetV2/Repeat/block35_2/Branch_0/Conv2d_1x1/BatchNorm/FusedBatchNorm',
            num_features=32, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_2_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionResnetV2/InceptionResnetV2/Repeat/block35_2/Branch_1/Conv2d_0a_1x1/BatchNorm/FusedBatchNorm',
            num_features=32, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_2_Branch_2_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionResnetV2/InceptionResnetV2/Repeat/block35_2/Branch_2/Conv2d_0a_1x1/BatchNorm/FusedBatchNorm',
            num_features=32, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_2_Branch_1_Conv2d_0b_3x3_Conv2D = self.__conv(2,
                                                                                                              name='InceptionResnetV2/InceptionResnetV2/Repeat/block35_2/Branch_1/Conv2d_0b_3x3/Conv2D',
                                                                                                              in_channels=32,
                                                                                                              out_channels=32,
                                                                                                              kernel_size=(
                                                                                                                  3, 3),
                                                                                                              stride=(
                                                                                                                  1, 1),
                                                                                                              groups=1,
                                                                                                              bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_2_Branch_2_Conv2d_0b_3x3_Conv2D = self.__conv(2,
                                                                                                              name='InceptionResnetV2/InceptionResnetV2/Repeat/block35_2/Branch_2/Conv2d_0b_3x3/Conv2D',
                                                                                                              in_channels=32,
                                                                                                              out_channels=48,
                                                                                                              kernel_size=(
                                                                                                                  3, 3),
                                                                                                              stride=(
                                                                                                                  1, 1),
                                                                                                              groups=1,
                                                                                                              bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_2_Branch_1_Conv2d_0b_3x3_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionResnetV2/InceptionResnetV2/Repeat/block35_2/Branch_1/Conv2d_0b_3x3/BatchNorm/FusedBatchNorm',
            num_features=32, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_2_Branch_2_Conv2d_0b_3x3_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionResnetV2/InceptionResnetV2/Repeat/block35_2/Branch_2/Conv2d_0b_3x3/BatchNorm/FusedBatchNorm',
            num_features=48, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_2_Branch_2_Conv2d_0c_3x3_Conv2D = self.__conv(2,
                                                                                                              name='InceptionResnetV2/InceptionResnetV2/Repeat/block35_2/Branch_2/Conv2d_0c_3x3/Conv2D',
                                                                                                              in_channels=48,
                                                                                                              out_channels=64,
                                                                                                              kernel_size=(
                                                                                                                  3, 3),
                                                                                                              stride=(
                                                                                                                  1, 1),
                                                                                                              groups=1,
                                                                                                              bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_2_Branch_2_Conv2d_0c_3x3_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionResnetV2/InceptionResnetV2/Repeat/block35_2/Branch_2/Conv2d_0c_3x3/BatchNorm/FusedBatchNorm',
            num_features=64, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_2_Conv2d_1x1_Conv2D = self.__conv(2,
                                                                                                  name='InceptionResnetV2/InceptionResnetV2/Repeat/block35_2/Conv2d_1x1/Conv2D',
                                                                                                  in_channels=128,
                                                                                                  out_channels=320,
                                                                                                  kernel_size=(1, 1),
                                                                                                  stride=(1, 1),
                                                                                                  groups=1, bias=True)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_3_Branch_0_Conv2d_1x1_Conv2D = self.__conv(2,
                                                                                                           name='InceptionResnetV2/InceptionResnetV2/Repeat/block35_3/Branch_0/Conv2d_1x1/Conv2D',
                                                                                                           in_channels=320,
                                                                                                           out_channels=32,
                                                                                                           kernel_size=(
                                                                                                               1, 1),
                                                                                                           stride=(
                                                                                                               1, 1),
                                                                                                           groups=1,
                                                                                                           bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_3_Branch_1_Conv2d_0a_1x1_Conv2D = self.__conv(2,
                                                                                                              name='InceptionResnetV2/InceptionResnetV2/Repeat/block35_3/Branch_1/Conv2d_0a_1x1/Conv2D',
                                                                                                              in_channels=320,
                                                                                                              out_channels=32,
                                                                                                              kernel_size=(
                                                                                                                  1, 1),
                                                                                                              stride=(
                                                                                                                  1, 1),
                                                                                                              groups=1,
                                                                                                              bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_3_Branch_2_Conv2d_0a_1x1_Conv2D = self.__conv(2,
                                                                                                              name='InceptionResnetV2/InceptionResnetV2/Repeat/block35_3/Branch_2/Conv2d_0a_1x1/Conv2D',
                                                                                                              in_channels=320,
                                                                                                              out_channels=32,
                                                                                                              kernel_size=(
                                                                                                                  1, 1),
                                                                                                              stride=(
                                                                                                                  1, 1),
                                                                                                              groups=1,
                                                                                                              bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_3_Branch_0_Conv2d_1x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionResnetV2/InceptionResnetV2/Repeat/block35_3/Branch_0/Conv2d_1x1/BatchNorm/FusedBatchNorm',
            num_features=32, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_3_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionResnetV2/InceptionResnetV2/Repeat/block35_3/Branch_1/Conv2d_0a_1x1/BatchNorm/FusedBatchNorm',
            num_features=32, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_3_Branch_2_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionResnetV2/InceptionResnetV2/Repeat/block35_3/Branch_2/Conv2d_0a_1x1/BatchNorm/FusedBatchNorm',
            num_features=32, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_3_Branch_1_Conv2d_0b_3x3_Conv2D = self.__conv(2,
                                                                                                              name='InceptionResnetV2/InceptionResnetV2/Repeat/block35_3/Branch_1/Conv2d_0b_3x3/Conv2D',
                                                                                                              in_channels=32,
                                                                                                              out_channels=32,
                                                                                                              kernel_size=(
                                                                                                                  3, 3),
                                                                                                              stride=(
                                                                                                                  1, 1),
                                                                                                              groups=1,
                                                                                                              bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_3_Branch_2_Conv2d_0b_3x3_Conv2D = self.__conv(2,
                                                                                                              name='InceptionResnetV2/InceptionResnetV2/Repeat/block35_3/Branch_2/Conv2d_0b_3x3/Conv2D',
                                                                                                              in_channels=32,
                                                                                                              out_channels=48,
                                                                                                              kernel_size=(
                                                                                                                  3, 3),
                                                                                                              stride=(
                                                                                                                  1, 1),
                                                                                                              groups=1,
                                                                                                              bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_3_Branch_1_Conv2d_0b_3x3_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionResnetV2/InceptionResnetV2/Repeat/block35_3/Branch_1/Conv2d_0b_3x3/BatchNorm/FusedBatchNorm',
            num_features=32, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_3_Branch_2_Conv2d_0b_3x3_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionResnetV2/InceptionResnetV2/Repeat/block35_3/Branch_2/Conv2d_0b_3x3/BatchNorm/FusedBatchNorm',
            num_features=48, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_3_Branch_2_Conv2d_0c_3x3_Conv2D = self.__conv(2,
                                                                                                              name='InceptionResnetV2/InceptionResnetV2/Repeat/block35_3/Branch_2/Conv2d_0c_3x3/Conv2D',
                                                                                                              in_channels=48,
                                                                                                              out_channels=64,
                                                                                                              kernel_size=(
                                                                                                                  3, 3),
                                                                                                              stride=(
                                                                                                                  1, 1),
                                                                                                              groups=1,
                                                                                                              bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_3_Branch_2_Conv2d_0c_3x3_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionResnetV2/InceptionResnetV2/Repeat/block35_3/Branch_2/Conv2d_0c_3x3/BatchNorm/FusedBatchNorm',
            num_features=64, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_3_Conv2d_1x1_Conv2D = self.__conv(2,
                                                                                                  name='InceptionResnetV2/InceptionResnetV2/Repeat/block35_3/Conv2d_1x1/Conv2D',
                                                                                                  in_channels=128,
                                                                                                  out_channels=320,
                                                                                                  kernel_size=(1, 1),
                                                                                                  stride=(1, 1),
                                                                                                  groups=1, bias=True)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_4_Branch_0_Conv2d_1x1_Conv2D = self.__conv(2,
                                                                                                           name='InceptionResnetV2/InceptionResnetV2/Repeat/block35_4/Branch_0/Conv2d_1x1/Conv2D',
                                                                                                           in_channels=320,
                                                                                                           out_channels=32,
                                                                                                           kernel_size=(
                                                                                                               1, 1),
                                                                                                           stride=(
                                                                                                               1, 1),
                                                                                                           groups=1,
                                                                                                           bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_4_Branch_1_Conv2d_0a_1x1_Conv2D = self.__conv(2,
                                                                                                              name='InceptionResnetV2/InceptionResnetV2/Repeat/block35_4/Branch_1/Conv2d_0a_1x1/Conv2D',
                                                                                                              in_channels=320,
                                                                                                              out_channels=32,
                                                                                                              kernel_size=(
                                                                                                                  1, 1),
                                                                                                              stride=(
                                                                                                                  1, 1),
                                                                                                              groups=1,
                                                                                                              bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_4_Branch_2_Conv2d_0a_1x1_Conv2D = self.__conv(2,
                                                                                                              name='InceptionResnetV2/InceptionResnetV2/Repeat/block35_4/Branch_2/Conv2d_0a_1x1/Conv2D',
                                                                                                              in_channels=320,
                                                                                                              out_channels=32,
                                                                                                              kernel_size=(
                                                                                                                  1, 1),
                                                                                                              stride=(
                                                                                                                  1, 1),
                                                                                                              groups=1,
                                                                                                              bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_4_Branch_0_Conv2d_1x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionResnetV2/InceptionResnetV2/Repeat/block35_4/Branch_0/Conv2d_1x1/BatchNorm/FusedBatchNorm',
            num_features=32, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_4_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionResnetV2/InceptionResnetV2/Repeat/block35_4/Branch_1/Conv2d_0a_1x1/BatchNorm/FusedBatchNorm',
            num_features=32, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_4_Branch_2_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionResnetV2/InceptionResnetV2/Repeat/block35_4/Branch_2/Conv2d_0a_1x1/BatchNorm/FusedBatchNorm',
            num_features=32, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_4_Branch_1_Conv2d_0b_3x3_Conv2D = self.__conv(2,
                                                                                                              name='InceptionResnetV2/InceptionResnetV2/Repeat/block35_4/Branch_1/Conv2d_0b_3x3/Conv2D',
                                                                                                              in_channels=32,
                                                                                                              out_channels=32,
                                                                                                              kernel_size=(
                                                                                                                  3, 3),
                                                                                                              stride=(
                                                                                                                  1, 1),
                                                                                                              groups=1,
                                                                                                              bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_4_Branch_2_Conv2d_0b_3x3_Conv2D = self.__conv(2,
                                                                                                              name='InceptionResnetV2/InceptionResnetV2/Repeat/block35_4/Branch_2/Conv2d_0b_3x3/Conv2D',
                                                                                                              in_channels=32,
                                                                                                              out_channels=48,
                                                                                                              kernel_size=(
                                                                                                                  3, 3),
                                                                                                              stride=(
                                                                                                                  1, 1),
                                                                                                              groups=1,
                                                                                                              bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_4_Branch_1_Conv2d_0b_3x3_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionResnetV2/InceptionResnetV2/Repeat/block35_4/Branch_1/Conv2d_0b_3x3/BatchNorm/FusedBatchNorm',
            num_features=32, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_4_Branch_2_Conv2d_0b_3x3_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionResnetV2/InceptionResnetV2/Repeat/block35_4/Branch_2/Conv2d_0b_3x3/BatchNorm/FusedBatchNorm',
            num_features=48, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_4_Branch_2_Conv2d_0c_3x3_Conv2D = self.__conv(2,
                                                                                                              name='InceptionResnetV2/InceptionResnetV2/Repeat/block35_4/Branch_2/Conv2d_0c_3x3/Conv2D',
                                                                                                              in_channels=48,
                                                                                                              out_channels=64,
                                                                                                              kernel_size=(
                                                                                                                  3, 3),
                                                                                                              stride=(
                                                                                                                  1, 1),
                                                                                                              groups=1,
                                                                                                              bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_4_Branch_2_Conv2d_0c_3x3_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionResnetV2/InceptionResnetV2/Repeat/block35_4/Branch_2/Conv2d_0c_3x3/BatchNorm/FusedBatchNorm',
            num_features=64, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_4_Conv2d_1x1_Conv2D = self.__conv(2,
                                                                                                  name='InceptionResnetV2/InceptionResnetV2/Repeat/block35_4/Conv2d_1x1/Conv2D',
                                                                                                  in_channels=128,
                                                                                                  out_channels=320,
                                                                                                  kernel_size=(1, 1),
                                                                                                  stride=(1, 1),
                                                                                                  groups=1, bias=True)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_5_Branch_0_Conv2d_1x1_Conv2D = self.__conv(2,
                                                                                                           name='InceptionResnetV2/InceptionResnetV2/Repeat/block35_5/Branch_0/Conv2d_1x1/Conv2D',
                                                                                                           in_channels=320,
                                                                                                           out_channels=32,
                                                                                                           kernel_size=(
                                                                                                               1, 1),
                                                                                                           stride=(
                                                                                                               1, 1),
                                                                                                           groups=1,
                                                                                                           bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_5_Branch_1_Conv2d_0a_1x1_Conv2D = self.__conv(2,
                                                                                                              name='InceptionResnetV2/InceptionResnetV2/Repeat/block35_5/Branch_1/Conv2d_0a_1x1/Conv2D',
                                                                                                              in_channels=320,
                                                                                                              out_channels=32,
                                                                                                              kernel_size=(
                                                                                                                  1, 1),
                                                                                                              stride=(
                                                                                                                  1, 1),
                                                                                                              groups=1,
                                                                                                              bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_5_Branch_2_Conv2d_0a_1x1_Conv2D = self.__conv(2,
                                                                                                              name='InceptionResnetV2/InceptionResnetV2/Repeat/block35_5/Branch_2/Conv2d_0a_1x1/Conv2D',
                                                                                                              in_channels=320,
                                                                                                              out_channels=32,
                                                                                                              kernel_size=(
                                                                                                                  1, 1),
                                                                                                              stride=(
                                                                                                                  1, 1),
                                                                                                              groups=1,
                                                                                                              bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_5_Branch_0_Conv2d_1x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionResnetV2/InceptionResnetV2/Repeat/block35_5/Branch_0/Conv2d_1x1/BatchNorm/FusedBatchNorm',
            num_features=32, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_5_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionResnetV2/InceptionResnetV2/Repeat/block35_5/Branch_1/Conv2d_0a_1x1/BatchNorm/FusedBatchNorm',
            num_features=32, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_5_Branch_2_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionResnetV2/InceptionResnetV2/Repeat/block35_5/Branch_2/Conv2d_0a_1x1/BatchNorm/FusedBatchNorm',
            num_features=32, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_5_Branch_1_Conv2d_0b_3x3_Conv2D = self.__conv(2,
                                                                                                              name='InceptionResnetV2/InceptionResnetV2/Repeat/block35_5/Branch_1/Conv2d_0b_3x3/Conv2D',
                                                                                                              in_channels=32,
                                                                                                              out_channels=32,
                                                                                                              kernel_size=(
                                                                                                                  3, 3),
                                                                                                              stride=(
                                                                                                                  1, 1),
                                                                                                              groups=1,
                                                                                                              bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_5_Branch_2_Conv2d_0b_3x3_Conv2D = self.__conv(2,
                                                                                                              name='InceptionResnetV2/InceptionResnetV2/Repeat/block35_5/Branch_2/Conv2d_0b_3x3/Conv2D',
                                                                                                              in_channels=32,
                                                                                                              out_channels=48,
                                                                                                              kernel_size=(
                                                                                                                  3, 3),
                                                                                                              stride=(
                                                                                                                  1, 1),
                                                                                                              groups=1,
                                                                                                              bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_5_Branch_1_Conv2d_0b_3x3_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionResnetV2/InceptionResnetV2/Repeat/block35_5/Branch_1/Conv2d_0b_3x3/BatchNorm/FusedBatchNorm',
            num_features=32, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_5_Branch_2_Conv2d_0b_3x3_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionResnetV2/InceptionResnetV2/Repeat/block35_5/Branch_2/Conv2d_0b_3x3/BatchNorm/FusedBatchNorm',
            num_features=48, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_5_Branch_2_Conv2d_0c_3x3_Conv2D = self.__conv(2,
                                                                                                              name='InceptionResnetV2/InceptionResnetV2/Repeat/block35_5/Branch_2/Conv2d_0c_3x3/Conv2D',
                                                                                                              in_channels=48,
                                                                                                              out_channels=64,
                                                                                                              kernel_size=(
                                                                                                                  3, 3),
                                                                                                              stride=(
                                                                                                                  1, 1),
                                                                                                              groups=1,
                                                                                                              bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_5_Branch_2_Conv2d_0c_3x3_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionResnetV2/InceptionResnetV2/Repeat/block35_5/Branch_2/Conv2d_0c_3x3/BatchNorm/FusedBatchNorm',
            num_features=64, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_5_Conv2d_1x1_Conv2D = self.__conv(2,
                                                                                                  name='InceptionResnetV2/InceptionResnetV2/Repeat/block35_5/Conv2d_1x1/Conv2D',
                                                                                                  in_channels=128,
                                                                                                  out_channels=320,
                                                                                                  kernel_size=(1, 1),
                                                                                                  stride=(1, 1),
                                                                                                  groups=1, bias=True)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_6_Branch_0_Conv2d_1x1_Conv2D = self.__conv(2,
                                                                                                           name='InceptionResnetV2/InceptionResnetV2/Repeat/block35_6/Branch_0/Conv2d_1x1/Conv2D',
                                                                                                           in_channels=320,
                                                                                                           out_channels=32,
                                                                                                           kernel_size=(
                                                                                                               1, 1),
                                                                                                           stride=(
                                                                                                               1, 1),
                                                                                                           groups=1,
                                                                                                           bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_6_Branch_1_Conv2d_0a_1x1_Conv2D = self.__conv(2,
                                                                                                              name='InceptionResnetV2/InceptionResnetV2/Repeat/block35_6/Branch_1/Conv2d_0a_1x1/Conv2D',
                                                                                                              in_channels=320,
                                                                                                              out_channels=32,
                                                                                                              kernel_size=(
                                                                                                                  1, 1),
                                                                                                              stride=(
                                                                                                                  1, 1),
                                                                                                              groups=1,
                                                                                                              bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_6_Branch_2_Conv2d_0a_1x1_Conv2D = self.__conv(2,
                                                                                                              name='InceptionResnetV2/InceptionResnetV2/Repeat/block35_6/Branch_2/Conv2d_0a_1x1/Conv2D',
                                                                                                              in_channels=320,
                                                                                                              out_channels=32,
                                                                                                              kernel_size=(
                                                                                                                  1, 1),
                                                                                                              stride=(
                                                                                                                  1, 1),
                                                                                                              groups=1,
                                                                                                              bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_6_Branch_0_Conv2d_1x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionResnetV2/InceptionResnetV2/Repeat/block35_6/Branch_0/Conv2d_1x1/BatchNorm/FusedBatchNorm',
            num_features=32, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_6_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionResnetV2/InceptionResnetV2/Repeat/block35_6/Branch_1/Conv2d_0a_1x1/BatchNorm/FusedBatchNorm',
            num_features=32, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_6_Branch_2_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionResnetV2/InceptionResnetV2/Repeat/block35_6/Branch_2/Conv2d_0a_1x1/BatchNorm/FusedBatchNorm',
            num_features=32, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_6_Branch_1_Conv2d_0b_3x3_Conv2D = self.__conv(2,
                                                                                                              name='InceptionResnetV2/InceptionResnetV2/Repeat/block35_6/Branch_1/Conv2d_0b_3x3/Conv2D',
                                                                                                              in_channels=32,
                                                                                                              out_channels=32,
                                                                                                              kernel_size=(
                                                                                                                  3, 3),
                                                                                                              stride=(
                                                                                                                  1, 1),
                                                                                                              groups=1,
                                                                                                              bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_6_Branch_2_Conv2d_0b_3x3_Conv2D = self.__conv(2,
                                                                                                              name='InceptionResnetV2/InceptionResnetV2/Repeat/block35_6/Branch_2/Conv2d_0b_3x3/Conv2D',
                                                                                                              in_channels=32,
                                                                                                              out_channels=48,
                                                                                                              kernel_size=(
                                                                                                                  3, 3),
                                                                                                              stride=(
                                                                                                                  1, 1),
                                                                                                              groups=1,
                                                                                                              bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_6_Branch_1_Conv2d_0b_3x3_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionResnetV2/InceptionResnetV2/Repeat/block35_6/Branch_1/Conv2d_0b_3x3/BatchNorm/FusedBatchNorm',
            num_features=32, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_6_Branch_2_Conv2d_0b_3x3_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionResnetV2/InceptionResnetV2/Repeat/block35_6/Branch_2/Conv2d_0b_3x3/BatchNorm/FusedBatchNorm',
            num_features=48, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_6_Branch_2_Conv2d_0c_3x3_Conv2D = self.__conv(2,
                                                                                                              name='InceptionResnetV2/InceptionResnetV2/Repeat/block35_6/Branch_2/Conv2d_0c_3x3/Conv2D',
                                                                                                              in_channels=48,
                                                                                                              out_channels=64,
                                                                                                              kernel_size=(
                                                                                                                  3, 3),
                                                                                                              stride=(
                                                                                                                  1, 1),
                                                                                                              groups=1,
                                                                                                              bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_6_Branch_2_Conv2d_0c_3x3_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionResnetV2/InceptionResnetV2/Repeat/block35_6/Branch_2/Conv2d_0c_3x3/BatchNorm/FusedBatchNorm',
            num_features=64, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_6_Conv2d_1x1_Conv2D = self.__conv(2,
                                                                                                  name='InceptionResnetV2/InceptionResnetV2/Repeat/block35_6/Conv2d_1x1/Conv2D',
                                                                                                  in_channels=128,
                                                                                                  out_channels=320,
                                                                                                  kernel_size=(1, 1),
                                                                                                  stride=(1, 1),
                                                                                                  groups=1, bias=True)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_7_Branch_0_Conv2d_1x1_Conv2D = self.__conv(2,
                                                                                                           name='InceptionResnetV2/InceptionResnetV2/Repeat/block35_7/Branch_0/Conv2d_1x1/Conv2D',
                                                                                                           in_channels=320,
                                                                                                           out_channels=32,
                                                                                                           kernel_size=(
                                                                                                               1, 1),
                                                                                                           stride=(
                                                                                                               1, 1),
                                                                                                           groups=1,
                                                                                                           bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_7_Branch_1_Conv2d_0a_1x1_Conv2D = self.__conv(2,
                                                                                                              name='InceptionResnetV2/InceptionResnetV2/Repeat/block35_7/Branch_1/Conv2d_0a_1x1/Conv2D',
                                                                                                              in_channels=320,
                                                                                                              out_channels=32,
                                                                                                              kernel_size=(
                                                                                                                  1, 1),
                                                                                                              stride=(
                                                                                                                  1, 1),
                                                                                                              groups=1,
                                                                                                              bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_7_Branch_2_Conv2d_0a_1x1_Conv2D = self.__conv(2,
                                                                                                              name='InceptionResnetV2/InceptionResnetV2/Repeat/block35_7/Branch_2/Conv2d_0a_1x1/Conv2D',
                                                                                                              in_channels=320,
                                                                                                              out_channels=32,
                                                                                                              kernel_size=(
                                                                                                                  1, 1),
                                                                                                              stride=(
                                                                                                                  1, 1),
                                                                                                              groups=1,
                                                                                                              bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_7_Branch_0_Conv2d_1x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionResnetV2/InceptionResnetV2/Repeat/block35_7/Branch_0/Conv2d_1x1/BatchNorm/FusedBatchNorm',
            num_features=32, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_7_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionResnetV2/InceptionResnetV2/Repeat/block35_7/Branch_1/Conv2d_0a_1x1/BatchNorm/FusedBatchNorm',
            num_features=32, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_7_Branch_2_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionResnetV2/InceptionResnetV2/Repeat/block35_7/Branch_2/Conv2d_0a_1x1/BatchNorm/FusedBatchNorm',
            num_features=32, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_7_Branch_1_Conv2d_0b_3x3_Conv2D = self.__conv(2,
                                                                                                              name='InceptionResnetV2/InceptionResnetV2/Repeat/block35_7/Branch_1/Conv2d_0b_3x3/Conv2D',
                                                                                                              in_channels=32,
                                                                                                              out_channels=32,
                                                                                                              kernel_size=(
                                                                                                                  3, 3),
                                                                                                              stride=(
                                                                                                                  1, 1),
                                                                                                              groups=1,
                                                                                                              bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_7_Branch_2_Conv2d_0b_3x3_Conv2D = self.__conv(2,
                                                                                                              name='InceptionResnetV2/InceptionResnetV2/Repeat/block35_7/Branch_2/Conv2d_0b_3x3/Conv2D',
                                                                                                              in_channels=32,
                                                                                                              out_channels=48,
                                                                                                              kernel_size=(
                                                                                                                  3, 3),
                                                                                                              stride=(
                                                                                                                  1, 1),
                                                                                                              groups=1,
                                                                                                              bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_7_Branch_1_Conv2d_0b_3x3_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionResnetV2/InceptionResnetV2/Repeat/block35_7/Branch_1/Conv2d_0b_3x3/BatchNorm/FusedBatchNorm',
            num_features=32, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_7_Branch_2_Conv2d_0b_3x3_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionResnetV2/InceptionResnetV2/Repeat/block35_7/Branch_2/Conv2d_0b_3x3/BatchNorm/FusedBatchNorm',
            num_features=48, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_7_Branch_2_Conv2d_0c_3x3_Conv2D = self.__conv(2,
                                                                                                              name='InceptionResnetV2/InceptionResnetV2/Repeat/block35_7/Branch_2/Conv2d_0c_3x3/Conv2D',
                                                                                                              in_channels=48,
                                                                                                              out_channels=64,
                                                                                                              kernel_size=(
                                                                                                                  3, 3),
                                                                                                              stride=(
                                                                                                                  1, 1),
                                                                                                              groups=1,
                                                                                                              bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_7_Branch_2_Conv2d_0c_3x3_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionResnetV2/InceptionResnetV2/Repeat/block35_7/Branch_2/Conv2d_0c_3x3/BatchNorm/FusedBatchNorm',
            num_features=64, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_7_Conv2d_1x1_Conv2D = self.__conv(2,
                                                                                                  name='InceptionResnetV2/InceptionResnetV2/Repeat/block35_7/Conv2d_1x1/Conv2D',
                                                                                                  in_channels=128,
                                                                                                  out_channels=320,
                                                                                                  kernel_size=(1, 1),
                                                                                                  stride=(1, 1),
                                                                                                  groups=1, bias=True)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_8_Branch_0_Conv2d_1x1_Conv2D = self.__conv(2,
                                                                                                           name='InceptionResnetV2/InceptionResnetV2/Repeat/block35_8/Branch_0/Conv2d_1x1/Conv2D',
                                                                                                           in_channels=320,
                                                                                                           out_channels=32,
                                                                                                           kernel_size=(
                                                                                                               1, 1),
                                                                                                           stride=(
                                                                                                               1, 1),
                                                                                                           groups=1,
                                                                                                           bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_8_Branch_1_Conv2d_0a_1x1_Conv2D = self.__conv(2,
                                                                                                              name='InceptionResnetV2/InceptionResnetV2/Repeat/block35_8/Branch_1/Conv2d_0a_1x1/Conv2D',
                                                                                                              in_channels=320,
                                                                                                              out_channels=32,
                                                                                                              kernel_size=(
                                                                                                                  1, 1),
                                                                                                              stride=(
                                                                                                                  1, 1),
                                                                                                              groups=1,
                                                                                                              bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_8_Branch_2_Conv2d_0a_1x1_Conv2D = self.__conv(2,
                                                                                                              name='InceptionResnetV2/InceptionResnetV2/Repeat/block35_8/Branch_2/Conv2d_0a_1x1/Conv2D',
                                                                                                              in_channels=320,
                                                                                                              out_channels=32,
                                                                                                              kernel_size=(
                                                                                                                  1, 1),
                                                                                                              stride=(
                                                                                                                  1, 1),
                                                                                                              groups=1,
                                                                                                              bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_8_Branch_0_Conv2d_1x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionResnetV2/InceptionResnetV2/Repeat/block35_8/Branch_0/Conv2d_1x1/BatchNorm/FusedBatchNorm',
            num_features=32, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_8_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionResnetV2/InceptionResnetV2/Repeat/block35_8/Branch_1/Conv2d_0a_1x1/BatchNorm/FusedBatchNorm',
            num_features=32, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_8_Branch_2_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionResnetV2/InceptionResnetV2/Repeat/block35_8/Branch_2/Conv2d_0a_1x1/BatchNorm/FusedBatchNorm',
            num_features=32, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_8_Branch_1_Conv2d_0b_3x3_Conv2D = self.__conv(2,
                                                                                                              name='InceptionResnetV2/InceptionResnetV2/Repeat/block35_8/Branch_1/Conv2d_0b_3x3/Conv2D',
                                                                                                              in_channels=32,
                                                                                                              out_channels=32,
                                                                                                              kernel_size=(
                                                                                                                  3, 3),
                                                                                                              stride=(
                                                                                                                  1, 1),
                                                                                                              groups=1,
                                                                                                              bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_8_Branch_2_Conv2d_0b_3x3_Conv2D = self.__conv(2,
                                                                                                              name='InceptionResnetV2/InceptionResnetV2/Repeat/block35_8/Branch_2/Conv2d_0b_3x3/Conv2D',
                                                                                                              in_channels=32,
                                                                                                              out_channels=48,
                                                                                                              kernel_size=(
                                                                                                                  3, 3),
                                                                                                              stride=(
                                                                                                                  1, 1),
                                                                                                              groups=1,
                                                                                                              bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_8_Branch_1_Conv2d_0b_3x3_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionResnetV2/InceptionResnetV2/Repeat/block35_8/Branch_1/Conv2d_0b_3x3/BatchNorm/FusedBatchNorm',
            num_features=32, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_8_Branch_2_Conv2d_0b_3x3_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionResnetV2/InceptionResnetV2/Repeat/block35_8/Branch_2/Conv2d_0b_3x3/BatchNorm/FusedBatchNorm',
            num_features=48, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_8_Branch_2_Conv2d_0c_3x3_Conv2D = self.__conv(2,
                                                                                                              name='InceptionResnetV2/InceptionResnetV2/Repeat/block35_8/Branch_2/Conv2d_0c_3x3/Conv2D',
                                                                                                              in_channels=48,
                                                                                                              out_channels=64,
                                                                                                              kernel_size=(
                                                                                                                  3, 3),
                                                                                                              stride=(
                                                                                                                  1, 1),
                                                                                                              groups=1,
                                                                                                              bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_8_Branch_2_Conv2d_0c_3x3_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionResnetV2/InceptionResnetV2/Repeat/block35_8/Branch_2/Conv2d_0c_3x3/BatchNorm/FusedBatchNorm',
            num_features=64, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_8_Conv2d_1x1_Conv2D = self.__conv(2,
                                                                                                  name='InceptionResnetV2/InceptionResnetV2/Repeat/block35_8/Conv2d_1x1/Conv2D',
                                                                                                  in_channels=128,
                                                                                                  out_channels=320,
                                                                                                  kernel_size=(1, 1),
                                                                                                  stride=(1, 1),
                                                                                                  groups=1, bias=True)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_9_Branch_0_Conv2d_1x1_Conv2D = self.__conv(2,
                                                                                                           name='InceptionResnetV2/InceptionResnetV2/Repeat/block35_9/Branch_0/Conv2d_1x1/Conv2D',
                                                                                                           in_channels=320,
                                                                                                           out_channels=32,
                                                                                                           kernel_size=(
                                                                                                               1, 1),
                                                                                                           stride=(
                                                                                                               1, 1),
                                                                                                           groups=1,
                                                                                                           bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_9_Branch_1_Conv2d_0a_1x1_Conv2D = self.__conv(2,
                                                                                                              name='InceptionResnetV2/InceptionResnetV2/Repeat/block35_9/Branch_1/Conv2d_0a_1x1/Conv2D',
                                                                                                              in_channels=320,
                                                                                                              out_channels=32,
                                                                                                              kernel_size=(
                                                                                                                  1, 1),
                                                                                                              stride=(
                                                                                                                  1, 1),
                                                                                                              groups=1,
                                                                                                              bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_9_Branch_2_Conv2d_0a_1x1_Conv2D = self.__conv(2,
                                                                                                              name='InceptionResnetV2/InceptionResnetV2/Repeat/block35_9/Branch_2/Conv2d_0a_1x1/Conv2D',
                                                                                                              in_channels=320,
                                                                                                              out_channels=32,
                                                                                                              kernel_size=(
                                                                                                                  1, 1),
                                                                                                              stride=(
                                                                                                                  1, 1),
                                                                                                              groups=1,
                                                                                                              bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_9_Branch_0_Conv2d_1x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionResnetV2/InceptionResnetV2/Repeat/block35_9/Branch_0/Conv2d_1x1/BatchNorm/FusedBatchNorm',
            num_features=32, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_9_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionResnetV2/InceptionResnetV2/Repeat/block35_9/Branch_1/Conv2d_0a_1x1/BatchNorm/FusedBatchNorm',
            num_features=32, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_9_Branch_2_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionResnetV2/InceptionResnetV2/Repeat/block35_9/Branch_2/Conv2d_0a_1x1/BatchNorm/FusedBatchNorm',
            num_features=32, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_9_Branch_1_Conv2d_0b_3x3_Conv2D = self.__conv(2,
                                                                                                              name='InceptionResnetV2/InceptionResnetV2/Repeat/block35_9/Branch_1/Conv2d_0b_3x3/Conv2D',
                                                                                                              in_channels=32,
                                                                                                              out_channels=32,
                                                                                                              kernel_size=(
                                                                                                                  3, 3),
                                                                                                              stride=(
                                                                                                                  1, 1),
                                                                                                              groups=1,
                                                                                                              bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_9_Branch_2_Conv2d_0b_3x3_Conv2D = self.__conv(2,
                                                                                                              name='InceptionResnetV2/InceptionResnetV2/Repeat/block35_9/Branch_2/Conv2d_0b_3x3/Conv2D',
                                                                                                              in_channels=32,
                                                                                                              out_channels=48,
                                                                                                              kernel_size=(
                                                                                                                  3, 3),
                                                                                                              stride=(
                                                                                                                  1, 1),
                                                                                                              groups=1,
                                                                                                              bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_9_Branch_1_Conv2d_0b_3x3_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionResnetV2/InceptionResnetV2/Repeat/block35_9/Branch_1/Conv2d_0b_3x3/BatchNorm/FusedBatchNorm',
            num_features=32, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_9_Branch_2_Conv2d_0b_3x3_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionResnetV2/InceptionResnetV2/Repeat/block35_9/Branch_2/Conv2d_0b_3x3/BatchNorm/FusedBatchNorm',
            num_features=48, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_9_Branch_2_Conv2d_0c_3x3_Conv2D = self.__conv(2,
                                                                                                              name='InceptionResnetV2/InceptionResnetV2/Repeat/block35_9/Branch_2/Conv2d_0c_3x3/Conv2D',
                                                                                                              in_channels=48,
                                                                                                              out_channels=64,
                                                                                                              kernel_size=(
                                                                                                                  3, 3),
                                                                                                              stride=(
                                                                                                                  1, 1),
                                                                                                              groups=1,
                                                                                                              bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_9_Branch_2_Conv2d_0c_3x3_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionResnetV2/InceptionResnetV2/Repeat/block35_9/Branch_2/Conv2d_0c_3x3/BatchNorm/FusedBatchNorm',
            num_features=64, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_9_Conv2d_1x1_Conv2D = self.__conv(2,
                                                                                                  name='InceptionResnetV2/InceptionResnetV2/Repeat/block35_9/Conv2d_1x1/Conv2D',
                                                                                                  in_channels=128,
                                                                                                  out_channels=320,
                                                                                                  kernel_size=(1, 1),
                                                                                                  stride=(1, 1),
                                                                                                  groups=1, bias=True)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_10_Branch_0_Conv2d_1x1_Conv2D = self.__conv(2,
                                                                                                            name='InceptionResnetV2/InceptionResnetV2/Repeat/block35_10/Branch_0/Conv2d_1x1/Conv2D',
                                                                                                            in_channels=320,
                                                                                                            out_channels=32,
                                                                                                            kernel_size=(
                                                                                                                1, 1),
                                                                                                            stride=(
                                                                                                                1, 1),
                                                                                                            groups=1,
                                                                                                            bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_10_Branch_1_Conv2d_0a_1x1_Conv2D = self.__conv(2,
                                                                                                               name='InceptionResnetV2/InceptionResnetV2/Repeat/block35_10/Branch_1/Conv2d_0a_1x1/Conv2D',
                                                                                                               in_channels=320,
                                                                                                               out_channels=32,
                                                                                                               kernel_size=(
                                                                                                                   1,
                                                                                                                   1),
                                                                                                               stride=(
                                                                                                                   1,
                                                                                                                   1),
                                                                                                               groups=1,
                                                                                                               bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_10_Branch_2_Conv2d_0a_1x1_Conv2D = self.__conv(2,
                                                                                                               name='InceptionResnetV2/InceptionResnetV2/Repeat/block35_10/Branch_2/Conv2d_0a_1x1/Conv2D',
                                                                                                               in_channels=320,
                                                                                                               out_channels=32,
                                                                                                               kernel_size=(
                                                                                                                   1,
                                                                                                                   1),
                                                                                                               stride=(
                                                                                                                   1,
                                                                                                                   1),
                                                                                                               groups=1,
                                                                                                               bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_10_Branch_0_Conv2d_1x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionResnetV2/InceptionResnetV2/Repeat/block35_10/Branch_0/Conv2d_1x1/BatchNorm/FusedBatchNorm',
            num_features=32, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_10_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionResnetV2/InceptionResnetV2/Repeat/block35_10/Branch_1/Conv2d_0a_1x1/BatchNorm/FusedBatchNorm',
            num_features=32, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_10_Branch_2_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionResnetV2/InceptionResnetV2/Repeat/block35_10/Branch_2/Conv2d_0a_1x1/BatchNorm/FusedBatchNorm',
            num_features=32, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_10_Branch_1_Conv2d_0b_3x3_Conv2D = self.__conv(2,
                                                                                                               name='InceptionResnetV2/InceptionResnetV2/Repeat/block35_10/Branch_1/Conv2d_0b_3x3/Conv2D',
                                                                                                               in_channels=32,
                                                                                                               out_channels=32,
                                                                                                               kernel_size=(
                                                                                                                   3,
                                                                                                                   3),
                                                                                                               stride=(
                                                                                                                   1,
                                                                                                                   1),
                                                                                                               groups=1,
                                                                                                               bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_10_Branch_2_Conv2d_0b_3x3_Conv2D = self.__conv(2,
                                                                                                               name='InceptionResnetV2/InceptionResnetV2/Repeat/block35_10/Branch_2/Conv2d_0b_3x3/Conv2D',
                                                                                                               in_channels=32,
                                                                                                               out_channels=48,
                                                                                                               kernel_size=(
                                                                                                                   3,
                                                                                                                   3),
                                                                                                               stride=(
                                                                                                                   1,
                                                                                                                   1),
                                                                                                               groups=1,
                                                                                                               bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_10_Branch_1_Conv2d_0b_3x3_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionResnetV2/InceptionResnetV2/Repeat/block35_10/Branch_1/Conv2d_0b_3x3/BatchNorm/FusedBatchNorm',
            num_features=32, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_10_Branch_2_Conv2d_0b_3x3_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionResnetV2/InceptionResnetV2/Repeat/block35_10/Branch_2/Conv2d_0b_3x3/BatchNorm/FusedBatchNorm',
            num_features=48, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_10_Branch_2_Conv2d_0c_3x3_Conv2D = self.__conv(2,
                                                                                                               name='InceptionResnetV2/InceptionResnetV2/Repeat/block35_10/Branch_2/Conv2d_0c_3x3/Conv2D',
                                                                                                               in_channels=48,
                                                                                                               out_channels=64,
                                                                                                               kernel_size=(
                                                                                                                   3,
                                                                                                                   3),
                                                                                                               stride=(
                                                                                                                   1,
                                                                                                                   1),
                                                                                                               groups=1,
                                                                                                               bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_10_Branch_2_Conv2d_0c_3x3_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionResnetV2/InceptionResnetV2/Repeat/block35_10/Branch_2/Conv2d_0c_3x3/BatchNorm/FusedBatchNorm',
            num_features=64, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_10_Conv2d_1x1_Conv2D = self.__conv(2,
                                                                                                   name='InceptionResnetV2/InceptionResnetV2/Repeat/block35_10/Conv2d_1x1/Conv2D',
                                                                                                   in_channels=128,
                                                                                                   out_channels=320,
                                                                                                   kernel_size=(1, 1),
                                                                                                   stride=(1, 1),
                                                                                                   groups=1, bias=True)
        self.InceptionResnetV2_InceptionResnetV2_Mixed_6a_Branch_0_Conv2d_1a_3x3_Conv2D = self.__conv(2,
                                                                                                      name='InceptionResnetV2/InceptionResnetV2/Mixed_6a/Branch_0/Conv2d_1a_3x3/Conv2D',
                                                                                                      in_channels=320,
                                                                                                      out_channels=384,
                                                                                                      kernel_size=(
                                                                                                          3, 3),
                                                                                                      stride=(2, 2),
                                                                                                      groups=1,
                                                                                                      bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Mixed_6a_Branch_1_Conv2d_0a_1x1_Conv2D = self.__conv(2,
                                                                                                      name='InceptionResnetV2/InceptionResnetV2/Mixed_6a/Branch_1/Conv2d_0a_1x1/Conv2D',
                                                                                                      in_channels=320,
                                                                                                      out_channels=256,
                                                                                                      kernel_size=(
                                                                                                          1, 1),
                                                                                                      stride=(1, 1),
                                                                                                      groups=1,
                                                                                                      bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Mixed_6a_Branch_0_Conv2d_1a_3x3_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionResnetV2/InceptionResnetV2/Mixed_6a/Branch_0/Conv2d_1a_3x3/BatchNorm/FusedBatchNorm',
            num_features=384, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Mixed_6a_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionResnetV2/InceptionResnetV2/Mixed_6a/Branch_1/Conv2d_0a_1x1/BatchNorm/FusedBatchNorm',
            num_features=256, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Mixed_6a_Branch_1_Conv2d_0b_3x3_Conv2D = self.__conv(2,
                                                                                                      name='InceptionResnetV2/InceptionResnetV2/Mixed_6a/Branch_1/Conv2d_0b_3x3/Conv2D',
                                                                                                      in_channels=256,
                                                                                                      out_channels=256,
                                                                                                      kernel_size=(
                                                                                                          3, 3),
                                                                                                      stride=(1, 1),
                                                                                                      groups=1,
                                                                                                      bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Mixed_6a_Branch_1_Conv2d_0b_3x3_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionResnetV2/InceptionResnetV2/Mixed_6a/Branch_1/Conv2d_0b_3x3/BatchNorm/FusedBatchNorm',
            num_features=256, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Mixed_6a_Branch_1_Conv2d_1a_3x3_Conv2D = self.__conv(2,
                                                                                                      name='InceptionResnetV2/InceptionResnetV2/Mixed_6a/Branch_1/Conv2d_1a_3x3/Conv2D',
                                                                                                      in_channels=256,
                                                                                                      out_channels=384,
                                                                                                      kernel_size=(
                                                                                                          3, 3),
                                                                                                      stride=(2, 2),
                                                                                                      groups=1,
                                                                                                      bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Mixed_6a_Branch_1_Conv2d_1a_3x3_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionResnetV2/InceptionResnetV2/Mixed_6a/Branch_1/Conv2d_1a_3x3/BatchNorm/FusedBatchNorm',
            num_features=384, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_1_Branch_0_Conv2d_1x1_Conv2D = self.__conv(2,
                                                                                                             name='InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_1/Branch_0/Conv2d_1x1/Conv2D',
                                                                                                             in_channels=1088,
                                                                                                             out_channels=192,
                                                                                                             kernel_size=(
                                                                                                                 1, 1),
                                                                                                             stride=(
                                                                                                                 1, 1),
                                                                                                             groups=1,
                                                                                                             bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_1_Branch_1_Conv2d_0a_1x1_Conv2D = self.__conv(2,
                                                                                                                name='InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_1/Branch_1/Conv2d_0a_1x1/Conv2D',
                                                                                                                in_channels=1088,
                                                                                                                out_channels=128,
                                                                                                                kernel_size=(
                                                                                                                    1,
                                                                                                                    1),
                                                                                                                stride=(
                                                                                                                    1,
                                                                                                                    1),
                                                                                                                groups=1,
                                                                                                                bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_1_Branch_0_Conv2d_1x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_1/Branch_0/Conv2d_1x1/BatchNorm/FusedBatchNorm',
            num_features=192, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_1_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_1/Branch_1/Conv2d_0a_1x1/BatchNorm/FusedBatchNorm',
            num_features=128, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_1_Branch_1_Conv2d_0b_1x7_Conv2D = self.__conv(2,
                                                                                                                name='InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_1/Branch_1/Conv2d_0b_1x7/Conv2D',
                                                                                                                in_channels=128,
                                                                                                                out_channels=160,
                                                                                                                kernel_size=(
                                                                                                                    1,
                                                                                                                    7),
                                                                                                                stride=(
                                                                                                                    1,
                                                                                                                    1),
                                                                                                                groups=1,
                                                                                                                bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_1_Branch_1_Conv2d_0b_1x7_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_1/Branch_1/Conv2d_0b_1x7/BatchNorm/FusedBatchNorm',
            num_features=160, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_1_Branch_1_Conv2d_0c_7x1_Conv2D = self.__conv(2,
                                                                                                                name='InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_1/Branch_1/Conv2d_0c_7x1/Conv2D',
                                                                                                                in_channels=160,
                                                                                                                out_channels=192,
                                                                                                                kernel_size=(
                                                                                                                    7,
                                                                                                                    1),
                                                                                                                stride=(
                                                                                                                    1,
                                                                                                                    1),
                                                                                                                groups=1,
                                                                                                                bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_1_Branch_1_Conv2d_0c_7x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_1/Branch_1/Conv2d_0c_7x1/BatchNorm/FusedBatchNorm',
            num_features=192, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_1_Conv2d_1x1_Conv2D = self.__conv(2,
                                                                                                    name='InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_1/Conv2d_1x1/Conv2D',
                                                                                                    in_channels=384,
                                                                                                    out_channels=1088,
                                                                                                    kernel_size=(1, 1),
                                                                                                    stride=(1, 1),
                                                                                                    groups=1, bias=True)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_2_Branch_0_Conv2d_1x1_Conv2D = self.__conv(2,
                                                                                                             name='InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_2/Branch_0/Conv2d_1x1/Conv2D',
                                                                                                             in_channels=1088,
                                                                                                             out_channels=192,
                                                                                                             kernel_size=(
                                                                                                                 1, 1),
                                                                                                             stride=(
                                                                                                                 1, 1),
                                                                                                             groups=1,
                                                                                                             bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_2_Branch_1_Conv2d_0a_1x1_Conv2D = self.__conv(2,
                                                                                                                name='InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_2/Branch_1/Conv2d_0a_1x1/Conv2D',
                                                                                                                in_channels=1088,
                                                                                                                out_channels=128,
                                                                                                                kernel_size=(
                                                                                                                    1,
                                                                                                                    1),
                                                                                                                stride=(
                                                                                                                    1,
                                                                                                                    1),
                                                                                                                groups=1,
                                                                                                                bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_2_Branch_0_Conv2d_1x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_2/Branch_0/Conv2d_1x1/BatchNorm/FusedBatchNorm',
            num_features=192, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_2_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_2/Branch_1/Conv2d_0a_1x1/BatchNorm/FusedBatchNorm',
            num_features=128, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_2_Branch_1_Conv2d_0b_1x7_Conv2D = self.__conv(2,
                                                                                                                name='InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_2/Branch_1/Conv2d_0b_1x7/Conv2D',
                                                                                                                in_channels=128,
                                                                                                                out_channels=160,
                                                                                                                kernel_size=(
                                                                                                                    1,
                                                                                                                    7),
                                                                                                                stride=(
                                                                                                                    1,
                                                                                                                    1),
                                                                                                                groups=1,
                                                                                                                bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_2_Branch_1_Conv2d_0b_1x7_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_2/Branch_1/Conv2d_0b_1x7/BatchNorm/FusedBatchNorm',
            num_features=160, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_2_Branch_1_Conv2d_0c_7x1_Conv2D = self.__conv(2,
                                                                                                                name='InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_2/Branch_1/Conv2d_0c_7x1/Conv2D',
                                                                                                                in_channels=160,
                                                                                                                out_channels=192,
                                                                                                                kernel_size=(
                                                                                                                    7,
                                                                                                                    1),
                                                                                                                stride=(
                                                                                                                    1,
                                                                                                                    1),
                                                                                                                groups=1,
                                                                                                                bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_2_Branch_1_Conv2d_0c_7x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_2/Branch_1/Conv2d_0c_7x1/BatchNorm/FusedBatchNorm',
            num_features=192, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_2_Conv2d_1x1_Conv2D = self.__conv(2,
                                                                                                    name='InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_2/Conv2d_1x1/Conv2D',
                                                                                                    in_channels=384,
                                                                                                    out_channels=1088,
                                                                                                    kernel_size=(1, 1),
                                                                                                    stride=(1, 1),
                                                                                                    groups=1, bias=True)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_3_Branch_0_Conv2d_1x1_Conv2D = self.__conv(2,
                                                                                                             name='InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_3/Branch_0/Conv2d_1x1/Conv2D',
                                                                                                             in_channels=1088,
                                                                                                             out_channels=192,
                                                                                                             kernel_size=(
                                                                                                                 1, 1),
                                                                                                             stride=(
                                                                                                                 1, 1),
                                                                                                             groups=1,
                                                                                                             bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_3_Branch_1_Conv2d_0a_1x1_Conv2D = self.__conv(2,
                                                                                                                name='InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_3/Branch_1/Conv2d_0a_1x1/Conv2D',
                                                                                                                in_channels=1088,
                                                                                                                out_channels=128,
                                                                                                                kernel_size=(
                                                                                                                    1,
                                                                                                                    1),
                                                                                                                stride=(
                                                                                                                    1,
                                                                                                                    1),
                                                                                                                groups=1,
                                                                                                                bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_3_Branch_0_Conv2d_1x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_3/Branch_0/Conv2d_1x1/BatchNorm/FusedBatchNorm',
            num_features=192, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_3_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_3/Branch_1/Conv2d_0a_1x1/BatchNorm/FusedBatchNorm',
            num_features=128, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_3_Branch_1_Conv2d_0b_1x7_Conv2D = self.__conv(2,
                                                                                                                name='InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_3/Branch_1/Conv2d_0b_1x7/Conv2D',
                                                                                                                in_channels=128,
                                                                                                                out_channels=160,
                                                                                                                kernel_size=(
                                                                                                                    1,
                                                                                                                    7),
                                                                                                                stride=(
                                                                                                                    1,
                                                                                                                    1),
                                                                                                                groups=1,
                                                                                                                bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_3_Branch_1_Conv2d_0b_1x7_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_3/Branch_1/Conv2d_0b_1x7/BatchNorm/FusedBatchNorm',
            num_features=160, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_3_Branch_1_Conv2d_0c_7x1_Conv2D = self.__conv(2,
                                                                                                                name='InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_3/Branch_1/Conv2d_0c_7x1/Conv2D',
                                                                                                                in_channels=160,
                                                                                                                out_channels=192,
                                                                                                                kernel_size=(
                                                                                                                    7,
                                                                                                                    1),
                                                                                                                stride=(
                                                                                                                    1,
                                                                                                                    1),
                                                                                                                groups=1,
                                                                                                                bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_3_Branch_1_Conv2d_0c_7x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_3/Branch_1/Conv2d_0c_7x1/BatchNorm/FusedBatchNorm',
            num_features=192, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_3_Conv2d_1x1_Conv2D = self.__conv(2,
                                                                                                    name='InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_3/Conv2d_1x1/Conv2D',
                                                                                                    in_channels=384,
                                                                                                    out_channels=1088,
                                                                                                    kernel_size=(1, 1),
                                                                                                    stride=(1, 1),
                                                                                                    groups=1, bias=True)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_4_Branch_0_Conv2d_1x1_Conv2D = self.__conv(2,
                                                                                                             name='InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_4/Branch_0/Conv2d_1x1/Conv2D',
                                                                                                             in_channels=1088,
                                                                                                             out_channels=192,
                                                                                                             kernel_size=(
                                                                                                                 1, 1),
                                                                                                             stride=(
                                                                                                                 1, 1),
                                                                                                             groups=1,
                                                                                                             bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_4_Branch_1_Conv2d_0a_1x1_Conv2D = self.__conv(2,
                                                                                                                name='InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_4/Branch_1/Conv2d_0a_1x1/Conv2D',
                                                                                                                in_channels=1088,
                                                                                                                out_channels=128,
                                                                                                                kernel_size=(
                                                                                                                    1,
                                                                                                                    1),
                                                                                                                stride=(
                                                                                                                    1,
                                                                                                                    1),
                                                                                                                groups=1,
                                                                                                                bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_4_Branch_0_Conv2d_1x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_4/Branch_0/Conv2d_1x1/BatchNorm/FusedBatchNorm',
            num_features=192, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_4_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_4/Branch_1/Conv2d_0a_1x1/BatchNorm/FusedBatchNorm',
            num_features=128, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_4_Branch_1_Conv2d_0b_1x7_Conv2D = self.__conv(2,
                                                                                                                name='InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_4/Branch_1/Conv2d_0b_1x7/Conv2D',
                                                                                                                in_channels=128,
                                                                                                                out_channels=160,
                                                                                                                kernel_size=(
                                                                                                                    1,
                                                                                                                    7),
                                                                                                                stride=(
                                                                                                                    1,
                                                                                                                    1),
                                                                                                                groups=1,
                                                                                                                bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_4_Branch_1_Conv2d_0b_1x7_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_4/Branch_1/Conv2d_0b_1x7/BatchNorm/FusedBatchNorm',
            num_features=160, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_4_Branch_1_Conv2d_0c_7x1_Conv2D = self.__conv(2,
                                                                                                                name='InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_4/Branch_1/Conv2d_0c_7x1/Conv2D',
                                                                                                                in_channels=160,
                                                                                                                out_channels=192,
                                                                                                                kernel_size=(
                                                                                                                    7,
                                                                                                                    1),
                                                                                                                stride=(
                                                                                                                    1,
                                                                                                                    1),
                                                                                                                groups=1,
                                                                                                                bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_4_Branch_1_Conv2d_0c_7x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_4/Branch_1/Conv2d_0c_7x1/BatchNorm/FusedBatchNorm',
            num_features=192, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_4_Conv2d_1x1_Conv2D = self.__conv(2,
                                                                                                    name='InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_4/Conv2d_1x1/Conv2D',
                                                                                                    in_channels=384,
                                                                                                    out_channels=1088,
                                                                                                    kernel_size=(1, 1),
                                                                                                    stride=(1, 1),
                                                                                                    groups=1, bias=True)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_5_Branch_0_Conv2d_1x1_Conv2D = self.__conv(2,
                                                                                                             name='InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_5/Branch_0/Conv2d_1x1/Conv2D',
                                                                                                             in_channels=1088,
                                                                                                             out_channels=192,
                                                                                                             kernel_size=(
                                                                                                                 1, 1),
                                                                                                             stride=(
                                                                                                                 1, 1),
                                                                                                             groups=1,
                                                                                                             bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_5_Branch_1_Conv2d_0a_1x1_Conv2D = self.__conv(2,
                                                                                                                name='InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_5/Branch_1/Conv2d_0a_1x1/Conv2D',
                                                                                                                in_channels=1088,
                                                                                                                out_channels=128,
                                                                                                                kernel_size=(
                                                                                                                    1,
                                                                                                                    1),
                                                                                                                stride=(
                                                                                                                    1,
                                                                                                                    1),
                                                                                                                groups=1,
                                                                                                                bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_5_Branch_0_Conv2d_1x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_5/Branch_0/Conv2d_1x1/BatchNorm/FusedBatchNorm',
            num_features=192, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_5_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_5/Branch_1/Conv2d_0a_1x1/BatchNorm/FusedBatchNorm',
            num_features=128, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_5_Branch_1_Conv2d_0b_1x7_Conv2D = self.__conv(2,
                                                                                                                name='InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_5/Branch_1/Conv2d_0b_1x7/Conv2D',
                                                                                                                in_channels=128,
                                                                                                                out_channels=160,
                                                                                                                kernel_size=(
                                                                                                                    1,
                                                                                                                    7),
                                                                                                                stride=(
                                                                                                                    1,
                                                                                                                    1),
                                                                                                                groups=1,
                                                                                                                bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_5_Branch_1_Conv2d_0b_1x7_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_5/Branch_1/Conv2d_0b_1x7/BatchNorm/FusedBatchNorm',
            num_features=160, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_5_Branch_1_Conv2d_0c_7x1_Conv2D = self.__conv(2,
                                                                                                                name='InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_5/Branch_1/Conv2d_0c_7x1/Conv2D',
                                                                                                                in_channels=160,
                                                                                                                out_channels=192,
                                                                                                                kernel_size=(
                                                                                                                    7,
                                                                                                                    1),
                                                                                                                stride=(
                                                                                                                    1,
                                                                                                                    1),
                                                                                                                groups=1,
                                                                                                                bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_5_Branch_1_Conv2d_0c_7x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_5/Branch_1/Conv2d_0c_7x1/BatchNorm/FusedBatchNorm',
            num_features=192, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_5_Conv2d_1x1_Conv2D = self.__conv(2,
                                                                                                    name='InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_5/Conv2d_1x1/Conv2D',
                                                                                                    in_channels=384,
                                                                                                    out_channels=1088,
                                                                                                    kernel_size=(1, 1),
                                                                                                    stride=(1, 1),
                                                                                                    groups=1, bias=True)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_6_Branch_0_Conv2d_1x1_Conv2D = self.__conv(2,
                                                                                                             name='InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_6/Branch_0/Conv2d_1x1/Conv2D',
                                                                                                             in_channels=1088,
                                                                                                             out_channels=192,
                                                                                                             kernel_size=(
                                                                                                                 1, 1),
                                                                                                             stride=(
                                                                                                                 1, 1),
                                                                                                             groups=1,
                                                                                                             bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_6_Branch_1_Conv2d_0a_1x1_Conv2D = self.__conv(2,
                                                                                                                name='InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_6/Branch_1/Conv2d_0a_1x1/Conv2D',
                                                                                                                in_channels=1088,
                                                                                                                out_channels=128,
                                                                                                                kernel_size=(
                                                                                                                    1,
                                                                                                                    1),
                                                                                                                stride=(
                                                                                                                    1,
                                                                                                                    1),
                                                                                                                groups=1,
                                                                                                                bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_6_Branch_0_Conv2d_1x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_6/Branch_0/Conv2d_1x1/BatchNorm/FusedBatchNorm',
            num_features=192, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_6_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_6/Branch_1/Conv2d_0a_1x1/BatchNorm/FusedBatchNorm',
            num_features=128, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_6_Branch_1_Conv2d_0b_1x7_Conv2D = self.__conv(2,
                                                                                                                name='InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_6/Branch_1/Conv2d_0b_1x7/Conv2D',
                                                                                                                in_channels=128,
                                                                                                                out_channels=160,
                                                                                                                kernel_size=(
                                                                                                                    1,
                                                                                                                    7),
                                                                                                                stride=(
                                                                                                                    1,
                                                                                                                    1),
                                                                                                                groups=1,
                                                                                                                bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_6_Branch_1_Conv2d_0b_1x7_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_6/Branch_1/Conv2d_0b_1x7/BatchNorm/FusedBatchNorm',
            num_features=160, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_6_Branch_1_Conv2d_0c_7x1_Conv2D = self.__conv(2,
                                                                                                                name='InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_6/Branch_1/Conv2d_0c_7x1/Conv2D',
                                                                                                                in_channels=160,
                                                                                                                out_channels=192,
                                                                                                                kernel_size=(
                                                                                                                    7,
                                                                                                                    1),
                                                                                                                stride=(
                                                                                                                    1,
                                                                                                                    1),
                                                                                                                groups=1,
                                                                                                                bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_6_Branch_1_Conv2d_0c_7x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_6/Branch_1/Conv2d_0c_7x1/BatchNorm/FusedBatchNorm',
            num_features=192, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_6_Conv2d_1x1_Conv2D = self.__conv(2,
                                                                                                    name='InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_6/Conv2d_1x1/Conv2D',
                                                                                                    in_channels=384,
                                                                                                    out_channels=1088,
                                                                                                    kernel_size=(1, 1),
                                                                                                    stride=(1, 1),
                                                                                                    groups=1, bias=True)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_7_Branch_0_Conv2d_1x1_Conv2D = self.__conv(2,
                                                                                                             name='InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_7/Branch_0/Conv2d_1x1/Conv2D',
                                                                                                             in_channels=1088,
                                                                                                             out_channels=192,
                                                                                                             kernel_size=(
                                                                                                                 1, 1),
                                                                                                             stride=(
                                                                                                                 1, 1),
                                                                                                             groups=1,
                                                                                                             bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_7_Branch_1_Conv2d_0a_1x1_Conv2D = self.__conv(2,
                                                                                                                name='InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_7/Branch_1/Conv2d_0a_1x1/Conv2D',
                                                                                                                in_channels=1088,
                                                                                                                out_channels=128,
                                                                                                                kernel_size=(
                                                                                                                    1,
                                                                                                                    1),
                                                                                                                stride=(
                                                                                                                    1,
                                                                                                                    1),
                                                                                                                groups=1,
                                                                                                                bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_7_Branch_0_Conv2d_1x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_7/Branch_0/Conv2d_1x1/BatchNorm/FusedBatchNorm',
            num_features=192, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_7_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_7/Branch_1/Conv2d_0a_1x1/BatchNorm/FusedBatchNorm',
            num_features=128, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_7_Branch_1_Conv2d_0b_1x7_Conv2D = self.__conv(2,
                                                                                                                name='InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_7/Branch_1/Conv2d_0b_1x7/Conv2D',
                                                                                                                in_channels=128,
                                                                                                                out_channels=160,
                                                                                                                kernel_size=(
                                                                                                                    1,
                                                                                                                    7),
                                                                                                                stride=(
                                                                                                                    1,
                                                                                                                    1),
                                                                                                                groups=1,
                                                                                                                bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_7_Branch_1_Conv2d_0b_1x7_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_7/Branch_1/Conv2d_0b_1x7/BatchNorm/FusedBatchNorm',
            num_features=160, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_7_Branch_1_Conv2d_0c_7x1_Conv2D = self.__conv(2,
                                                                                                                name='InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_7/Branch_1/Conv2d_0c_7x1/Conv2D',
                                                                                                                in_channels=160,
                                                                                                                out_channels=192,
                                                                                                                kernel_size=(
                                                                                                                    7,
                                                                                                                    1),
                                                                                                                stride=(
                                                                                                                    1,
                                                                                                                    1),
                                                                                                                groups=1,
                                                                                                                bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_7_Branch_1_Conv2d_0c_7x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_7/Branch_1/Conv2d_0c_7x1/BatchNorm/FusedBatchNorm',
            num_features=192, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_7_Conv2d_1x1_Conv2D = self.__conv(2,
                                                                                                    name='InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_7/Conv2d_1x1/Conv2D',
                                                                                                    in_channels=384,
                                                                                                    out_channels=1088,
                                                                                                    kernel_size=(1, 1),
                                                                                                    stride=(1, 1),
                                                                                                    groups=1, bias=True)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_8_Branch_0_Conv2d_1x1_Conv2D = self.__conv(2,
                                                                                                             name='InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_8/Branch_0/Conv2d_1x1/Conv2D',
                                                                                                             in_channels=1088,
                                                                                                             out_channels=192,
                                                                                                             kernel_size=(
                                                                                                                 1, 1),
                                                                                                             stride=(
                                                                                                                 1, 1),
                                                                                                             groups=1,
                                                                                                             bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_8_Branch_1_Conv2d_0a_1x1_Conv2D = self.__conv(2,
                                                                                                                name='InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_8/Branch_1/Conv2d_0a_1x1/Conv2D',
                                                                                                                in_channels=1088,
                                                                                                                out_channels=128,
                                                                                                                kernel_size=(
                                                                                                                    1,
                                                                                                                    1),
                                                                                                                stride=(
                                                                                                                    1,
                                                                                                                    1),
                                                                                                                groups=1,
                                                                                                                bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_8_Branch_0_Conv2d_1x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_8/Branch_0/Conv2d_1x1/BatchNorm/FusedBatchNorm',
            num_features=192, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_8_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_8/Branch_1/Conv2d_0a_1x1/BatchNorm/FusedBatchNorm',
            num_features=128, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_8_Branch_1_Conv2d_0b_1x7_Conv2D = self.__conv(2,
                                                                                                                name='InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_8/Branch_1/Conv2d_0b_1x7/Conv2D',
                                                                                                                in_channels=128,
                                                                                                                out_channels=160,
                                                                                                                kernel_size=(
                                                                                                                    1,
                                                                                                                    7),
                                                                                                                stride=(
                                                                                                                    1,
                                                                                                                    1),
                                                                                                                groups=1,
                                                                                                                bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_8_Branch_1_Conv2d_0b_1x7_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_8/Branch_1/Conv2d_0b_1x7/BatchNorm/FusedBatchNorm',
            num_features=160, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_8_Branch_1_Conv2d_0c_7x1_Conv2D = self.__conv(2,
                                                                                                                name='InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_8/Branch_1/Conv2d_0c_7x1/Conv2D',
                                                                                                                in_channels=160,
                                                                                                                out_channels=192,
                                                                                                                kernel_size=(
                                                                                                                    7,
                                                                                                                    1),
                                                                                                                stride=(
                                                                                                                    1,
                                                                                                                    1),
                                                                                                                groups=1,
                                                                                                                bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_8_Branch_1_Conv2d_0c_7x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_8/Branch_1/Conv2d_0c_7x1/BatchNorm/FusedBatchNorm',
            num_features=192, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_8_Conv2d_1x1_Conv2D = self.__conv(2,
                                                                                                    name='InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_8/Conv2d_1x1/Conv2D',
                                                                                                    in_channels=384,
                                                                                                    out_channels=1088,
                                                                                                    kernel_size=(1, 1),
                                                                                                    stride=(1, 1),
                                                                                                    groups=1, bias=True)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_9_Branch_0_Conv2d_1x1_Conv2D = self.__conv(2,
                                                                                                             name='InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_9/Branch_0/Conv2d_1x1/Conv2D',
                                                                                                             in_channels=1088,
                                                                                                             out_channels=192,
                                                                                                             kernel_size=(
                                                                                                                 1, 1),
                                                                                                             stride=(
                                                                                                                 1, 1),
                                                                                                             groups=1,
                                                                                                             bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_9_Branch_1_Conv2d_0a_1x1_Conv2D = self.__conv(2,
                                                                                                                name='InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_9/Branch_1/Conv2d_0a_1x1/Conv2D',
                                                                                                                in_channels=1088,
                                                                                                                out_channels=128,
                                                                                                                kernel_size=(
                                                                                                                    1,
                                                                                                                    1),
                                                                                                                stride=(
                                                                                                                    1,
                                                                                                                    1),
                                                                                                                groups=1,
                                                                                                                bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_9_Branch_0_Conv2d_1x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_9/Branch_0/Conv2d_1x1/BatchNorm/FusedBatchNorm',
            num_features=192, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_9_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_9/Branch_1/Conv2d_0a_1x1/BatchNorm/FusedBatchNorm',
            num_features=128, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_9_Branch_1_Conv2d_0b_1x7_Conv2D = self.__conv(2,
                                                                                                                name='InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_9/Branch_1/Conv2d_0b_1x7/Conv2D',
                                                                                                                in_channels=128,
                                                                                                                out_channels=160,
                                                                                                                kernel_size=(
                                                                                                                    1,
                                                                                                                    7),
                                                                                                                stride=(
                                                                                                                    1,
                                                                                                                    1),
                                                                                                                groups=1,
                                                                                                                bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_9_Branch_1_Conv2d_0b_1x7_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_9/Branch_1/Conv2d_0b_1x7/BatchNorm/FusedBatchNorm',
            num_features=160, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_9_Branch_1_Conv2d_0c_7x1_Conv2D = self.__conv(2,
                                                                                                                name='InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_9/Branch_1/Conv2d_0c_7x1/Conv2D',
                                                                                                                in_channels=160,
                                                                                                                out_channels=192,
                                                                                                                kernel_size=(
                                                                                                                    7,
                                                                                                                    1),
                                                                                                                stride=(
                                                                                                                    1,
                                                                                                                    1),
                                                                                                                groups=1,
                                                                                                                bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_9_Branch_1_Conv2d_0c_7x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_9/Branch_1/Conv2d_0c_7x1/BatchNorm/FusedBatchNorm',
            num_features=192, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_9_Conv2d_1x1_Conv2D = self.__conv(2,
                                                                                                    name='InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_9/Conv2d_1x1/Conv2D',
                                                                                                    in_channels=384,
                                                                                                    out_channels=1088,
                                                                                                    kernel_size=(1, 1),
                                                                                                    stride=(1, 1),
                                                                                                    groups=1, bias=True)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_10_Branch_0_Conv2d_1x1_Conv2D = self.__conv(2,
                                                                                                              name='InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_10/Branch_0/Conv2d_1x1/Conv2D',
                                                                                                              in_channels=1088,
                                                                                                              out_channels=192,
                                                                                                              kernel_size=(
                                                                                                                  1, 1),
                                                                                                              stride=(
                                                                                                                  1, 1),
                                                                                                              groups=1,
                                                                                                              bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_10_Branch_1_Conv2d_0a_1x1_Conv2D = self.__conv(2,
                                                                                                                 name='InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_10/Branch_1/Conv2d_0a_1x1/Conv2D',
                                                                                                                 in_channels=1088,
                                                                                                                 out_channels=128,
                                                                                                                 kernel_size=(
                                                                                                                     1,
                                                                                                                     1),
                                                                                                                 stride=(
                                                                                                                     1,
                                                                                                                     1),
                                                                                                                 groups=1,
                                                                                                                 bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_10_Branch_0_Conv2d_1x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_10/Branch_0/Conv2d_1x1/BatchNorm/FusedBatchNorm',
            num_features=192, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_10_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2,
            'InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_10/Branch_1/Conv2d_0a_1x1/BatchNorm/FusedBatchNorm',
            num_features=128, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_10_Branch_1_Conv2d_0b_1x7_Conv2D = self.__conv(2,
                                                                                                                 name='InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_10/Branch_1/Conv2d_0b_1x7/Conv2D',
                                                                                                                 in_channels=128,
                                                                                                                 out_channels=160,
                                                                                                                 kernel_size=(
                                                                                                                     1,
                                                                                                                     7),
                                                                                                                 stride=(
                                                                                                                     1,
                                                                                                                     1),
                                                                                                                 groups=1,
                                                                                                                 bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_10_Branch_1_Conv2d_0b_1x7_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2,
            'InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_10/Branch_1/Conv2d_0b_1x7/BatchNorm/FusedBatchNorm',
            num_features=160, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_10_Branch_1_Conv2d_0c_7x1_Conv2D = self.__conv(2,
                                                                                                                 name='InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_10/Branch_1/Conv2d_0c_7x1/Conv2D',
                                                                                                                 in_channels=160,
                                                                                                                 out_channels=192,
                                                                                                                 kernel_size=(
                                                                                                                     7,
                                                                                                                     1),
                                                                                                                 stride=(
                                                                                                                     1,
                                                                                                                     1),
                                                                                                                 groups=1,
                                                                                                                 bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_10_Branch_1_Conv2d_0c_7x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2,
            'InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_10/Branch_1/Conv2d_0c_7x1/BatchNorm/FusedBatchNorm',
            num_features=192, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_10_Conv2d_1x1_Conv2D = self.__conv(2,
                                                                                                     name='InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_10/Conv2d_1x1/Conv2D',
                                                                                                     in_channels=384,
                                                                                                     out_channels=1088,
                                                                                                     kernel_size=(1, 1),
                                                                                                     stride=(1, 1),
                                                                                                     groups=1,
                                                                                                     bias=True)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_11_Branch_0_Conv2d_1x1_Conv2D = self.__conv(2,
                                                                                                              name='InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_11/Branch_0/Conv2d_1x1/Conv2D',
                                                                                                              in_channels=1088,
                                                                                                              out_channels=192,
                                                                                                              kernel_size=(
                                                                                                                  1, 1),
                                                                                                              stride=(
                                                                                                                  1, 1),
                                                                                                              groups=1,
                                                                                                              bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_11_Branch_1_Conv2d_0a_1x1_Conv2D = self.__conv(2,
                                                                                                                 name='InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_11/Branch_1/Conv2d_0a_1x1/Conv2D',
                                                                                                                 in_channels=1088,
                                                                                                                 out_channels=128,
                                                                                                                 kernel_size=(
                                                                                                                     1,
                                                                                                                     1),
                                                                                                                 stride=(
                                                                                                                     1,
                                                                                                                     1),
                                                                                                                 groups=1,
                                                                                                                 bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_11_Branch_0_Conv2d_1x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_11/Branch_0/Conv2d_1x1/BatchNorm/FusedBatchNorm',
            num_features=192, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_11_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2,
            'InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_11/Branch_1/Conv2d_0a_1x1/BatchNorm/FusedBatchNorm',
            num_features=128, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_11_Branch_1_Conv2d_0b_1x7_Conv2D = self.__conv(2,
                                                                                                                 name='InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_11/Branch_1/Conv2d_0b_1x7/Conv2D',
                                                                                                                 in_channels=128,
                                                                                                                 out_channels=160,
                                                                                                                 kernel_size=(
                                                                                                                     1,
                                                                                                                     7),
                                                                                                                 stride=(
                                                                                                                     1,
                                                                                                                     1),
                                                                                                                 groups=1,
                                                                                                                 bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_11_Branch_1_Conv2d_0b_1x7_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2,
            'InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_11/Branch_1/Conv2d_0b_1x7/BatchNorm/FusedBatchNorm',
            num_features=160, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_11_Branch_1_Conv2d_0c_7x1_Conv2D = self.__conv(2,
                                                                                                                 name='InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_11/Branch_1/Conv2d_0c_7x1/Conv2D',
                                                                                                                 in_channels=160,
                                                                                                                 out_channels=192,
                                                                                                                 kernel_size=(
                                                                                                                     7,
                                                                                                                     1),
                                                                                                                 stride=(
                                                                                                                     1,
                                                                                                                     1),
                                                                                                                 groups=1,
                                                                                                                 bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_11_Branch_1_Conv2d_0c_7x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2,
            'InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_11/Branch_1/Conv2d_0c_7x1/BatchNorm/FusedBatchNorm',
            num_features=192, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_11_Conv2d_1x1_Conv2D = self.__conv(2,
                                                                                                     name='InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_11/Conv2d_1x1/Conv2D',
                                                                                                     in_channels=384,
                                                                                                     out_channels=1088,
                                                                                                     kernel_size=(1, 1),
                                                                                                     stride=(1, 1),
                                                                                                     groups=1,
                                                                                                     bias=True)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_12_Branch_0_Conv2d_1x1_Conv2D = self.__conv(2,
                                                                                                              name='InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_12/Branch_0/Conv2d_1x1/Conv2D',
                                                                                                              in_channels=1088,
                                                                                                              out_channels=192,
                                                                                                              kernel_size=(
                                                                                                                  1, 1),
                                                                                                              stride=(
                                                                                                                  1, 1),
                                                                                                              groups=1,
                                                                                                              bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_12_Branch_1_Conv2d_0a_1x1_Conv2D = self.__conv(2,
                                                                                                                 name='InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_12/Branch_1/Conv2d_0a_1x1/Conv2D',
                                                                                                                 in_channels=1088,
                                                                                                                 out_channels=128,
                                                                                                                 kernel_size=(
                                                                                                                     1,
                                                                                                                     1),
                                                                                                                 stride=(
                                                                                                                     1,
                                                                                                                     1),
                                                                                                                 groups=1,
                                                                                                                 bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_12_Branch_0_Conv2d_1x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_12/Branch_0/Conv2d_1x1/BatchNorm/FusedBatchNorm',
            num_features=192, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_12_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2,
            'InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_12/Branch_1/Conv2d_0a_1x1/BatchNorm/FusedBatchNorm',
            num_features=128, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_12_Branch_1_Conv2d_0b_1x7_Conv2D = self.__conv(2,
                                                                                                                 name='InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_12/Branch_1/Conv2d_0b_1x7/Conv2D',
                                                                                                                 in_channels=128,
                                                                                                                 out_channels=160,
                                                                                                                 kernel_size=(
                                                                                                                     1,
                                                                                                                     7),
                                                                                                                 stride=(
                                                                                                                     1,
                                                                                                                     1),
                                                                                                                 groups=1,
                                                                                                                 bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_12_Branch_1_Conv2d_0b_1x7_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2,
            'InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_12/Branch_1/Conv2d_0b_1x7/BatchNorm/FusedBatchNorm',
            num_features=160, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_12_Branch_1_Conv2d_0c_7x1_Conv2D = self.__conv(2,
                                                                                                                 name='InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_12/Branch_1/Conv2d_0c_7x1/Conv2D',
                                                                                                                 in_channels=160,
                                                                                                                 out_channels=192,
                                                                                                                 kernel_size=(
                                                                                                                     7,
                                                                                                                     1),
                                                                                                                 stride=(
                                                                                                                     1,
                                                                                                                     1),
                                                                                                                 groups=1,
                                                                                                                 bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_12_Branch_1_Conv2d_0c_7x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2,
            'InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_12/Branch_1/Conv2d_0c_7x1/BatchNorm/FusedBatchNorm',
            num_features=192, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_12_Conv2d_1x1_Conv2D = self.__conv(2,
                                                                                                     name='InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_12/Conv2d_1x1/Conv2D',
                                                                                                     in_channels=384,
                                                                                                     out_channels=1088,
                                                                                                     kernel_size=(1, 1),
                                                                                                     stride=(1, 1),
                                                                                                     groups=1,
                                                                                                     bias=True)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_13_Branch_0_Conv2d_1x1_Conv2D = self.__conv(2,
                                                                                                              name='InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_13/Branch_0/Conv2d_1x1/Conv2D',
                                                                                                              in_channels=1088,
                                                                                                              out_channels=192,
                                                                                                              kernel_size=(
                                                                                                                  1, 1),
                                                                                                              stride=(
                                                                                                                  1, 1),
                                                                                                              groups=1,
                                                                                                              bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_13_Branch_1_Conv2d_0a_1x1_Conv2D = self.__conv(2,
                                                                                                                 name='InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_13/Branch_1/Conv2d_0a_1x1/Conv2D',
                                                                                                                 in_channels=1088,
                                                                                                                 out_channels=128,
                                                                                                                 kernel_size=(
                                                                                                                     1,
                                                                                                                     1),
                                                                                                                 stride=(
                                                                                                                     1,
                                                                                                                     1),
                                                                                                                 groups=1,
                                                                                                                 bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_13_Branch_0_Conv2d_1x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_13/Branch_0/Conv2d_1x1/BatchNorm/FusedBatchNorm',
            num_features=192, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_13_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2,
            'InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_13/Branch_1/Conv2d_0a_1x1/BatchNorm/FusedBatchNorm',
            num_features=128, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_13_Branch_1_Conv2d_0b_1x7_Conv2D = self.__conv(2,
                                                                                                                 name='InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_13/Branch_1/Conv2d_0b_1x7/Conv2D',
                                                                                                                 in_channels=128,
                                                                                                                 out_channels=160,
                                                                                                                 kernel_size=(
                                                                                                                     1,
                                                                                                                     7),
                                                                                                                 stride=(
                                                                                                                     1,
                                                                                                                     1),
                                                                                                                 groups=1,
                                                                                                                 bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_13_Branch_1_Conv2d_0b_1x7_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2,
            'InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_13/Branch_1/Conv2d_0b_1x7/BatchNorm/FusedBatchNorm',
            num_features=160, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_13_Branch_1_Conv2d_0c_7x1_Conv2D = self.__conv(2,
                                                                                                                 name='InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_13/Branch_1/Conv2d_0c_7x1/Conv2D',
                                                                                                                 in_channels=160,
                                                                                                                 out_channels=192,
                                                                                                                 kernel_size=(
                                                                                                                     7,
                                                                                                                     1),
                                                                                                                 stride=(
                                                                                                                     1,
                                                                                                                     1),
                                                                                                                 groups=1,
                                                                                                                 bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_13_Branch_1_Conv2d_0c_7x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2,
            'InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_13/Branch_1/Conv2d_0c_7x1/BatchNorm/FusedBatchNorm',
            num_features=192, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_13_Conv2d_1x1_Conv2D = self.__conv(2,
                                                                                                     name='InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_13/Conv2d_1x1/Conv2D',
                                                                                                     in_channels=384,
                                                                                                     out_channels=1088,
                                                                                                     kernel_size=(1, 1),
                                                                                                     stride=(1, 1),
                                                                                                     groups=1,
                                                                                                     bias=True)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_14_Branch_0_Conv2d_1x1_Conv2D = self.__conv(2,
                                                                                                              name='InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_14/Branch_0/Conv2d_1x1/Conv2D',
                                                                                                              in_channels=1088,
                                                                                                              out_channels=192,
                                                                                                              kernel_size=(
                                                                                                                  1, 1),
                                                                                                              stride=(
                                                                                                                  1, 1),
                                                                                                              groups=1,
                                                                                                              bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_14_Branch_1_Conv2d_0a_1x1_Conv2D = self.__conv(2,
                                                                                                                 name='InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_14/Branch_1/Conv2d_0a_1x1/Conv2D',
                                                                                                                 in_channels=1088,
                                                                                                                 out_channels=128,
                                                                                                                 kernel_size=(
                                                                                                                     1,
                                                                                                                     1),
                                                                                                                 stride=(
                                                                                                                     1,
                                                                                                                     1),
                                                                                                                 groups=1,
                                                                                                                 bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_14_Branch_0_Conv2d_1x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_14/Branch_0/Conv2d_1x1/BatchNorm/FusedBatchNorm',
            num_features=192, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_14_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2,
            'InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_14/Branch_1/Conv2d_0a_1x1/BatchNorm/FusedBatchNorm',
            num_features=128, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_14_Branch_1_Conv2d_0b_1x7_Conv2D = self.__conv(2,
                                                                                                                 name='InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_14/Branch_1/Conv2d_0b_1x7/Conv2D',
                                                                                                                 in_channels=128,
                                                                                                                 out_channels=160,
                                                                                                                 kernel_size=(
                                                                                                                     1,
                                                                                                                     7),
                                                                                                                 stride=(
                                                                                                                     1,
                                                                                                                     1),
                                                                                                                 groups=1,
                                                                                                                 bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_14_Branch_1_Conv2d_0b_1x7_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2,
            'InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_14/Branch_1/Conv2d_0b_1x7/BatchNorm/FusedBatchNorm',
            num_features=160, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_14_Branch_1_Conv2d_0c_7x1_Conv2D = self.__conv(2,
                                                                                                                 name='InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_14/Branch_1/Conv2d_0c_7x1/Conv2D',
                                                                                                                 in_channels=160,
                                                                                                                 out_channels=192,
                                                                                                                 kernel_size=(
                                                                                                                     7,
                                                                                                                     1),
                                                                                                                 stride=(
                                                                                                                     1,
                                                                                                                     1),
                                                                                                                 groups=1,
                                                                                                                 bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_14_Branch_1_Conv2d_0c_7x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2,
            'InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_14/Branch_1/Conv2d_0c_7x1/BatchNorm/FusedBatchNorm',
            num_features=192, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_14_Conv2d_1x1_Conv2D = self.__conv(2,
                                                                                                     name='InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_14/Conv2d_1x1/Conv2D',
                                                                                                     in_channels=384,
                                                                                                     out_channels=1088,
                                                                                                     kernel_size=(1, 1),
                                                                                                     stride=(1, 1),
                                                                                                     groups=1,
                                                                                                     bias=True)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_15_Branch_0_Conv2d_1x1_Conv2D = self.__conv(2,
                                                                                                              name='InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_15/Branch_0/Conv2d_1x1/Conv2D',
                                                                                                              in_channels=1088,
                                                                                                              out_channels=192,
                                                                                                              kernel_size=(
                                                                                                                  1, 1),
                                                                                                              stride=(
                                                                                                                  1, 1),
                                                                                                              groups=1,
                                                                                                              bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_15_Branch_1_Conv2d_0a_1x1_Conv2D = self.__conv(2,
                                                                                                                 name='InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_15/Branch_1/Conv2d_0a_1x1/Conv2D',
                                                                                                                 in_channels=1088,
                                                                                                                 out_channels=128,
                                                                                                                 kernel_size=(
                                                                                                                     1,
                                                                                                                     1),
                                                                                                                 stride=(
                                                                                                                     1,
                                                                                                                     1),
                                                                                                                 groups=1,
                                                                                                                 bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_15_Branch_0_Conv2d_1x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_15/Branch_0/Conv2d_1x1/BatchNorm/FusedBatchNorm',
            num_features=192, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_15_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2,
            'InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_15/Branch_1/Conv2d_0a_1x1/BatchNorm/FusedBatchNorm',
            num_features=128, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_15_Branch_1_Conv2d_0b_1x7_Conv2D = self.__conv(2,
                                                                                                                 name='InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_15/Branch_1/Conv2d_0b_1x7/Conv2D',
                                                                                                                 in_channels=128,
                                                                                                                 out_channels=160,
                                                                                                                 kernel_size=(
                                                                                                                     1,
                                                                                                                     7),
                                                                                                                 stride=(
                                                                                                                     1,
                                                                                                                     1),
                                                                                                                 groups=1,
                                                                                                                 bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_15_Branch_1_Conv2d_0b_1x7_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2,
            'InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_15/Branch_1/Conv2d_0b_1x7/BatchNorm/FusedBatchNorm',
            num_features=160, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_15_Branch_1_Conv2d_0c_7x1_Conv2D = self.__conv(2,
                                                                                                                 name='InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_15/Branch_1/Conv2d_0c_7x1/Conv2D',
                                                                                                                 in_channels=160,
                                                                                                                 out_channels=192,
                                                                                                                 kernel_size=(
                                                                                                                     7,
                                                                                                                     1),
                                                                                                                 stride=(
                                                                                                                     1,
                                                                                                                     1),
                                                                                                                 groups=1,
                                                                                                                 bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_15_Branch_1_Conv2d_0c_7x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2,
            'InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_15/Branch_1/Conv2d_0c_7x1/BatchNorm/FusedBatchNorm',
            num_features=192, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_15_Conv2d_1x1_Conv2D = self.__conv(2,
                                                                                                     name='InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_15/Conv2d_1x1/Conv2D',
                                                                                                     in_channels=384,
                                                                                                     out_channels=1088,
                                                                                                     kernel_size=(1, 1),
                                                                                                     stride=(1, 1),
                                                                                                     groups=1,
                                                                                                     bias=True)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_16_Branch_0_Conv2d_1x1_Conv2D = self.__conv(2,
                                                                                                              name='InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_16/Branch_0/Conv2d_1x1/Conv2D',
                                                                                                              in_channels=1088,
                                                                                                              out_channels=192,
                                                                                                              kernel_size=(
                                                                                                                  1, 1),
                                                                                                              stride=(
                                                                                                                  1, 1),
                                                                                                              groups=1,
                                                                                                              bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_16_Branch_1_Conv2d_0a_1x1_Conv2D = self.__conv(2,
                                                                                                                 name='InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_16/Branch_1/Conv2d_0a_1x1/Conv2D',
                                                                                                                 in_channels=1088,
                                                                                                                 out_channels=128,
                                                                                                                 kernel_size=(
                                                                                                                     1,
                                                                                                                     1),
                                                                                                                 stride=(
                                                                                                                     1,
                                                                                                                     1),
                                                                                                                 groups=1,
                                                                                                                 bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_16_Branch_0_Conv2d_1x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_16/Branch_0/Conv2d_1x1/BatchNorm/FusedBatchNorm',
            num_features=192, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_16_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2,
            'InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_16/Branch_1/Conv2d_0a_1x1/BatchNorm/FusedBatchNorm',
            num_features=128, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_16_Branch_1_Conv2d_0b_1x7_Conv2D = self.__conv(2,
                                                                                                                 name='InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_16/Branch_1/Conv2d_0b_1x7/Conv2D',
                                                                                                                 in_channels=128,
                                                                                                                 out_channels=160,
                                                                                                                 kernel_size=(
                                                                                                                     1,
                                                                                                                     7),
                                                                                                                 stride=(
                                                                                                                     1,
                                                                                                                     1),
                                                                                                                 groups=1,
                                                                                                                 bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_16_Branch_1_Conv2d_0b_1x7_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2,
            'InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_16/Branch_1/Conv2d_0b_1x7/BatchNorm/FusedBatchNorm',
            num_features=160, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_16_Branch_1_Conv2d_0c_7x1_Conv2D = self.__conv(2,
                                                                                                                 name='InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_16/Branch_1/Conv2d_0c_7x1/Conv2D',
                                                                                                                 in_channels=160,
                                                                                                                 out_channels=192,
                                                                                                                 kernel_size=(
                                                                                                                     7,
                                                                                                                     1),
                                                                                                                 stride=(
                                                                                                                     1,
                                                                                                                     1),
                                                                                                                 groups=1,
                                                                                                                 bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_16_Branch_1_Conv2d_0c_7x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2,
            'InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_16/Branch_1/Conv2d_0c_7x1/BatchNorm/FusedBatchNorm',
            num_features=192, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_16_Conv2d_1x1_Conv2D = self.__conv(2,
                                                                                                     name='InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_16/Conv2d_1x1/Conv2D',
                                                                                                     in_channels=384,
                                                                                                     out_channels=1088,
                                                                                                     kernel_size=(1, 1),
                                                                                                     stride=(1, 1),
                                                                                                     groups=1,
                                                                                                     bias=True)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_17_Branch_0_Conv2d_1x1_Conv2D = self.__conv(2,
                                                                                                              name='InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_17/Branch_0/Conv2d_1x1/Conv2D',
                                                                                                              in_channels=1088,
                                                                                                              out_channels=192,
                                                                                                              kernel_size=(
                                                                                                                  1, 1),
                                                                                                              stride=(
                                                                                                                  1, 1),
                                                                                                              groups=1,
                                                                                                              bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_17_Branch_1_Conv2d_0a_1x1_Conv2D = self.__conv(2,
                                                                                                                 name='InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_17/Branch_1/Conv2d_0a_1x1/Conv2D',
                                                                                                                 in_channels=1088,
                                                                                                                 out_channels=128,
                                                                                                                 kernel_size=(
                                                                                                                     1,
                                                                                                                     1),
                                                                                                                 stride=(
                                                                                                                     1,
                                                                                                                     1),
                                                                                                                 groups=1,
                                                                                                                 bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_17_Branch_0_Conv2d_1x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_17/Branch_0/Conv2d_1x1/BatchNorm/FusedBatchNorm',
            num_features=192, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_17_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2,
            'InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_17/Branch_1/Conv2d_0a_1x1/BatchNorm/FusedBatchNorm',
            num_features=128, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_17_Branch_1_Conv2d_0b_1x7_Conv2D = self.__conv(2,
                                                                                                                 name='InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_17/Branch_1/Conv2d_0b_1x7/Conv2D',
                                                                                                                 in_channels=128,
                                                                                                                 out_channels=160,
                                                                                                                 kernel_size=(
                                                                                                                     1,
                                                                                                                     7),
                                                                                                                 stride=(
                                                                                                                     1,
                                                                                                                     1),
                                                                                                                 groups=1,
                                                                                                                 bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_17_Branch_1_Conv2d_0b_1x7_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2,
            'InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_17/Branch_1/Conv2d_0b_1x7/BatchNorm/FusedBatchNorm',
            num_features=160, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_17_Branch_1_Conv2d_0c_7x1_Conv2D = self.__conv(2,
                                                                                                                 name='InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_17/Branch_1/Conv2d_0c_7x1/Conv2D',
                                                                                                                 in_channels=160,
                                                                                                                 out_channels=192,
                                                                                                                 kernel_size=(
                                                                                                                     7,
                                                                                                                     1),
                                                                                                                 stride=(
                                                                                                                     1,
                                                                                                                     1),
                                                                                                                 groups=1,
                                                                                                                 bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_17_Branch_1_Conv2d_0c_7x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2,
            'InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_17/Branch_1/Conv2d_0c_7x1/BatchNorm/FusedBatchNorm',
            num_features=192, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_17_Conv2d_1x1_Conv2D = self.__conv(2,
                                                                                                     name='InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_17/Conv2d_1x1/Conv2D',
                                                                                                     in_channels=384,
                                                                                                     out_channels=1088,
                                                                                                     kernel_size=(1, 1),
                                                                                                     stride=(1, 1),
                                                                                                     groups=1,
                                                                                                     bias=True)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_18_Branch_0_Conv2d_1x1_Conv2D = self.__conv(2,
                                                                                                              name='InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_18/Branch_0/Conv2d_1x1/Conv2D',
                                                                                                              in_channels=1088,
                                                                                                              out_channels=192,
                                                                                                              kernel_size=(
                                                                                                                  1, 1),
                                                                                                              stride=(
                                                                                                                  1, 1),
                                                                                                              groups=1,
                                                                                                              bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_18_Branch_1_Conv2d_0a_1x1_Conv2D = self.__conv(2,
                                                                                                                 name='InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_18/Branch_1/Conv2d_0a_1x1/Conv2D',
                                                                                                                 in_channels=1088,
                                                                                                                 out_channels=128,
                                                                                                                 kernel_size=(
                                                                                                                     1,
                                                                                                                     1),
                                                                                                                 stride=(
                                                                                                                     1,
                                                                                                                     1),
                                                                                                                 groups=1,
                                                                                                                 bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_18_Branch_0_Conv2d_1x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_18/Branch_0/Conv2d_1x1/BatchNorm/FusedBatchNorm',
            num_features=192, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_18_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2,
            'InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_18/Branch_1/Conv2d_0a_1x1/BatchNorm/FusedBatchNorm',
            num_features=128, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_18_Branch_1_Conv2d_0b_1x7_Conv2D = self.__conv(2,
                                                                                                                 name='InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_18/Branch_1/Conv2d_0b_1x7/Conv2D',
                                                                                                                 in_channels=128,
                                                                                                                 out_channels=160,
                                                                                                                 kernel_size=(
                                                                                                                     1,
                                                                                                                     7),
                                                                                                                 stride=(
                                                                                                                     1,
                                                                                                                     1),
                                                                                                                 groups=1,
                                                                                                                 bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_18_Branch_1_Conv2d_0b_1x7_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2,
            'InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_18/Branch_1/Conv2d_0b_1x7/BatchNorm/FusedBatchNorm',
            num_features=160, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_18_Branch_1_Conv2d_0c_7x1_Conv2D = self.__conv(2,
                                                                                                                 name='InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_18/Branch_1/Conv2d_0c_7x1/Conv2D',
                                                                                                                 in_channels=160,
                                                                                                                 out_channels=192,
                                                                                                                 kernel_size=(
                                                                                                                     7,
                                                                                                                     1),
                                                                                                                 stride=(
                                                                                                                     1,
                                                                                                                     1),
                                                                                                                 groups=1,
                                                                                                                 bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_18_Branch_1_Conv2d_0c_7x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2,
            'InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_18/Branch_1/Conv2d_0c_7x1/BatchNorm/FusedBatchNorm',
            num_features=192, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_18_Conv2d_1x1_Conv2D = self.__conv(2,
                                                                                                     name='InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_18/Conv2d_1x1/Conv2D',
                                                                                                     in_channels=384,
                                                                                                     out_channels=1088,
                                                                                                     kernel_size=(1, 1),
                                                                                                     stride=(1, 1),
                                                                                                     groups=1,
                                                                                                     bias=True)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_19_Branch_0_Conv2d_1x1_Conv2D = self.__conv(2,
                                                                                                              name='InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_19/Branch_0/Conv2d_1x1/Conv2D',
                                                                                                              in_channels=1088,
                                                                                                              out_channels=192,
                                                                                                              kernel_size=(
                                                                                                                  1, 1),
                                                                                                              stride=(
                                                                                                                  1, 1),
                                                                                                              groups=1,
                                                                                                              bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_19_Branch_1_Conv2d_0a_1x1_Conv2D = self.__conv(2,
                                                                                                                 name='InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_19/Branch_1/Conv2d_0a_1x1/Conv2D',
                                                                                                                 in_channels=1088,
                                                                                                                 out_channels=128,
                                                                                                                 kernel_size=(
                                                                                                                     1,
                                                                                                                     1),
                                                                                                                 stride=(
                                                                                                                     1,
                                                                                                                     1),
                                                                                                                 groups=1,
                                                                                                                 bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_19_Branch_0_Conv2d_1x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_19/Branch_0/Conv2d_1x1/BatchNorm/FusedBatchNorm',
            num_features=192, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_19_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2,
            'InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_19/Branch_1/Conv2d_0a_1x1/BatchNorm/FusedBatchNorm',
            num_features=128, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_19_Branch_1_Conv2d_0b_1x7_Conv2D = self.__conv(2,
                                                                                                                 name='InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_19/Branch_1/Conv2d_0b_1x7/Conv2D',
                                                                                                                 in_channels=128,
                                                                                                                 out_channels=160,
                                                                                                                 kernel_size=(
                                                                                                                     1,
                                                                                                                     7),
                                                                                                                 stride=(
                                                                                                                     1,
                                                                                                                     1),
                                                                                                                 groups=1,
                                                                                                                 bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_19_Branch_1_Conv2d_0b_1x7_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2,
            'InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_19/Branch_1/Conv2d_0b_1x7/BatchNorm/FusedBatchNorm',
            num_features=160, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_19_Branch_1_Conv2d_0c_7x1_Conv2D = self.__conv(2,
                                                                                                                 name='InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_19/Branch_1/Conv2d_0c_7x1/Conv2D',
                                                                                                                 in_channels=160,
                                                                                                                 out_channels=192,
                                                                                                                 kernel_size=(
                                                                                                                     7,
                                                                                                                     1),
                                                                                                                 stride=(
                                                                                                                     1,
                                                                                                                     1),
                                                                                                                 groups=1,
                                                                                                                 bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_19_Branch_1_Conv2d_0c_7x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2,
            'InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_19/Branch_1/Conv2d_0c_7x1/BatchNorm/FusedBatchNorm',
            num_features=192, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_19_Conv2d_1x1_Conv2D = self.__conv(2,
                                                                                                     name='InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_19/Conv2d_1x1/Conv2D',
                                                                                                     in_channels=384,
                                                                                                     out_channels=1088,
                                                                                                     kernel_size=(1, 1),
                                                                                                     stride=(1, 1),
                                                                                                     groups=1,
                                                                                                     bias=True)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_20_Branch_0_Conv2d_1x1_Conv2D = self.__conv(2,
                                                                                                              name='InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_20/Branch_0/Conv2d_1x1/Conv2D',
                                                                                                              in_channels=1088,
                                                                                                              out_channels=192,
                                                                                                              kernel_size=(
                                                                                                                  1, 1),
                                                                                                              stride=(
                                                                                                                  1, 1),
                                                                                                              groups=1,
                                                                                                              bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_20_Branch_1_Conv2d_0a_1x1_Conv2D = self.__conv(2,
                                                                                                                 name='InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_20/Branch_1/Conv2d_0a_1x1/Conv2D',
                                                                                                                 in_channels=1088,
                                                                                                                 out_channels=128,
                                                                                                                 kernel_size=(
                                                                                                                     1,
                                                                                                                     1),
                                                                                                                 stride=(
                                                                                                                     1,
                                                                                                                     1),
                                                                                                                 groups=1,
                                                                                                                 bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_20_Branch_0_Conv2d_1x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_20/Branch_0/Conv2d_1x1/BatchNorm/FusedBatchNorm',
            num_features=192, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_20_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2,
            'InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_20/Branch_1/Conv2d_0a_1x1/BatchNorm/FusedBatchNorm',
            num_features=128, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_20_Branch_1_Conv2d_0b_1x7_Conv2D = self.__conv(2,
                                                                                                                 name='InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_20/Branch_1/Conv2d_0b_1x7/Conv2D',
                                                                                                                 in_channels=128,
                                                                                                                 out_channels=160,
                                                                                                                 kernel_size=(
                                                                                                                     1,
                                                                                                                     7),
                                                                                                                 stride=(
                                                                                                                     1,
                                                                                                                     1),
                                                                                                                 groups=1,
                                                                                                                 bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_20_Branch_1_Conv2d_0b_1x7_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2,
            'InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_20/Branch_1/Conv2d_0b_1x7/BatchNorm/FusedBatchNorm',
            num_features=160, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_20_Branch_1_Conv2d_0c_7x1_Conv2D = self.__conv(2,
                                                                                                                 name='InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_20/Branch_1/Conv2d_0c_7x1/Conv2D',
                                                                                                                 in_channels=160,
                                                                                                                 out_channels=192,
                                                                                                                 kernel_size=(
                                                                                                                     7,
                                                                                                                     1),
                                                                                                                 stride=(
                                                                                                                     1,
                                                                                                                     1),
                                                                                                                 groups=1,
                                                                                                                 bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_20_Branch_1_Conv2d_0c_7x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2,
            'InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_20/Branch_1/Conv2d_0c_7x1/BatchNorm/FusedBatchNorm',
            num_features=192, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_20_Conv2d_1x1_Conv2D = self.__conv(2,
                                                                                                     name='InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_20/Conv2d_1x1/Conv2D',
                                                                                                     in_channels=384,
                                                                                                     out_channels=1088,
                                                                                                     kernel_size=(1, 1),
                                                                                                     stride=(1, 1),
                                                                                                     groups=1,
                                                                                                     bias=True)
        self.InceptionResnetV2_InceptionResnetV2_Mixed_7a_Branch_0_Conv2d_0a_1x1_Conv2D = self.__conv(2,
                                                                                                      name='InceptionResnetV2/InceptionResnetV2/Mixed_7a/Branch_0/Conv2d_0a_1x1/Conv2D',
                                                                                                      in_channels=1088,
                                                                                                      out_channels=256,
                                                                                                      kernel_size=(
                                                                                                          1, 1),
                                                                                                      stride=(1, 1),
                                                                                                      groups=1,
                                                                                                      bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Mixed_7a_Branch_1_Conv2d_0a_1x1_Conv2D = self.__conv(2,
                                                                                                      name='InceptionResnetV2/InceptionResnetV2/Mixed_7a/Branch_1/Conv2d_0a_1x1/Conv2D',
                                                                                                      in_channels=1088,
                                                                                                      out_channels=256,
                                                                                                      kernel_size=(
                                                                                                          1, 1),
                                                                                                      stride=(1, 1),
                                                                                                      groups=1,
                                                                                                      bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Mixed_7a_Branch_2_Conv2d_0a_1x1_Conv2D = self.__conv(2,
                                                                                                      name='InceptionResnetV2/InceptionResnetV2/Mixed_7a/Branch_2/Conv2d_0a_1x1/Conv2D',
                                                                                                      in_channels=1088,
                                                                                                      out_channels=256,
                                                                                                      kernel_size=(
                                                                                                          1, 1),
                                                                                                      stride=(1, 1),
                                                                                                      groups=1,
                                                                                                      bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Mixed_7a_Branch_0_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionResnetV2/InceptionResnetV2/Mixed_7a/Branch_0/Conv2d_0a_1x1/BatchNorm/FusedBatchNorm',
            num_features=256, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Mixed_7a_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionResnetV2/InceptionResnetV2/Mixed_7a/Branch_1/Conv2d_0a_1x1/BatchNorm/FusedBatchNorm',
            num_features=256, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Mixed_7a_Branch_2_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionResnetV2/InceptionResnetV2/Mixed_7a/Branch_2/Conv2d_0a_1x1/BatchNorm/FusedBatchNorm',
            num_features=256, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_AuxLogits_Conv2d_1b_1x1_Conv2D = self.__conv(2,
                                                                            name='InceptionResnetV2/AuxLogits/Conv2d_1b_1x1/Conv2D',
                                                                            in_channels=1088, out_channels=128,
                                                                            kernel_size=(1, 1), stride=(1, 1), groups=1,
                                                                            bias=None)
        self.InceptionResnetV2_AuxLogits_Conv2d_1b_1x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(2,
                                                                                                             'InceptionResnetV2/AuxLogits/Conv2d_1b_1x1/BatchNorm/FusedBatchNorm',
                                                                                                             num_features=128,
                                                                                                             eps=0.0010000000474974513,
                                                                                                             momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Mixed_7a_Branch_0_Conv2d_1a_3x3_Conv2D = self.__conv(2,
                                                                                                      name='InceptionResnetV2/InceptionResnetV2/Mixed_7a/Branch_0/Conv2d_1a_3x3/Conv2D',
                                                                                                      in_channels=256,
                                                                                                      out_channels=384,
                                                                                                      kernel_size=(
                                                                                                          3, 3),
                                                                                                      stride=(2, 2),
                                                                                                      groups=1,
                                                                                                      bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Mixed_7a_Branch_1_Conv2d_1a_3x3_Conv2D = self.__conv(2,
                                                                                                      name='InceptionResnetV2/InceptionResnetV2/Mixed_7a/Branch_1/Conv2d_1a_3x3/Conv2D',
                                                                                                      in_channels=256,
                                                                                                      out_channels=288,
                                                                                                      kernel_size=(
                                                                                                          3, 3),
                                                                                                      stride=(2, 2),
                                                                                                      groups=1,
                                                                                                      bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Mixed_7a_Branch_2_Conv2d_0b_3x3_Conv2D = self.__conv(2,
                                                                                                      name='InceptionResnetV2/InceptionResnetV2/Mixed_7a/Branch_2/Conv2d_0b_3x3/Conv2D',
                                                                                                      in_channels=256,
                                                                                                      out_channels=288,
                                                                                                      kernel_size=(
                                                                                                          3, 3),
                                                                                                      stride=(1, 1),
                                                                                                      groups=1,
                                                                                                      bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Mixed_7a_Branch_0_Conv2d_1a_3x3_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionResnetV2/InceptionResnetV2/Mixed_7a/Branch_0/Conv2d_1a_3x3/BatchNorm/FusedBatchNorm',
            num_features=384, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Mixed_7a_Branch_1_Conv2d_1a_3x3_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionResnetV2/InceptionResnetV2/Mixed_7a/Branch_1/Conv2d_1a_3x3/BatchNorm/FusedBatchNorm',
            num_features=288, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Mixed_7a_Branch_2_Conv2d_0b_3x3_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionResnetV2/InceptionResnetV2/Mixed_7a/Branch_2/Conv2d_0b_3x3/BatchNorm/FusedBatchNorm',
            num_features=288, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_AuxLogits_Conv2d_2a_5x5_Conv2D = self.__conv(2,
                                                                            name='InceptionResnetV2/AuxLogits/Conv2d_2a_5x5/Conv2D',
                                                                            in_channels=128, out_channels=768,
                                                                            kernel_size=(5, 5), stride=(1, 1), groups=1,
                                                                            bias=None)
        self.InceptionResnetV2_AuxLogits_Conv2d_2a_5x5_BatchNorm_FusedBatchNorm = self.__batch_normalization(2,
                                                                                                             'InceptionResnetV2/AuxLogits/Conv2d_2a_5x5/BatchNorm/FusedBatchNorm',
                                                                                                             num_features=768,
                                                                                                             eps=0.0010000000474974513,
                                                                                                             momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Mixed_7a_Branch_2_Conv2d_1a_3x3_Conv2D = self.__conv(2,
                                                                                                      name='InceptionResnetV2/InceptionResnetV2/Mixed_7a/Branch_2/Conv2d_1a_3x3/Conv2D',
                                                                                                      in_channels=288,
                                                                                                      out_channels=320,
                                                                                                      kernel_size=(
                                                                                                          3, 3),
                                                                                                      stride=(2, 2),
                                                                                                      groups=1,
                                                                                                      bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Mixed_7a_Branch_2_Conv2d_1a_3x3_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionResnetV2/InceptionResnetV2/Mixed_7a/Branch_2/Conv2d_1a_3x3/BatchNorm/FusedBatchNorm',
            num_features=320, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_AuxLogits_Logits_MatMul = self.__dense(name='InceptionResnetV2/AuxLogits/Logits/MatMul',
                                                                      in_features=768, out_features=1001, bias=True)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_1_Branch_0_Conv2d_1x1_Conv2D = self.__conv(2,
                                                                                                            name='InceptionResnetV2/InceptionResnetV2/Repeat_2/block8_1/Branch_0/Conv2d_1x1/Conv2D',
                                                                                                            in_channels=2080,
                                                                                                            out_channels=192,
                                                                                                            kernel_size=(
                                                                                                                1, 1),
                                                                                                            stride=(
                                                                                                                1, 1),
                                                                                                            groups=1,
                                                                                                            bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_1_Branch_1_Conv2d_0a_1x1_Conv2D = self.__conv(2,
                                                                                                               name='InceptionResnetV2/InceptionResnetV2/Repeat_2/block8_1/Branch_1/Conv2d_0a_1x1/Conv2D',
                                                                                                               in_channels=2080,
                                                                                                               out_channels=192,
                                                                                                               kernel_size=(
                                                                                                                   1,
                                                                                                                   1),
                                                                                                               stride=(
                                                                                                                   1,
                                                                                                                   1),
                                                                                                               groups=1,
                                                                                                               bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_1_Branch_0_Conv2d_1x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionResnetV2/InceptionResnetV2/Repeat_2/block8_1/Branch_0/Conv2d_1x1/BatchNorm/FusedBatchNorm',
            num_features=192, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_1_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionResnetV2/InceptionResnetV2/Repeat_2/block8_1/Branch_1/Conv2d_0a_1x1/BatchNorm/FusedBatchNorm',
            num_features=192, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_1_Branch_1_Conv2d_0b_1x3_Conv2D = self.__conv(2,
                                                                                                               name='InceptionResnetV2/InceptionResnetV2/Repeat_2/block8_1/Branch_1/Conv2d_0b_1x3/Conv2D',
                                                                                                               in_channels=192,
                                                                                                               out_channels=224,
                                                                                                               kernel_size=(
                                                                                                                   1,
                                                                                                                   3),
                                                                                                               stride=(
                                                                                                                   1,
                                                                                                                   1),
                                                                                                               groups=1,
                                                                                                               bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_1_Branch_1_Conv2d_0b_1x3_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionResnetV2/InceptionResnetV2/Repeat_2/block8_1/Branch_1/Conv2d_0b_1x3/BatchNorm/FusedBatchNorm',
            num_features=224, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_1_Branch_1_Conv2d_0c_3x1_Conv2D = self.__conv(2,
                                                                                                               name='InceptionResnetV2/InceptionResnetV2/Repeat_2/block8_1/Branch_1/Conv2d_0c_3x1/Conv2D',
                                                                                                               in_channels=224,
                                                                                                               out_channels=256,
                                                                                                               kernel_size=(
                                                                                                                   3,
                                                                                                                   1),
                                                                                                               stride=(
                                                                                                                   1,
                                                                                                                   1),
                                                                                                               groups=1,
                                                                                                               bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_1_Branch_1_Conv2d_0c_3x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionResnetV2/InceptionResnetV2/Repeat_2/block8_1/Branch_1/Conv2d_0c_3x1/BatchNorm/FusedBatchNorm',
            num_features=256, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_1_Conv2d_1x1_Conv2D = self.__conv(2,
                                                                                                   name='InceptionResnetV2/InceptionResnetV2/Repeat_2/block8_1/Conv2d_1x1/Conv2D',
                                                                                                   in_channels=448,
                                                                                                   out_channels=2080,
                                                                                                   kernel_size=(1, 1),
                                                                                                   stride=(1, 1),
                                                                                                   groups=1, bias=True)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_2_Branch_0_Conv2d_1x1_Conv2D = self.__conv(2,
                                                                                                            name='InceptionResnetV2/InceptionResnetV2/Repeat_2/block8_2/Branch_0/Conv2d_1x1/Conv2D',
                                                                                                            in_channels=2080,
                                                                                                            out_channels=192,
                                                                                                            kernel_size=(
                                                                                                                1, 1),
                                                                                                            stride=(
                                                                                                                1, 1),
                                                                                                            groups=1,
                                                                                                            bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_2_Branch_1_Conv2d_0a_1x1_Conv2D = self.__conv(2,
                                                                                                               name='InceptionResnetV2/InceptionResnetV2/Repeat_2/block8_2/Branch_1/Conv2d_0a_1x1/Conv2D',
                                                                                                               in_channels=2080,
                                                                                                               out_channels=192,
                                                                                                               kernel_size=(
                                                                                                                   1,
                                                                                                                   1),
                                                                                                               stride=(
                                                                                                                   1,
                                                                                                                   1),
                                                                                                               groups=1,
                                                                                                               bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_2_Branch_0_Conv2d_1x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionResnetV2/InceptionResnetV2/Repeat_2/block8_2/Branch_0/Conv2d_1x1/BatchNorm/FusedBatchNorm',
            num_features=192, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_2_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionResnetV2/InceptionResnetV2/Repeat_2/block8_2/Branch_1/Conv2d_0a_1x1/BatchNorm/FusedBatchNorm',
            num_features=192, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_2_Branch_1_Conv2d_0b_1x3_Conv2D = self.__conv(2,
                                                                                                               name='InceptionResnetV2/InceptionResnetV2/Repeat_2/block8_2/Branch_1/Conv2d_0b_1x3/Conv2D',
                                                                                                               in_channels=192,
                                                                                                               out_channels=224,
                                                                                                               kernel_size=(
                                                                                                                   1,
                                                                                                                   3),
                                                                                                               stride=(
                                                                                                                   1,
                                                                                                                   1),
                                                                                                               groups=1,
                                                                                                               bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_2_Branch_1_Conv2d_0b_1x3_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionResnetV2/InceptionResnetV2/Repeat_2/block8_2/Branch_1/Conv2d_0b_1x3/BatchNorm/FusedBatchNorm',
            num_features=224, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_2_Branch_1_Conv2d_0c_3x1_Conv2D = self.__conv(2,
                                                                                                               name='InceptionResnetV2/InceptionResnetV2/Repeat_2/block8_2/Branch_1/Conv2d_0c_3x1/Conv2D',
                                                                                                               in_channels=224,
                                                                                                               out_channels=256,
                                                                                                               kernel_size=(
                                                                                                                   3,
                                                                                                                   1),
                                                                                                               stride=(
                                                                                                                   1,
                                                                                                                   1),
                                                                                                               groups=1,
                                                                                                               bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_2_Branch_1_Conv2d_0c_3x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionResnetV2/InceptionResnetV2/Repeat_2/block8_2/Branch_1/Conv2d_0c_3x1/BatchNorm/FusedBatchNorm',
            num_features=256, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_2_Conv2d_1x1_Conv2D = self.__conv(2,
                                                                                                   name='InceptionResnetV2/InceptionResnetV2/Repeat_2/block8_2/Conv2d_1x1/Conv2D',
                                                                                                   in_channels=448,
                                                                                                   out_channels=2080,
                                                                                                   kernel_size=(1, 1),
                                                                                                   stride=(1, 1),
                                                                                                   groups=1, bias=True)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_3_Branch_0_Conv2d_1x1_Conv2D = self.__conv(2,
                                                                                                            name='InceptionResnetV2/InceptionResnetV2/Repeat_2/block8_3/Branch_0/Conv2d_1x1/Conv2D',
                                                                                                            in_channels=2080,
                                                                                                            out_channels=192,
                                                                                                            kernel_size=(
                                                                                                                1, 1),
                                                                                                            stride=(
                                                                                                                1, 1),
                                                                                                            groups=1,
                                                                                                            bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_3_Branch_1_Conv2d_0a_1x1_Conv2D = self.__conv(2,
                                                                                                               name='InceptionResnetV2/InceptionResnetV2/Repeat_2/block8_3/Branch_1/Conv2d_0a_1x1/Conv2D',
                                                                                                               in_channels=2080,
                                                                                                               out_channels=192,
                                                                                                               kernel_size=(
                                                                                                                   1,
                                                                                                                   1),
                                                                                                               stride=(
                                                                                                                   1,
                                                                                                                   1),
                                                                                                               groups=1,
                                                                                                               bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_3_Branch_0_Conv2d_1x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionResnetV2/InceptionResnetV2/Repeat_2/block8_3/Branch_0/Conv2d_1x1/BatchNorm/FusedBatchNorm',
            num_features=192, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_3_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionResnetV2/InceptionResnetV2/Repeat_2/block8_3/Branch_1/Conv2d_0a_1x1/BatchNorm/FusedBatchNorm',
            num_features=192, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_3_Branch_1_Conv2d_0b_1x3_Conv2D = self.__conv(2,
                                                                                                               name='InceptionResnetV2/InceptionResnetV2/Repeat_2/block8_3/Branch_1/Conv2d_0b_1x3/Conv2D',
                                                                                                               in_channels=192,
                                                                                                               out_channels=224,
                                                                                                               kernel_size=(
                                                                                                                   1,
                                                                                                                   3),
                                                                                                               stride=(
                                                                                                                   1,
                                                                                                                   1),
                                                                                                               groups=1,
                                                                                                               bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_3_Branch_1_Conv2d_0b_1x3_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionResnetV2/InceptionResnetV2/Repeat_2/block8_3/Branch_1/Conv2d_0b_1x3/BatchNorm/FusedBatchNorm',
            num_features=224, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_3_Branch_1_Conv2d_0c_3x1_Conv2D = self.__conv(2,
                                                                                                               name='InceptionResnetV2/InceptionResnetV2/Repeat_2/block8_3/Branch_1/Conv2d_0c_3x1/Conv2D',
                                                                                                               in_channels=224,
                                                                                                               out_channels=256,
                                                                                                               kernel_size=(
                                                                                                                   3,
                                                                                                                   1),
                                                                                                               stride=(
                                                                                                                   1,
                                                                                                                   1),
                                                                                                               groups=1,
                                                                                                               bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_3_Branch_1_Conv2d_0c_3x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionResnetV2/InceptionResnetV2/Repeat_2/block8_3/Branch_1/Conv2d_0c_3x1/BatchNorm/FusedBatchNorm',
            num_features=256, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_3_Conv2d_1x1_Conv2D = self.__conv(2,
                                                                                                   name='InceptionResnetV2/InceptionResnetV2/Repeat_2/block8_3/Conv2d_1x1/Conv2D',
                                                                                                   in_channels=448,
                                                                                                   out_channels=2080,
                                                                                                   kernel_size=(1, 1),
                                                                                                   stride=(1, 1),
                                                                                                   groups=1, bias=True)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_4_Branch_0_Conv2d_1x1_Conv2D = self.__conv(2,
                                                                                                            name='InceptionResnetV2/InceptionResnetV2/Repeat_2/block8_4/Branch_0/Conv2d_1x1/Conv2D',
                                                                                                            in_channels=2080,
                                                                                                            out_channels=192,
                                                                                                            kernel_size=(
                                                                                                                1, 1),
                                                                                                            stride=(
                                                                                                                1, 1),
                                                                                                            groups=1,
                                                                                                            bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_4_Branch_1_Conv2d_0a_1x1_Conv2D = self.__conv(2,
                                                                                                               name='InceptionResnetV2/InceptionResnetV2/Repeat_2/block8_4/Branch_1/Conv2d_0a_1x1/Conv2D',
                                                                                                               in_channels=2080,
                                                                                                               out_channels=192,
                                                                                                               kernel_size=(
                                                                                                                   1,
                                                                                                                   1),
                                                                                                               stride=(
                                                                                                                   1,
                                                                                                                   1),
                                                                                                               groups=1,
                                                                                                               bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_4_Branch_0_Conv2d_1x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionResnetV2/InceptionResnetV2/Repeat_2/block8_4/Branch_0/Conv2d_1x1/BatchNorm/FusedBatchNorm',
            num_features=192, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_4_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionResnetV2/InceptionResnetV2/Repeat_2/block8_4/Branch_1/Conv2d_0a_1x1/BatchNorm/FusedBatchNorm',
            num_features=192, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_4_Branch_1_Conv2d_0b_1x3_Conv2D = self.__conv(2,
                                                                                                               name='InceptionResnetV2/InceptionResnetV2/Repeat_2/block8_4/Branch_1/Conv2d_0b_1x3/Conv2D',
                                                                                                               in_channels=192,
                                                                                                               out_channels=224,
                                                                                                               kernel_size=(
                                                                                                                   1,
                                                                                                                   3),
                                                                                                               stride=(
                                                                                                                   1,
                                                                                                                   1),
                                                                                                               groups=1,
                                                                                                               bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_4_Branch_1_Conv2d_0b_1x3_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionResnetV2/InceptionResnetV2/Repeat_2/block8_4/Branch_1/Conv2d_0b_1x3/BatchNorm/FusedBatchNorm',
            num_features=224, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_4_Branch_1_Conv2d_0c_3x1_Conv2D = self.__conv(2,
                                                                                                               name='InceptionResnetV2/InceptionResnetV2/Repeat_2/block8_4/Branch_1/Conv2d_0c_3x1/Conv2D',
                                                                                                               in_channels=224,
                                                                                                               out_channels=256,
                                                                                                               kernel_size=(
                                                                                                                   3,
                                                                                                                   1),
                                                                                                               stride=(
                                                                                                                   1,
                                                                                                                   1),
                                                                                                               groups=1,
                                                                                                               bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_4_Branch_1_Conv2d_0c_3x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionResnetV2/InceptionResnetV2/Repeat_2/block8_4/Branch_1/Conv2d_0c_3x1/BatchNorm/FusedBatchNorm',
            num_features=256, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_4_Conv2d_1x1_Conv2D = self.__conv(2,
                                                                                                   name='InceptionResnetV2/InceptionResnetV2/Repeat_2/block8_4/Conv2d_1x1/Conv2D',
                                                                                                   in_channels=448,
                                                                                                   out_channels=2080,
                                                                                                   kernel_size=(1, 1),
                                                                                                   stride=(1, 1),
                                                                                                   groups=1, bias=True)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_5_Branch_0_Conv2d_1x1_Conv2D = self.__conv(2,
                                                                                                            name='InceptionResnetV2/InceptionResnetV2/Repeat_2/block8_5/Branch_0/Conv2d_1x1/Conv2D',
                                                                                                            in_channels=2080,
                                                                                                            out_channels=192,
                                                                                                            kernel_size=(
                                                                                                                1, 1),
                                                                                                            stride=(
                                                                                                                1, 1),
                                                                                                            groups=1,
                                                                                                            bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_5_Branch_1_Conv2d_0a_1x1_Conv2D = self.__conv(2,
                                                                                                               name='InceptionResnetV2/InceptionResnetV2/Repeat_2/block8_5/Branch_1/Conv2d_0a_1x1/Conv2D',
                                                                                                               in_channels=2080,
                                                                                                               out_channels=192,
                                                                                                               kernel_size=(
                                                                                                                   1,
                                                                                                                   1),
                                                                                                               stride=(
                                                                                                                   1,
                                                                                                                   1),
                                                                                                               groups=1,
                                                                                                               bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_5_Branch_0_Conv2d_1x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionResnetV2/InceptionResnetV2/Repeat_2/block8_5/Branch_0/Conv2d_1x1/BatchNorm/FusedBatchNorm',
            num_features=192, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_5_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionResnetV2/InceptionResnetV2/Repeat_2/block8_5/Branch_1/Conv2d_0a_1x1/BatchNorm/FusedBatchNorm',
            num_features=192, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_5_Branch_1_Conv2d_0b_1x3_Conv2D = self.__conv(2,
                                                                                                               name='InceptionResnetV2/InceptionResnetV2/Repeat_2/block8_5/Branch_1/Conv2d_0b_1x3/Conv2D',
                                                                                                               in_channels=192,
                                                                                                               out_channels=224,
                                                                                                               kernel_size=(
                                                                                                                   1,
                                                                                                                   3),
                                                                                                               stride=(
                                                                                                                   1,
                                                                                                                   1),
                                                                                                               groups=1,
                                                                                                               bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_5_Branch_1_Conv2d_0b_1x3_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionResnetV2/InceptionResnetV2/Repeat_2/block8_5/Branch_1/Conv2d_0b_1x3/BatchNorm/FusedBatchNorm',
            num_features=224, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_5_Branch_1_Conv2d_0c_3x1_Conv2D = self.__conv(2,
                                                                                                               name='InceptionResnetV2/InceptionResnetV2/Repeat_2/block8_5/Branch_1/Conv2d_0c_3x1/Conv2D',
                                                                                                               in_channels=224,
                                                                                                               out_channels=256,
                                                                                                               kernel_size=(
                                                                                                                   3,
                                                                                                                   1),
                                                                                                               stride=(
                                                                                                                   1,
                                                                                                                   1),
                                                                                                               groups=1,
                                                                                                               bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_5_Branch_1_Conv2d_0c_3x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionResnetV2/InceptionResnetV2/Repeat_2/block8_5/Branch_1/Conv2d_0c_3x1/BatchNorm/FusedBatchNorm',
            num_features=256, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_5_Conv2d_1x1_Conv2D = self.__conv(2,
                                                                                                   name='InceptionResnetV2/InceptionResnetV2/Repeat_2/block8_5/Conv2d_1x1/Conv2D',
                                                                                                   in_channels=448,
                                                                                                   out_channels=2080,
                                                                                                   kernel_size=(1, 1),
                                                                                                   stride=(1, 1),
                                                                                                   groups=1, bias=True)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_6_Branch_0_Conv2d_1x1_Conv2D = self.__conv(2,
                                                                                                            name='InceptionResnetV2/InceptionResnetV2/Repeat_2/block8_6/Branch_0/Conv2d_1x1/Conv2D',
                                                                                                            in_channels=2080,
                                                                                                            out_channels=192,
                                                                                                            kernel_size=(
                                                                                                                1, 1),
                                                                                                            stride=(
                                                                                                                1, 1),
                                                                                                            groups=1,
                                                                                                            bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_6_Branch_1_Conv2d_0a_1x1_Conv2D = self.__conv(2,
                                                                                                               name='InceptionResnetV2/InceptionResnetV2/Repeat_2/block8_6/Branch_1/Conv2d_0a_1x1/Conv2D',
                                                                                                               in_channels=2080,
                                                                                                               out_channels=192,
                                                                                                               kernel_size=(
                                                                                                                   1,
                                                                                                                   1),
                                                                                                               stride=(
                                                                                                                   1,
                                                                                                                   1),
                                                                                                               groups=1,
                                                                                                               bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_6_Branch_0_Conv2d_1x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionResnetV2/InceptionResnetV2/Repeat_2/block8_6/Branch_0/Conv2d_1x1/BatchNorm/FusedBatchNorm',
            num_features=192, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_6_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionResnetV2/InceptionResnetV2/Repeat_2/block8_6/Branch_1/Conv2d_0a_1x1/BatchNorm/FusedBatchNorm',
            num_features=192, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_6_Branch_1_Conv2d_0b_1x3_Conv2D = self.__conv(2,
                                                                                                               name='InceptionResnetV2/InceptionResnetV2/Repeat_2/block8_6/Branch_1/Conv2d_0b_1x3/Conv2D',
                                                                                                               in_channels=192,
                                                                                                               out_channels=224,
                                                                                                               kernel_size=(
                                                                                                                   1,
                                                                                                                   3),
                                                                                                               stride=(
                                                                                                                   1,
                                                                                                                   1),
                                                                                                               groups=1,
                                                                                                               bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_6_Branch_1_Conv2d_0b_1x3_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionResnetV2/InceptionResnetV2/Repeat_2/block8_6/Branch_1/Conv2d_0b_1x3/BatchNorm/FusedBatchNorm',
            num_features=224, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_6_Branch_1_Conv2d_0c_3x1_Conv2D = self.__conv(2,
                                                                                                               name='InceptionResnetV2/InceptionResnetV2/Repeat_2/block8_6/Branch_1/Conv2d_0c_3x1/Conv2D',
                                                                                                               in_channels=224,
                                                                                                               out_channels=256,
                                                                                                               kernel_size=(
                                                                                                                   3,
                                                                                                                   1),
                                                                                                               stride=(
                                                                                                                   1,
                                                                                                                   1),
                                                                                                               groups=1,
                                                                                                               bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_6_Branch_1_Conv2d_0c_3x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionResnetV2/InceptionResnetV2/Repeat_2/block8_6/Branch_1/Conv2d_0c_3x1/BatchNorm/FusedBatchNorm',
            num_features=256, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_6_Conv2d_1x1_Conv2D = self.__conv(2,
                                                                                                   name='InceptionResnetV2/InceptionResnetV2/Repeat_2/block8_6/Conv2d_1x1/Conv2D',
                                                                                                   in_channels=448,
                                                                                                   out_channels=2080,
                                                                                                   kernel_size=(1, 1),
                                                                                                   stride=(1, 1),
                                                                                                   groups=1, bias=True)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_7_Branch_0_Conv2d_1x1_Conv2D = self.__conv(2,
                                                                                                            name='InceptionResnetV2/InceptionResnetV2/Repeat_2/block8_7/Branch_0/Conv2d_1x1/Conv2D',
                                                                                                            in_channels=2080,
                                                                                                            out_channels=192,
                                                                                                            kernel_size=(
                                                                                                                1, 1),
                                                                                                            stride=(
                                                                                                                1, 1),
                                                                                                            groups=1,
                                                                                                            bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_7_Branch_1_Conv2d_0a_1x1_Conv2D = self.__conv(2,
                                                                                                               name='InceptionResnetV2/InceptionResnetV2/Repeat_2/block8_7/Branch_1/Conv2d_0a_1x1/Conv2D',
                                                                                                               in_channels=2080,
                                                                                                               out_channels=192,
                                                                                                               kernel_size=(
                                                                                                                   1,
                                                                                                                   1),
                                                                                                               stride=(
                                                                                                                   1,
                                                                                                                   1),
                                                                                                               groups=1,
                                                                                                               bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_7_Branch_0_Conv2d_1x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionResnetV2/InceptionResnetV2/Repeat_2/block8_7/Branch_0/Conv2d_1x1/BatchNorm/FusedBatchNorm',
            num_features=192, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_7_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionResnetV2/InceptionResnetV2/Repeat_2/block8_7/Branch_1/Conv2d_0a_1x1/BatchNorm/FusedBatchNorm',
            num_features=192, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_7_Branch_1_Conv2d_0b_1x3_Conv2D = self.__conv(2,
                                                                                                               name='InceptionResnetV2/InceptionResnetV2/Repeat_2/block8_7/Branch_1/Conv2d_0b_1x3/Conv2D',
                                                                                                               in_channels=192,
                                                                                                               out_channels=224,
                                                                                                               kernel_size=(
                                                                                                                   1,
                                                                                                                   3),
                                                                                                               stride=(
                                                                                                                   1,
                                                                                                                   1),
                                                                                                               groups=1,
                                                                                                               bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_7_Branch_1_Conv2d_0b_1x3_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionResnetV2/InceptionResnetV2/Repeat_2/block8_7/Branch_1/Conv2d_0b_1x3/BatchNorm/FusedBatchNorm',
            num_features=224, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_7_Branch_1_Conv2d_0c_3x1_Conv2D = self.__conv(2,
                                                                                                               name='InceptionResnetV2/InceptionResnetV2/Repeat_2/block8_7/Branch_1/Conv2d_0c_3x1/Conv2D',
                                                                                                               in_channels=224,
                                                                                                               out_channels=256,
                                                                                                               kernel_size=(
                                                                                                                   3,
                                                                                                                   1),
                                                                                                               stride=(
                                                                                                                   1,
                                                                                                                   1),
                                                                                                               groups=1,
                                                                                                               bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_7_Branch_1_Conv2d_0c_3x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionResnetV2/InceptionResnetV2/Repeat_2/block8_7/Branch_1/Conv2d_0c_3x1/BatchNorm/FusedBatchNorm',
            num_features=256, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_7_Conv2d_1x1_Conv2D = self.__conv(2,
                                                                                                   name='InceptionResnetV2/InceptionResnetV2/Repeat_2/block8_7/Conv2d_1x1/Conv2D',
                                                                                                   in_channels=448,
                                                                                                   out_channels=2080,
                                                                                                   kernel_size=(1, 1),
                                                                                                   stride=(1, 1),
                                                                                                   groups=1, bias=True)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_8_Branch_0_Conv2d_1x1_Conv2D = self.__conv(2,
                                                                                                            name='InceptionResnetV2/InceptionResnetV2/Repeat_2/block8_8/Branch_0/Conv2d_1x1/Conv2D',
                                                                                                            in_channels=2080,
                                                                                                            out_channels=192,
                                                                                                            kernel_size=(
                                                                                                                1, 1),
                                                                                                            stride=(
                                                                                                                1, 1),
                                                                                                            groups=1,
                                                                                                            bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_8_Branch_1_Conv2d_0a_1x1_Conv2D = self.__conv(2,
                                                                                                               name='InceptionResnetV2/InceptionResnetV2/Repeat_2/block8_8/Branch_1/Conv2d_0a_1x1/Conv2D',
                                                                                                               in_channels=2080,
                                                                                                               out_channels=192,
                                                                                                               kernel_size=(
                                                                                                                   1,
                                                                                                                   1),
                                                                                                               stride=(
                                                                                                                   1,
                                                                                                                   1),
                                                                                                               groups=1,
                                                                                                               bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_8_Branch_0_Conv2d_1x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionResnetV2/InceptionResnetV2/Repeat_2/block8_8/Branch_0/Conv2d_1x1/BatchNorm/FusedBatchNorm',
            num_features=192, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_8_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionResnetV2/InceptionResnetV2/Repeat_2/block8_8/Branch_1/Conv2d_0a_1x1/BatchNorm/FusedBatchNorm',
            num_features=192, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_8_Branch_1_Conv2d_0b_1x3_Conv2D = self.__conv(2,
                                                                                                               name='InceptionResnetV2/InceptionResnetV2/Repeat_2/block8_8/Branch_1/Conv2d_0b_1x3/Conv2D',
                                                                                                               in_channels=192,
                                                                                                               out_channels=224,
                                                                                                               kernel_size=(
                                                                                                                   1,
                                                                                                                   3),
                                                                                                               stride=(
                                                                                                                   1,
                                                                                                                   1),
                                                                                                               groups=1,
                                                                                                               bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_8_Branch_1_Conv2d_0b_1x3_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionResnetV2/InceptionResnetV2/Repeat_2/block8_8/Branch_1/Conv2d_0b_1x3/BatchNorm/FusedBatchNorm',
            num_features=224, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_8_Branch_1_Conv2d_0c_3x1_Conv2D = self.__conv(2,
                                                                                                               name='InceptionResnetV2/InceptionResnetV2/Repeat_2/block8_8/Branch_1/Conv2d_0c_3x1/Conv2D',
                                                                                                               in_channels=224,
                                                                                                               out_channels=256,
                                                                                                               kernel_size=(
                                                                                                                   3,
                                                                                                                   1),
                                                                                                               stride=(
                                                                                                                   1,
                                                                                                                   1),
                                                                                                               groups=1,
                                                                                                               bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_8_Branch_1_Conv2d_0c_3x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionResnetV2/InceptionResnetV2/Repeat_2/block8_8/Branch_1/Conv2d_0c_3x1/BatchNorm/FusedBatchNorm',
            num_features=256, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_8_Conv2d_1x1_Conv2D = self.__conv(2,
                                                                                                   name='InceptionResnetV2/InceptionResnetV2/Repeat_2/block8_8/Conv2d_1x1/Conv2D',
                                                                                                   in_channels=448,
                                                                                                   out_channels=2080,
                                                                                                   kernel_size=(1, 1),
                                                                                                   stride=(1, 1),
                                                                                                   groups=1, bias=True)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_9_Branch_0_Conv2d_1x1_Conv2D = self.__conv(2,
                                                                                                            name='InceptionResnetV2/InceptionResnetV2/Repeat_2/block8_9/Branch_0/Conv2d_1x1/Conv2D',
                                                                                                            in_channels=2080,
                                                                                                            out_channels=192,
                                                                                                            kernel_size=(
                                                                                                                1, 1),
                                                                                                            stride=(
                                                                                                                1, 1),
                                                                                                            groups=1,
                                                                                                            bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_9_Branch_1_Conv2d_0a_1x1_Conv2D = self.__conv(2,
                                                                                                               name='InceptionResnetV2/InceptionResnetV2/Repeat_2/block8_9/Branch_1/Conv2d_0a_1x1/Conv2D',
                                                                                                               in_channels=2080,
                                                                                                               out_channels=192,
                                                                                                               kernel_size=(
                                                                                                                   1,
                                                                                                                   1),
                                                                                                               stride=(
                                                                                                                   1,
                                                                                                                   1),
                                                                                                               groups=1,
                                                                                                               bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_9_Branch_0_Conv2d_1x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionResnetV2/InceptionResnetV2/Repeat_2/block8_9/Branch_0/Conv2d_1x1/BatchNorm/FusedBatchNorm',
            num_features=192, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_9_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionResnetV2/InceptionResnetV2/Repeat_2/block8_9/Branch_1/Conv2d_0a_1x1/BatchNorm/FusedBatchNorm',
            num_features=192, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_9_Branch_1_Conv2d_0b_1x3_Conv2D = self.__conv(2,
                                                                                                               name='InceptionResnetV2/InceptionResnetV2/Repeat_2/block8_9/Branch_1/Conv2d_0b_1x3/Conv2D',
                                                                                                               in_channels=192,
                                                                                                               out_channels=224,
                                                                                                               kernel_size=(
                                                                                                                   1,
                                                                                                                   3),
                                                                                                               stride=(
                                                                                                                   1,
                                                                                                                   1),
                                                                                                               groups=1,
                                                                                                               bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_9_Branch_1_Conv2d_0b_1x3_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionResnetV2/InceptionResnetV2/Repeat_2/block8_9/Branch_1/Conv2d_0b_1x3/BatchNorm/FusedBatchNorm',
            num_features=224, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_9_Branch_1_Conv2d_0c_3x1_Conv2D = self.__conv(2,
                                                                                                               name='InceptionResnetV2/InceptionResnetV2/Repeat_2/block8_9/Branch_1/Conv2d_0c_3x1/Conv2D',
                                                                                                               in_channels=224,
                                                                                                               out_channels=256,
                                                                                                               kernel_size=(
                                                                                                                   3,
                                                                                                                   1),
                                                                                                               stride=(
                                                                                                                   1,
                                                                                                                   1),
                                                                                                               groups=1,
                                                                                                               bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_9_Branch_1_Conv2d_0c_3x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionResnetV2/InceptionResnetV2/Repeat_2/block8_9/Branch_1/Conv2d_0c_3x1/BatchNorm/FusedBatchNorm',
            num_features=256, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_9_Conv2d_1x1_Conv2D = self.__conv(2,
                                                                                                   name='InceptionResnetV2/InceptionResnetV2/Repeat_2/block8_9/Conv2d_1x1/Conv2D',
                                                                                                   in_channels=448,
                                                                                                   out_channels=2080,
                                                                                                   kernel_size=(1, 1),
                                                                                                   stride=(1, 1),
                                                                                                   groups=1, bias=True)
        self.InceptionResnetV2_InceptionResnetV2_Block8_Branch_0_Conv2d_1x1_Conv2D = self.__conv(2,
                                                                                                 name='InceptionResnetV2/InceptionResnetV2/Block8/Branch_0/Conv2d_1x1/Conv2D',
                                                                                                 in_channels=2080,
                                                                                                 out_channels=192,
                                                                                                 kernel_size=(1, 1),
                                                                                                 stride=(1, 1),
                                                                                                 groups=1, bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Block8_Branch_1_Conv2d_0a_1x1_Conv2D = self.__conv(2,
                                                                                                    name='InceptionResnetV2/InceptionResnetV2/Block8/Branch_1/Conv2d_0a_1x1/Conv2D',
                                                                                                    in_channels=2080,
                                                                                                    out_channels=192,
                                                                                                    kernel_size=(1, 1),
                                                                                                    stride=(1, 1),
                                                                                                    groups=1, bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Block8_Branch_0_Conv2d_1x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionResnetV2/InceptionResnetV2/Block8/Branch_0/Conv2d_1x1/BatchNorm/FusedBatchNorm',
            num_features=192, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Block8_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionResnetV2/InceptionResnetV2/Block8/Branch_1/Conv2d_0a_1x1/BatchNorm/FusedBatchNorm',
            num_features=192, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Block8_Branch_1_Conv2d_0b_1x3_Conv2D = self.__conv(2,
                                                                                                    name='InceptionResnetV2/InceptionResnetV2/Block8/Branch_1/Conv2d_0b_1x3/Conv2D',
                                                                                                    in_channels=192,
                                                                                                    out_channels=224,
                                                                                                    kernel_size=(1, 3),
                                                                                                    stride=(1, 1),
                                                                                                    groups=1, bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Block8_Branch_1_Conv2d_0b_1x3_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionResnetV2/InceptionResnetV2/Block8/Branch_1/Conv2d_0b_1x3/BatchNorm/FusedBatchNorm',
            num_features=224, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Block8_Branch_1_Conv2d_0c_3x1_Conv2D = self.__conv(2,
                                                                                                    name='InceptionResnetV2/InceptionResnetV2/Block8/Branch_1/Conv2d_0c_3x1/Conv2D',
                                                                                                    in_channels=224,
                                                                                                    out_channels=256,
                                                                                                    kernel_size=(3, 1),
                                                                                                    stride=(1, 1),
                                                                                                    groups=1, bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Block8_Branch_1_Conv2d_0c_3x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionResnetV2/InceptionResnetV2/Block8/Branch_1/Conv2d_0c_3x1/BatchNorm/FusedBatchNorm',
            num_features=256, eps=0.0010000000474974513, momentum=0.0)
        self.InceptionResnetV2_InceptionResnetV2_Block8_Conv2d_1x1_Conv2D = self.__conv(2,
                                                                                        name='InceptionResnetV2/InceptionResnetV2/Block8/Conv2d_1x1/Conv2D',
                                                                                        in_channels=448,
                                                                                        out_channels=2080,
                                                                                        kernel_size=(1, 1),
                                                                                        stride=(1, 1), groups=1,
                                                                                        bias=True)
        self.InceptionResnetV2_InceptionResnetV2_Conv2d_7b_1x1_Conv2D = self.__conv(2,
                                                                                    name='InceptionResnetV2/InceptionResnetV2/Conv2d_7b_1x1/Conv2D',
                                                                                    in_channels=2080, out_channels=1536,
                                                                                    kernel_size=(1, 1), stride=(1, 1),
                                                                                    groups=1, bias=None)
        self.InceptionResnetV2_InceptionResnetV2_Conv2d_7b_1x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(2,
                                                                                                                     'InceptionResnetV2/InceptionResnetV2/Conv2d_7b_1x1/BatchNorm/FusedBatchNorm',
                                                                                                                     num_features=1536,
                                                                                                                     eps=0.0010000000474974513,
                                                                                                                     momentum=0.0)
        self.InceptionResnetV2_Logits_Logits_MatMul = self.__dense(name='InceptionResnetV2/Logits/Logits/MatMul',
                                                                   in_features=1536, out_features=1001, bias=True)

        from torch.nn.parameter import Parameter
        self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_1_mul_x = Parameter(
            torch.autograd.Variable(torch.Tensor([0.17000000178813934]), requires_grad=False))
        self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_2_mul_x = Parameter(
            torch.autograd.Variable(torch.Tensor([0.17000000178813934]), requires_grad=False))
        self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_3_mul_x = Parameter(
            torch.autograd.Variable(torch.Tensor([0.17000000178813934]), requires_grad=False))
        self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_4_mul_x = Parameter(
            torch.autograd.Variable(torch.Tensor([0.17000000178813934]), requires_grad=False))
        self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_5_mul_x = Parameter(
            torch.autograd.Variable(torch.Tensor([0.17000000178813934]), requires_grad=False))
        self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_6_mul_x = Parameter(
            torch.autograd.Variable(torch.Tensor([0.17000000178813934]), requires_grad=False))
        self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_7_mul_x = Parameter(
            torch.autograd.Variable(torch.Tensor([0.17000000178813934]), requires_grad=False))
        self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_8_mul_x = Parameter(
            torch.autograd.Variable(torch.Tensor([0.17000000178813934]), requires_grad=False))
        self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_9_mul_x = Parameter(
            torch.autograd.Variable(torch.Tensor([0.17000000178813934]), requires_grad=False))
        self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_10_mul_x = Parameter(
            torch.autograd.Variable(torch.Tensor([0.17000000178813934]), requires_grad=False))
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_1_mul_x = Parameter(
            torch.autograd.Variable(torch.Tensor([0.10000000149011612]), requires_grad=False))
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_2_mul_x = Parameter(
            torch.autograd.Variable(torch.Tensor([0.10000000149011612]), requires_grad=False))
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_3_mul_x = Parameter(
            torch.autograd.Variable(torch.Tensor([0.10000000149011612]), requires_grad=False))
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_4_mul_x = Parameter(
            torch.autograd.Variable(torch.Tensor([0.10000000149011612]), requires_grad=False))
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_5_mul_x = Parameter(
            torch.autograd.Variable(torch.Tensor([0.10000000149011612]), requires_grad=False))
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_6_mul_x = Parameter(
            torch.autograd.Variable(torch.Tensor([0.10000000149011612]), requires_grad=False))
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_7_mul_x = Parameter(
            torch.autograd.Variable(torch.Tensor([0.10000000149011612]), requires_grad=False))
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_8_mul_x = Parameter(
            torch.autograd.Variable(torch.Tensor([0.10000000149011612]), requires_grad=False))
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_9_mul_x = Parameter(
            torch.autograd.Variable(torch.Tensor([0.10000000149011612]), requires_grad=False))
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_10_mul_x = Parameter(
            torch.autograd.Variable(torch.Tensor([0.10000000149011612]), requires_grad=False))
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_11_mul_x = Parameter(
            torch.autograd.Variable(torch.Tensor([0.10000000149011612]), requires_grad=False))
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_12_mul_x = Parameter(
            torch.autograd.Variable(torch.Tensor([0.10000000149011612]), requires_grad=False))
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_13_mul_x = Parameter(
            torch.autograd.Variable(torch.Tensor([0.10000000149011612]), requires_grad=False))
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_14_mul_x = Parameter(
            torch.autograd.Variable(torch.Tensor([0.10000000149011612]), requires_grad=False))
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_15_mul_x = Parameter(
            torch.autograd.Variable(torch.Tensor([0.10000000149011612]), requires_grad=False))
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_16_mul_x = Parameter(
            torch.autograd.Variable(torch.Tensor([0.10000000149011612]), requires_grad=False))
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_17_mul_x = Parameter(
            torch.autograd.Variable(torch.Tensor([0.10000000149011612]), requires_grad=False))
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_18_mul_x = Parameter(
            torch.autograd.Variable(torch.Tensor([0.10000000149011612]), requires_grad=False))
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_19_mul_x = Parameter(
            torch.autograd.Variable(torch.Tensor([0.10000000149011612]), requires_grad=False))
        self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_20_mul_x = Parameter(
            torch.autograd.Variable(torch.Tensor([0.10000000149011612]), requires_grad=False))
        self.InceptionResnetV2_AuxLogits_Flatten_flatten_Reshape_shape_1 = Parameter(
            torch.autograd.Variable(torch.Tensor([-1]), requires_grad=False))
        self.InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_1_mul_x = Parameter(
            torch.autograd.Variable(torch.Tensor([0.20000000298023224]), requires_grad=False))
        self.InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_2_mul_x = Parameter(
            torch.autograd.Variable(torch.Tensor([0.20000000298023224]), requires_grad=False))
        self.InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_3_mul_x = Parameter(
            torch.autograd.Variable(torch.Tensor([0.20000000298023224]), requires_grad=False))
        self.InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_4_mul_x = Parameter(
            torch.autograd.Variable(torch.Tensor([0.20000000298023224]), requires_grad=False))
        self.InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_5_mul_x = Parameter(
            torch.autograd.Variable(torch.Tensor([0.20000000298023224]), requires_grad=False))
        self.InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_6_mul_x = Parameter(
            torch.autograd.Variable(torch.Tensor([0.20000000298023224]), requires_grad=False))
        self.InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_7_mul_x = Parameter(
            torch.autograd.Variable(torch.Tensor([0.20000000298023224]), requires_grad=False))
        self.InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_8_mul_x = Parameter(
            torch.autograd.Variable(torch.Tensor([0.20000000298023224]), requires_grad=False))
        self.InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_9_mul_x = Parameter(
            torch.autograd.Variable(torch.Tensor([0.20000000298023224]), requires_grad=False))
        self.InceptionResnetV2_InceptionResnetV2_Block8_mul_x = Parameter(
            torch.autograd.Variable(torch.Tensor([1.0]), requires_grad=False))
        self.InceptionResnetV2_Logits_Flatten_flatten_Reshape_shape_1 = Parameter(
            torch.autograd.Variable(torch.Tensor([-1]), requires_grad=False))

    def forward(self, x):
        InceptionResnetV2_InceptionResnetV2_Conv2d_1a_3x3_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Conv2d_1a_3x3_Conv2D(
            x)
        InceptionResnetV2_InceptionResnetV2_Conv2d_1a_3x3_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Conv2d_1a_3x3_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Conv2d_1a_3x3_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Conv2d_1a_3x3_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Conv2d_1a_3x3_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Conv2d_2a_3x3_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Conv2d_2a_3x3_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Conv2d_1a_3x3_Relu)
        InceptionResnetV2_InceptionResnetV2_Conv2d_2a_3x3_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Conv2d_2a_3x3_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Conv2d_2a_3x3_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Conv2d_2a_3x3_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Conv2d_2a_3x3_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Conv2d_2b_3x3_Conv2D_pad = F.pad(
            InceptionResnetV2_InceptionResnetV2_Conv2d_2a_3x3_Relu, (1, 1, 1, 1))
        InceptionResnetV2_InceptionResnetV2_Conv2d_2b_3x3_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Conv2d_2b_3x3_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Conv2d_2b_3x3_Conv2D_pad)
        InceptionResnetV2_InceptionResnetV2_Conv2d_2b_3x3_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Conv2d_2b_3x3_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Conv2d_2b_3x3_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Conv2d_2b_3x3_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Conv2d_2b_3x3_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_MaxPool_3a_3x3_MaxPool, InceptionResnetV2_InceptionResnetV2_MaxPool_3a_3x3_MaxPool_idx = F.max_pool2d(
            InceptionResnetV2_InceptionResnetV2_Conv2d_2b_3x3_Relu, kernel_size=(3, 3), stride=(2, 2), padding=0,
            ceil_mode=False, return_indices=True)
        InceptionResnetV2_InceptionResnetV2_Conv2d_3b_1x1_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Conv2d_3b_1x1_Conv2D(
            InceptionResnetV2_InceptionResnetV2_MaxPool_3a_3x3_MaxPool)
        InceptionResnetV2_InceptionResnetV2_Conv2d_3b_1x1_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Conv2d_3b_1x1_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Conv2d_3b_1x1_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Conv2d_3b_1x1_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Conv2d_3b_1x1_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Conv2d_4a_3x3_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Conv2d_4a_3x3_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Conv2d_3b_1x1_Relu)
        InceptionResnetV2_InceptionResnetV2_Conv2d_4a_3x3_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Conv2d_4a_3x3_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Conv2d_4a_3x3_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Conv2d_4a_3x3_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Conv2d_4a_3x3_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_MaxPool_5a_3x3_MaxPool, InceptionResnetV2_InceptionResnetV2_MaxPool_5a_3x3_MaxPool_idx = F.max_pool2d(
            InceptionResnetV2_InceptionResnetV2_Conv2d_4a_3x3_Relu, kernel_size=(3, 3), stride=(2, 2), padding=0,
            ceil_mode=False, return_indices=True)
        InceptionResnetV2_InceptionResnetV2_Mixed_5b_Branch_0_Conv2d_1x1_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Mixed_5b_Branch_0_Conv2d_1x1_Conv2D(
            InceptionResnetV2_InceptionResnetV2_MaxPool_5a_3x3_MaxPool)
        InceptionResnetV2_InceptionResnetV2_Mixed_5b_Branch_1_Conv2d_0a_1x1_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Mixed_5b_Branch_1_Conv2d_0a_1x1_Conv2D(
            InceptionResnetV2_InceptionResnetV2_MaxPool_5a_3x3_MaxPool)
        InceptionResnetV2_InceptionResnetV2_Mixed_5b_Branch_2_Conv2d_0a_1x1_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Mixed_5b_Branch_2_Conv2d_0a_1x1_Conv2D(
            InceptionResnetV2_InceptionResnetV2_MaxPool_5a_3x3_MaxPool)
        InceptionResnetV2_InceptionResnetV2_Mixed_5b_Branch_3_AvgPool_0a_3x3_AvgPool = F.avg_pool2d(
            InceptionResnetV2_InceptionResnetV2_MaxPool_5a_3x3_MaxPool, kernel_size=(3, 3), stride=(1, 1), padding=(1,),
            ceil_mode=False, count_include_pad=False)
        InceptionResnetV2_InceptionResnetV2_Mixed_5b_Branch_0_Conv2d_1x1_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Mixed_5b_Branch_0_Conv2d_1x1_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Mixed_5b_Branch_0_Conv2d_1x1_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Mixed_5b_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Mixed_5b_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Mixed_5b_Branch_1_Conv2d_0a_1x1_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Mixed_5b_Branch_2_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Mixed_5b_Branch_2_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Mixed_5b_Branch_2_Conv2d_0a_1x1_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Mixed_5b_Branch_3_Conv2d_0b_1x1_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Mixed_5b_Branch_3_Conv2d_0b_1x1_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Mixed_5b_Branch_3_AvgPool_0a_3x3_AvgPool)
        InceptionResnetV2_InceptionResnetV2_Mixed_5b_Branch_0_Conv2d_1x1_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Mixed_5b_Branch_0_Conv2d_1x1_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Mixed_5b_Branch_1_Conv2d_0a_1x1_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Mixed_5b_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Mixed_5b_Branch_2_Conv2d_0a_1x1_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Mixed_5b_Branch_2_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Mixed_5b_Branch_3_Conv2d_0b_1x1_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Mixed_5b_Branch_3_Conv2d_0b_1x1_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Mixed_5b_Branch_3_Conv2d_0b_1x1_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Mixed_5b_Branch_1_Conv2d_0b_5x5_Conv2D_pad = F.pad(
            InceptionResnetV2_InceptionResnetV2_Mixed_5b_Branch_1_Conv2d_0a_1x1_Relu, (2, 2, 2, 2))
        InceptionResnetV2_InceptionResnetV2_Mixed_5b_Branch_1_Conv2d_0b_5x5_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Mixed_5b_Branch_1_Conv2d_0b_5x5_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Mixed_5b_Branch_1_Conv2d_0b_5x5_Conv2D_pad)
        InceptionResnetV2_InceptionResnetV2_Mixed_5b_Branch_2_Conv2d_0b_3x3_Conv2D_pad = F.pad(
            InceptionResnetV2_InceptionResnetV2_Mixed_5b_Branch_2_Conv2d_0a_1x1_Relu, (1, 1, 1, 1))
        InceptionResnetV2_InceptionResnetV2_Mixed_5b_Branch_2_Conv2d_0b_3x3_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Mixed_5b_Branch_2_Conv2d_0b_3x3_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Mixed_5b_Branch_2_Conv2d_0b_3x3_Conv2D_pad)
        InceptionResnetV2_InceptionResnetV2_Mixed_5b_Branch_3_Conv2d_0b_1x1_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Mixed_5b_Branch_3_Conv2d_0b_1x1_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Mixed_5b_Branch_1_Conv2d_0b_5x5_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Mixed_5b_Branch_1_Conv2d_0b_5x5_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Mixed_5b_Branch_1_Conv2d_0b_5x5_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Mixed_5b_Branch_2_Conv2d_0b_3x3_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Mixed_5b_Branch_2_Conv2d_0b_3x3_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Mixed_5b_Branch_2_Conv2d_0b_3x3_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Mixed_5b_Branch_1_Conv2d_0b_5x5_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Mixed_5b_Branch_1_Conv2d_0b_5x5_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Mixed_5b_Branch_2_Conv2d_0b_3x3_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Mixed_5b_Branch_2_Conv2d_0b_3x3_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Mixed_5b_Branch_2_Conv2d_0c_3x3_Conv2D_pad = F.pad(
            InceptionResnetV2_InceptionResnetV2_Mixed_5b_Branch_2_Conv2d_0b_3x3_Relu, (1, 1, 1, 1))
        InceptionResnetV2_InceptionResnetV2_Mixed_5b_Branch_2_Conv2d_0c_3x3_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Mixed_5b_Branch_2_Conv2d_0c_3x3_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Mixed_5b_Branch_2_Conv2d_0c_3x3_Conv2D_pad)
        InceptionResnetV2_InceptionResnetV2_Mixed_5b_Branch_2_Conv2d_0c_3x3_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Mixed_5b_Branch_2_Conv2d_0c_3x3_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Mixed_5b_Branch_2_Conv2d_0c_3x3_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Mixed_5b_Branch_2_Conv2d_0c_3x3_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Mixed_5b_Branch_2_Conv2d_0c_3x3_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Mixed_5b_concat = torch.cat((
            InceptionResnetV2_InceptionResnetV2_Mixed_5b_Branch_0_Conv2d_1x1_Relu,
            InceptionResnetV2_InceptionResnetV2_Mixed_5b_Branch_1_Conv2d_0b_5x5_Relu,
            InceptionResnetV2_InceptionResnetV2_Mixed_5b_Branch_2_Conv2d_0c_3x3_Relu,
            InceptionResnetV2_InceptionResnetV2_Mixed_5b_Branch_3_Conv2d_0b_1x1_Relu,),
            1)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_1_Branch_0_Conv2d_1x1_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_1_Branch_0_Conv2d_1x1_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Mixed_5b_concat)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_1_Branch_1_Conv2d_0a_1x1_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_1_Branch_1_Conv2d_0a_1x1_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Mixed_5b_concat)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_1_Branch_2_Conv2d_0a_1x1_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_1_Branch_2_Conv2d_0a_1x1_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Mixed_5b_concat)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_1_Branch_0_Conv2d_1x1_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_1_Branch_0_Conv2d_1x1_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_1_Branch_0_Conv2d_1x1_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_1_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_1_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_1_Branch_1_Conv2d_0a_1x1_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_1_Branch_2_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_1_Branch_2_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_1_Branch_2_Conv2d_0a_1x1_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_1_Branch_0_Conv2d_1x1_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_1_Branch_0_Conv2d_1x1_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_1_Branch_1_Conv2d_0a_1x1_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_1_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_1_Branch_2_Conv2d_0a_1x1_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_1_Branch_2_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_1_Branch_1_Conv2d_0b_3x3_Conv2D_pad = F.pad(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_1_Branch_1_Conv2d_0a_1x1_Relu, (1, 1, 1, 1))
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_1_Branch_1_Conv2d_0b_3x3_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_1_Branch_1_Conv2d_0b_3x3_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_1_Branch_1_Conv2d_0b_3x3_Conv2D_pad)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_1_Branch_2_Conv2d_0b_3x3_Conv2D_pad = F.pad(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_1_Branch_2_Conv2d_0a_1x1_Relu, (1, 1, 1, 1))
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_1_Branch_2_Conv2d_0b_3x3_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_1_Branch_2_Conv2d_0b_3x3_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_1_Branch_2_Conv2d_0b_3x3_Conv2D_pad)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_1_Branch_1_Conv2d_0b_3x3_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_1_Branch_1_Conv2d_0b_3x3_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_1_Branch_1_Conv2d_0b_3x3_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_1_Branch_2_Conv2d_0b_3x3_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_1_Branch_2_Conv2d_0b_3x3_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_1_Branch_2_Conv2d_0b_3x3_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_1_Branch_1_Conv2d_0b_3x3_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_1_Branch_1_Conv2d_0b_3x3_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_1_Branch_2_Conv2d_0b_3x3_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_1_Branch_2_Conv2d_0b_3x3_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_1_Branch_2_Conv2d_0c_3x3_Conv2D_pad = F.pad(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_1_Branch_2_Conv2d_0b_3x3_Relu, (1, 1, 1, 1))
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_1_Branch_2_Conv2d_0c_3x3_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_1_Branch_2_Conv2d_0c_3x3_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_1_Branch_2_Conv2d_0c_3x3_Conv2D_pad)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_1_Branch_2_Conv2d_0c_3x3_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_1_Branch_2_Conv2d_0c_3x3_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_1_Branch_2_Conv2d_0c_3x3_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_1_Branch_2_Conv2d_0c_3x3_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_1_Branch_2_Conv2d_0c_3x3_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_1_concat = torch.cat((
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_1_Branch_0_Conv2d_1x1_Relu,
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_1_Branch_1_Conv2d_0b_3x3_Relu,
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_1_Branch_2_Conv2d_0c_3x3_Relu,),
            1)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_1_Conv2d_1x1_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_1_Conv2d_1x1_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_1_concat)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_1_mul = self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_1_mul_x * InceptionResnetV2_InceptionResnetV2_Repeat_block35_1_Conv2d_1x1_Conv2D
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_1_add = InceptionResnetV2_InceptionResnetV2_Mixed_5b_concat + InceptionResnetV2_InceptionResnetV2_Repeat_block35_1_mul
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_1_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_1_add)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_2_Branch_0_Conv2d_1x1_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_2_Branch_0_Conv2d_1x1_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_1_Relu)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_2_Branch_1_Conv2d_0a_1x1_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_2_Branch_1_Conv2d_0a_1x1_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_1_Relu)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_2_Branch_2_Conv2d_0a_1x1_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_2_Branch_2_Conv2d_0a_1x1_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_1_Relu)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_2_Branch_0_Conv2d_1x1_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_2_Branch_0_Conv2d_1x1_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_2_Branch_0_Conv2d_1x1_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_2_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_2_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_2_Branch_1_Conv2d_0a_1x1_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_2_Branch_2_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_2_Branch_2_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_2_Branch_2_Conv2d_0a_1x1_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_2_Branch_0_Conv2d_1x1_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_2_Branch_0_Conv2d_1x1_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_2_Branch_1_Conv2d_0a_1x1_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_2_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_2_Branch_2_Conv2d_0a_1x1_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_2_Branch_2_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_2_Branch_1_Conv2d_0b_3x3_Conv2D_pad = F.pad(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_2_Branch_1_Conv2d_0a_1x1_Relu, (1, 1, 1, 1))
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_2_Branch_1_Conv2d_0b_3x3_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_2_Branch_1_Conv2d_0b_3x3_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_2_Branch_1_Conv2d_0b_3x3_Conv2D_pad)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_2_Branch_2_Conv2d_0b_3x3_Conv2D_pad = F.pad(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_2_Branch_2_Conv2d_0a_1x1_Relu, (1, 1, 1, 1))
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_2_Branch_2_Conv2d_0b_3x3_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_2_Branch_2_Conv2d_0b_3x3_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_2_Branch_2_Conv2d_0b_3x3_Conv2D_pad)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_2_Branch_1_Conv2d_0b_3x3_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_2_Branch_1_Conv2d_0b_3x3_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_2_Branch_1_Conv2d_0b_3x3_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_2_Branch_2_Conv2d_0b_3x3_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_2_Branch_2_Conv2d_0b_3x3_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_2_Branch_2_Conv2d_0b_3x3_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_2_Branch_1_Conv2d_0b_3x3_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_2_Branch_1_Conv2d_0b_3x3_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_2_Branch_2_Conv2d_0b_3x3_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_2_Branch_2_Conv2d_0b_3x3_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_2_Branch_2_Conv2d_0c_3x3_Conv2D_pad = F.pad(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_2_Branch_2_Conv2d_0b_3x3_Relu, (1, 1, 1, 1))
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_2_Branch_2_Conv2d_0c_3x3_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_2_Branch_2_Conv2d_0c_3x3_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_2_Branch_2_Conv2d_0c_3x3_Conv2D_pad)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_2_Branch_2_Conv2d_0c_3x3_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_2_Branch_2_Conv2d_0c_3x3_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_2_Branch_2_Conv2d_0c_3x3_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_2_Branch_2_Conv2d_0c_3x3_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_2_Branch_2_Conv2d_0c_3x3_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_2_concat = torch.cat((
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_2_Branch_0_Conv2d_1x1_Relu,
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_2_Branch_1_Conv2d_0b_3x3_Relu,
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_2_Branch_2_Conv2d_0c_3x3_Relu,),
            1)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_2_Conv2d_1x1_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_2_Conv2d_1x1_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_2_concat)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_2_mul = self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_2_mul_x * InceptionResnetV2_InceptionResnetV2_Repeat_block35_2_Conv2d_1x1_Conv2D
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_2_add = InceptionResnetV2_InceptionResnetV2_Repeat_block35_1_Relu + InceptionResnetV2_InceptionResnetV2_Repeat_block35_2_mul
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_2_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_2_add)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_3_Branch_0_Conv2d_1x1_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_3_Branch_0_Conv2d_1x1_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_2_Relu)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_3_Branch_1_Conv2d_0a_1x1_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_3_Branch_1_Conv2d_0a_1x1_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_2_Relu)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_3_Branch_2_Conv2d_0a_1x1_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_3_Branch_2_Conv2d_0a_1x1_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_2_Relu)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_3_Branch_0_Conv2d_1x1_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_3_Branch_0_Conv2d_1x1_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_3_Branch_0_Conv2d_1x1_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_3_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_3_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_3_Branch_1_Conv2d_0a_1x1_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_3_Branch_2_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_3_Branch_2_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_3_Branch_2_Conv2d_0a_1x1_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_3_Branch_0_Conv2d_1x1_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_3_Branch_0_Conv2d_1x1_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_3_Branch_1_Conv2d_0a_1x1_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_3_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_3_Branch_2_Conv2d_0a_1x1_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_3_Branch_2_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_3_Branch_1_Conv2d_0b_3x3_Conv2D_pad = F.pad(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_3_Branch_1_Conv2d_0a_1x1_Relu, (1, 1, 1, 1))
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_3_Branch_1_Conv2d_0b_3x3_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_3_Branch_1_Conv2d_0b_3x3_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_3_Branch_1_Conv2d_0b_3x3_Conv2D_pad)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_3_Branch_2_Conv2d_0b_3x3_Conv2D_pad = F.pad(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_3_Branch_2_Conv2d_0a_1x1_Relu, (1, 1, 1, 1))
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_3_Branch_2_Conv2d_0b_3x3_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_3_Branch_2_Conv2d_0b_3x3_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_3_Branch_2_Conv2d_0b_3x3_Conv2D_pad)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_3_Branch_1_Conv2d_0b_3x3_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_3_Branch_1_Conv2d_0b_3x3_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_3_Branch_1_Conv2d_0b_3x3_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_3_Branch_2_Conv2d_0b_3x3_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_3_Branch_2_Conv2d_0b_3x3_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_3_Branch_2_Conv2d_0b_3x3_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_3_Branch_1_Conv2d_0b_3x3_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_3_Branch_1_Conv2d_0b_3x3_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_3_Branch_2_Conv2d_0b_3x3_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_3_Branch_2_Conv2d_0b_3x3_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_3_Branch_2_Conv2d_0c_3x3_Conv2D_pad = F.pad(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_3_Branch_2_Conv2d_0b_3x3_Relu, (1, 1, 1, 1))
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_3_Branch_2_Conv2d_0c_3x3_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_3_Branch_2_Conv2d_0c_3x3_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_3_Branch_2_Conv2d_0c_3x3_Conv2D_pad)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_3_Branch_2_Conv2d_0c_3x3_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_3_Branch_2_Conv2d_0c_3x3_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_3_Branch_2_Conv2d_0c_3x3_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_3_Branch_2_Conv2d_0c_3x3_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_3_Branch_2_Conv2d_0c_3x3_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_3_concat = torch.cat((
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_3_Branch_0_Conv2d_1x1_Relu,
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_3_Branch_1_Conv2d_0b_3x3_Relu,
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_3_Branch_2_Conv2d_0c_3x3_Relu,),
            1)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_3_Conv2d_1x1_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_3_Conv2d_1x1_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_3_concat)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_3_mul = self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_3_mul_x * InceptionResnetV2_InceptionResnetV2_Repeat_block35_3_Conv2d_1x1_Conv2D
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_3_add = InceptionResnetV2_InceptionResnetV2_Repeat_block35_2_Relu + InceptionResnetV2_InceptionResnetV2_Repeat_block35_3_mul
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_3_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_3_add)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_4_Branch_0_Conv2d_1x1_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_4_Branch_0_Conv2d_1x1_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_3_Relu)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_4_Branch_1_Conv2d_0a_1x1_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_4_Branch_1_Conv2d_0a_1x1_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_3_Relu)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_4_Branch_2_Conv2d_0a_1x1_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_4_Branch_2_Conv2d_0a_1x1_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_3_Relu)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_4_Branch_0_Conv2d_1x1_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_4_Branch_0_Conv2d_1x1_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_4_Branch_0_Conv2d_1x1_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_4_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_4_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_4_Branch_1_Conv2d_0a_1x1_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_4_Branch_2_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_4_Branch_2_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_4_Branch_2_Conv2d_0a_1x1_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_4_Branch_0_Conv2d_1x1_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_4_Branch_0_Conv2d_1x1_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_4_Branch_1_Conv2d_0a_1x1_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_4_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_4_Branch_2_Conv2d_0a_1x1_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_4_Branch_2_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_4_Branch_1_Conv2d_0b_3x3_Conv2D_pad = F.pad(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_4_Branch_1_Conv2d_0a_1x1_Relu, (1, 1, 1, 1))
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_4_Branch_1_Conv2d_0b_3x3_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_4_Branch_1_Conv2d_0b_3x3_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_4_Branch_1_Conv2d_0b_3x3_Conv2D_pad)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_4_Branch_2_Conv2d_0b_3x3_Conv2D_pad = F.pad(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_4_Branch_2_Conv2d_0a_1x1_Relu, (1, 1, 1, 1))
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_4_Branch_2_Conv2d_0b_3x3_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_4_Branch_2_Conv2d_0b_3x3_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_4_Branch_2_Conv2d_0b_3x3_Conv2D_pad)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_4_Branch_1_Conv2d_0b_3x3_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_4_Branch_1_Conv2d_0b_3x3_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_4_Branch_1_Conv2d_0b_3x3_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_4_Branch_2_Conv2d_0b_3x3_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_4_Branch_2_Conv2d_0b_3x3_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_4_Branch_2_Conv2d_0b_3x3_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_4_Branch_1_Conv2d_0b_3x3_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_4_Branch_1_Conv2d_0b_3x3_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_4_Branch_2_Conv2d_0b_3x3_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_4_Branch_2_Conv2d_0b_3x3_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_4_Branch_2_Conv2d_0c_3x3_Conv2D_pad = F.pad(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_4_Branch_2_Conv2d_0b_3x3_Relu, (1, 1, 1, 1))
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_4_Branch_2_Conv2d_0c_3x3_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_4_Branch_2_Conv2d_0c_3x3_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_4_Branch_2_Conv2d_0c_3x3_Conv2D_pad)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_4_Branch_2_Conv2d_0c_3x3_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_4_Branch_2_Conv2d_0c_3x3_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_4_Branch_2_Conv2d_0c_3x3_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_4_Branch_2_Conv2d_0c_3x3_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_4_Branch_2_Conv2d_0c_3x3_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_4_concat = torch.cat((
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_4_Branch_0_Conv2d_1x1_Relu,
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_4_Branch_1_Conv2d_0b_3x3_Relu,
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_4_Branch_2_Conv2d_0c_3x3_Relu,),
            1)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_4_Conv2d_1x1_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_4_Conv2d_1x1_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_4_concat)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_4_mul = self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_4_mul_x * InceptionResnetV2_InceptionResnetV2_Repeat_block35_4_Conv2d_1x1_Conv2D
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_4_add = InceptionResnetV2_InceptionResnetV2_Repeat_block35_3_Relu + InceptionResnetV2_InceptionResnetV2_Repeat_block35_4_mul
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_4_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_4_add)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_5_Branch_0_Conv2d_1x1_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_5_Branch_0_Conv2d_1x1_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_4_Relu)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_5_Branch_1_Conv2d_0a_1x1_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_5_Branch_1_Conv2d_0a_1x1_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_4_Relu)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_5_Branch_2_Conv2d_0a_1x1_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_5_Branch_2_Conv2d_0a_1x1_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_4_Relu)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_5_Branch_0_Conv2d_1x1_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_5_Branch_0_Conv2d_1x1_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_5_Branch_0_Conv2d_1x1_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_5_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_5_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_5_Branch_1_Conv2d_0a_1x1_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_5_Branch_2_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_5_Branch_2_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_5_Branch_2_Conv2d_0a_1x1_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_5_Branch_0_Conv2d_1x1_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_5_Branch_0_Conv2d_1x1_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_5_Branch_1_Conv2d_0a_1x1_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_5_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_5_Branch_2_Conv2d_0a_1x1_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_5_Branch_2_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_5_Branch_1_Conv2d_0b_3x3_Conv2D_pad = F.pad(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_5_Branch_1_Conv2d_0a_1x1_Relu, (1, 1, 1, 1))
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_5_Branch_1_Conv2d_0b_3x3_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_5_Branch_1_Conv2d_0b_3x3_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_5_Branch_1_Conv2d_0b_3x3_Conv2D_pad)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_5_Branch_2_Conv2d_0b_3x3_Conv2D_pad = F.pad(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_5_Branch_2_Conv2d_0a_1x1_Relu, (1, 1, 1, 1))
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_5_Branch_2_Conv2d_0b_3x3_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_5_Branch_2_Conv2d_0b_3x3_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_5_Branch_2_Conv2d_0b_3x3_Conv2D_pad)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_5_Branch_1_Conv2d_0b_3x3_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_5_Branch_1_Conv2d_0b_3x3_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_5_Branch_1_Conv2d_0b_3x3_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_5_Branch_2_Conv2d_0b_3x3_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_5_Branch_2_Conv2d_0b_3x3_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_5_Branch_2_Conv2d_0b_3x3_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_5_Branch_1_Conv2d_0b_3x3_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_5_Branch_1_Conv2d_0b_3x3_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_5_Branch_2_Conv2d_0b_3x3_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_5_Branch_2_Conv2d_0b_3x3_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_5_Branch_2_Conv2d_0c_3x3_Conv2D_pad = F.pad(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_5_Branch_2_Conv2d_0b_3x3_Relu, (1, 1, 1, 1))
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_5_Branch_2_Conv2d_0c_3x3_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_5_Branch_2_Conv2d_0c_3x3_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_5_Branch_2_Conv2d_0c_3x3_Conv2D_pad)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_5_Branch_2_Conv2d_0c_3x3_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_5_Branch_2_Conv2d_0c_3x3_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_5_Branch_2_Conv2d_0c_3x3_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_5_Branch_2_Conv2d_0c_3x3_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_5_Branch_2_Conv2d_0c_3x3_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_5_concat = torch.cat((
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_5_Branch_0_Conv2d_1x1_Relu,
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_5_Branch_1_Conv2d_0b_3x3_Relu,
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_5_Branch_2_Conv2d_0c_3x3_Relu,),
            1)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_5_Conv2d_1x1_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_5_Conv2d_1x1_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_5_concat)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_5_mul = self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_5_mul_x * InceptionResnetV2_InceptionResnetV2_Repeat_block35_5_Conv2d_1x1_Conv2D
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_5_add = InceptionResnetV2_InceptionResnetV2_Repeat_block35_4_Relu + InceptionResnetV2_InceptionResnetV2_Repeat_block35_5_mul
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_5_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_5_add)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_6_Branch_0_Conv2d_1x1_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_6_Branch_0_Conv2d_1x1_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_5_Relu)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_6_Branch_1_Conv2d_0a_1x1_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_6_Branch_1_Conv2d_0a_1x1_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_5_Relu)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_6_Branch_2_Conv2d_0a_1x1_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_6_Branch_2_Conv2d_0a_1x1_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_5_Relu)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_6_Branch_0_Conv2d_1x1_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_6_Branch_0_Conv2d_1x1_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_6_Branch_0_Conv2d_1x1_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_6_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_6_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_6_Branch_1_Conv2d_0a_1x1_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_6_Branch_2_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_6_Branch_2_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_6_Branch_2_Conv2d_0a_1x1_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_6_Branch_0_Conv2d_1x1_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_6_Branch_0_Conv2d_1x1_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_6_Branch_1_Conv2d_0a_1x1_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_6_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_6_Branch_2_Conv2d_0a_1x1_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_6_Branch_2_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_6_Branch_1_Conv2d_0b_3x3_Conv2D_pad = F.pad(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_6_Branch_1_Conv2d_0a_1x1_Relu, (1, 1, 1, 1))
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_6_Branch_1_Conv2d_0b_3x3_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_6_Branch_1_Conv2d_0b_3x3_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_6_Branch_1_Conv2d_0b_3x3_Conv2D_pad)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_6_Branch_2_Conv2d_0b_3x3_Conv2D_pad = F.pad(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_6_Branch_2_Conv2d_0a_1x1_Relu, (1, 1, 1, 1))
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_6_Branch_2_Conv2d_0b_3x3_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_6_Branch_2_Conv2d_0b_3x3_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_6_Branch_2_Conv2d_0b_3x3_Conv2D_pad)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_6_Branch_1_Conv2d_0b_3x3_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_6_Branch_1_Conv2d_0b_3x3_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_6_Branch_1_Conv2d_0b_3x3_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_6_Branch_2_Conv2d_0b_3x3_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_6_Branch_2_Conv2d_0b_3x3_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_6_Branch_2_Conv2d_0b_3x3_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_6_Branch_1_Conv2d_0b_3x3_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_6_Branch_1_Conv2d_0b_3x3_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_6_Branch_2_Conv2d_0b_3x3_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_6_Branch_2_Conv2d_0b_3x3_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_6_Branch_2_Conv2d_0c_3x3_Conv2D_pad = F.pad(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_6_Branch_2_Conv2d_0b_3x3_Relu, (1, 1, 1, 1))
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_6_Branch_2_Conv2d_0c_3x3_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_6_Branch_2_Conv2d_0c_3x3_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_6_Branch_2_Conv2d_0c_3x3_Conv2D_pad)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_6_Branch_2_Conv2d_0c_3x3_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_6_Branch_2_Conv2d_0c_3x3_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_6_Branch_2_Conv2d_0c_3x3_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_6_Branch_2_Conv2d_0c_3x3_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_6_Branch_2_Conv2d_0c_3x3_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_6_concat = torch.cat((
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_6_Branch_0_Conv2d_1x1_Relu,
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_6_Branch_1_Conv2d_0b_3x3_Relu,
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_6_Branch_2_Conv2d_0c_3x3_Relu,),
            1)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_6_Conv2d_1x1_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_6_Conv2d_1x1_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_6_concat)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_6_mul = self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_6_mul_x * InceptionResnetV2_InceptionResnetV2_Repeat_block35_6_Conv2d_1x1_Conv2D
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_6_add = InceptionResnetV2_InceptionResnetV2_Repeat_block35_5_Relu + InceptionResnetV2_InceptionResnetV2_Repeat_block35_6_mul
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_6_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_6_add)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_7_Branch_0_Conv2d_1x1_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_7_Branch_0_Conv2d_1x1_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_6_Relu)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_7_Branch_1_Conv2d_0a_1x1_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_7_Branch_1_Conv2d_0a_1x1_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_6_Relu)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_7_Branch_2_Conv2d_0a_1x1_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_7_Branch_2_Conv2d_0a_1x1_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_6_Relu)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_7_Branch_0_Conv2d_1x1_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_7_Branch_0_Conv2d_1x1_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_7_Branch_0_Conv2d_1x1_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_7_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_7_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_7_Branch_1_Conv2d_0a_1x1_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_7_Branch_2_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_7_Branch_2_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_7_Branch_2_Conv2d_0a_1x1_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_7_Branch_0_Conv2d_1x1_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_7_Branch_0_Conv2d_1x1_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_7_Branch_1_Conv2d_0a_1x1_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_7_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_7_Branch_2_Conv2d_0a_1x1_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_7_Branch_2_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_7_Branch_1_Conv2d_0b_3x3_Conv2D_pad = F.pad(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_7_Branch_1_Conv2d_0a_1x1_Relu, (1, 1, 1, 1))
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_7_Branch_1_Conv2d_0b_3x3_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_7_Branch_1_Conv2d_0b_3x3_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_7_Branch_1_Conv2d_0b_3x3_Conv2D_pad)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_7_Branch_2_Conv2d_0b_3x3_Conv2D_pad = F.pad(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_7_Branch_2_Conv2d_0a_1x1_Relu, (1, 1, 1, 1))
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_7_Branch_2_Conv2d_0b_3x3_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_7_Branch_2_Conv2d_0b_3x3_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_7_Branch_2_Conv2d_0b_3x3_Conv2D_pad)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_7_Branch_1_Conv2d_0b_3x3_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_7_Branch_1_Conv2d_0b_3x3_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_7_Branch_1_Conv2d_0b_3x3_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_7_Branch_2_Conv2d_0b_3x3_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_7_Branch_2_Conv2d_0b_3x3_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_7_Branch_2_Conv2d_0b_3x3_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_7_Branch_1_Conv2d_0b_3x3_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_7_Branch_1_Conv2d_0b_3x3_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_7_Branch_2_Conv2d_0b_3x3_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_7_Branch_2_Conv2d_0b_3x3_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_7_Branch_2_Conv2d_0c_3x3_Conv2D_pad = F.pad(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_7_Branch_2_Conv2d_0b_3x3_Relu, (1, 1, 1, 1))
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_7_Branch_2_Conv2d_0c_3x3_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_7_Branch_2_Conv2d_0c_3x3_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_7_Branch_2_Conv2d_0c_3x3_Conv2D_pad)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_7_Branch_2_Conv2d_0c_3x3_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_7_Branch_2_Conv2d_0c_3x3_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_7_Branch_2_Conv2d_0c_3x3_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_7_Branch_2_Conv2d_0c_3x3_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_7_Branch_2_Conv2d_0c_3x3_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_7_concat = torch.cat((
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_7_Branch_0_Conv2d_1x1_Relu,
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_7_Branch_1_Conv2d_0b_3x3_Relu,
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_7_Branch_2_Conv2d_0c_3x3_Relu,),
            1)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_7_Conv2d_1x1_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_7_Conv2d_1x1_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_7_concat)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_7_mul = self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_7_mul_x * InceptionResnetV2_InceptionResnetV2_Repeat_block35_7_Conv2d_1x1_Conv2D
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_7_add = InceptionResnetV2_InceptionResnetV2_Repeat_block35_6_Relu + InceptionResnetV2_InceptionResnetV2_Repeat_block35_7_mul
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_7_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_7_add)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_8_Branch_0_Conv2d_1x1_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_8_Branch_0_Conv2d_1x1_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_7_Relu)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_8_Branch_1_Conv2d_0a_1x1_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_8_Branch_1_Conv2d_0a_1x1_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_7_Relu)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_8_Branch_2_Conv2d_0a_1x1_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_8_Branch_2_Conv2d_0a_1x1_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_7_Relu)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_8_Branch_0_Conv2d_1x1_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_8_Branch_0_Conv2d_1x1_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_8_Branch_0_Conv2d_1x1_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_8_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_8_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_8_Branch_1_Conv2d_0a_1x1_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_8_Branch_2_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_8_Branch_2_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_8_Branch_2_Conv2d_0a_1x1_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_8_Branch_0_Conv2d_1x1_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_8_Branch_0_Conv2d_1x1_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_8_Branch_1_Conv2d_0a_1x1_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_8_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_8_Branch_2_Conv2d_0a_1x1_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_8_Branch_2_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_8_Branch_1_Conv2d_0b_3x3_Conv2D_pad = F.pad(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_8_Branch_1_Conv2d_0a_1x1_Relu, (1, 1, 1, 1))
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_8_Branch_1_Conv2d_0b_3x3_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_8_Branch_1_Conv2d_0b_3x3_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_8_Branch_1_Conv2d_0b_3x3_Conv2D_pad)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_8_Branch_2_Conv2d_0b_3x3_Conv2D_pad = F.pad(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_8_Branch_2_Conv2d_0a_1x1_Relu, (1, 1, 1, 1))
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_8_Branch_2_Conv2d_0b_3x3_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_8_Branch_2_Conv2d_0b_3x3_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_8_Branch_2_Conv2d_0b_3x3_Conv2D_pad)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_8_Branch_1_Conv2d_0b_3x3_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_8_Branch_1_Conv2d_0b_3x3_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_8_Branch_1_Conv2d_0b_3x3_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_8_Branch_2_Conv2d_0b_3x3_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_8_Branch_2_Conv2d_0b_3x3_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_8_Branch_2_Conv2d_0b_3x3_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_8_Branch_1_Conv2d_0b_3x3_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_8_Branch_1_Conv2d_0b_3x3_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_8_Branch_2_Conv2d_0b_3x3_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_8_Branch_2_Conv2d_0b_3x3_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_8_Branch_2_Conv2d_0c_3x3_Conv2D_pad = F.pad(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_8_Branch_2_Conv2d_0b_3x3_Relu, (1, 1, 1, 1))
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_8_Branch_2_Conv2d_0c_3x3_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_8_Branch_2_Conv2d_0c_3x3_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_8_Branch_2_Conv2d_0c_3x3_Conv2D_pad)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_8_Branch_2_Conv2d_0c_3x3_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_8_Branch_2_Conv2d_0c_3x3_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_8_Branch_2_Conv2d_0c_3x3_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_8_Branch_2_Conv2d_0c_3x3_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_8_Branch_2_Conv2d_0c_3x3_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_8_concat = torch.cat((
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_8_Branch_0_Conv2d_1x1_Relu,
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_8_Branch_1_Conv2d_0b_3x3_Relu,
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_8_Branch_2_Conv2d_0c_3x3_Relu,),
            1)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_8_Conv2d_1x1_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_8_Conv2d_1x1_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_8_concat)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_8_mul = self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_8_mul_x * InceptionResnetV2_InceptionResnetV2_Repeat_block35_8_Conv2d_1x1_Conv2D
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_8_add = InceptionResnetV2_InceptionResnetV2_Repeat_block35_7_Relu + InceptionResnetV2_InceptionResnetV2_Repeat_block35_8_mul
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_8_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_8_add)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_9_Branch_0_Conv2d_1x1_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_9_Branch_0_Conv2d_1x1_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_8_Relu)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_9_Branch_1_Conv2d_0a_1x1_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_9_Branch_1_Conv2d_0a_1x1_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_8_Relu)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_9_Branch_2_Conv2d_0a_1x1_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_9_Branch_2_Conv2d_0a_1x1_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_8_Relu)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_9_Branch_0_Conv2d_1x1_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_9_Branch_0_Conv2d_1x1_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_9_Branch_0_Conv2d_1x1_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_9_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_9_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_9_Branch_1_Conv2d_0a_1x1_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_9_Branch_2_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_9_Branch_2_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_9_Branch_2_Conv2d_0a_1x1_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_9_Branch_0_Conv2d_1x1_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_9_Branch_0_Conv2d_1x1_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_9_Branch_1_Conv2d_0a_1x1_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_9_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_9_Branch_2_Conv2d_0a_1x1_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_9_Branch_2_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_9_Branch_1_Conv2d_0b_3x3_Conv2D_pad = F.pad(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_9_Branch_1_Conv2d_0a_1x1_Relu, (1, 1, 1, 1))
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_9_Branch_1_Conv2d_0b_3x3_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_9_Branch_1_Conv2d_0b_3x3_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_9_Branch_1_Conv2d_0b_3x3_Conv2D_pad)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_9_Branch_2_Conv2d_0b_3x3_Conv2D_pad = F.pad(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_9_Branch_2_Conv2d_0a_1x1_Relu, (1, 1, 1, 1))
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_9_Branch_2_Conv2d_0b_3x3_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_9_Branch_2_Conv2d_0b_3x3_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_9_Branch_2_Conv2d_0b_3x3_Conv2D_pad)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_9_Branch_1_Conv2d_0b_3x3_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_9_Branch_1_Conv2d_0b_3x3_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_9_Branch_1_Conv2d_0b_3x3_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_9_Branch_2_Conv2d_0b_3x3_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_9_Branch_2_Conv2d_0b_3x3_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_9_Branch_2_Conv2d_0b_3x3_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_9_Branch_1_Conv2d_0b_3x3_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_9_Branch_1_Conv2d_0b_3x3_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_9_Branch_2_Conv2d_0b_3x3_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_9_Branch_2_Conv2d_0b_3x3_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_9_Branch_2_Conv2d_0c_3x3_Conv2D_pad = F.pad(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_9_Branch_2_Conv2d_0b_3x3_Relu, (1, 1, 1, 1))
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_9_Branch_2_Conv2d_0c_3x3_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_9_Branch_2_Conv2d_0c_3x3_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_9_Branch_2_Conv2d_0c_3x3_Conv2D_pad)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_9_Branch_2_Conv2d_0c_3x3_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_9_Branch_2_Conv2d_0c_3x3_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_9_Branch_2_Conv2d_0c_3x3_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_9_Branch_2_Conv2d_0c_3x3_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_9_Branch_2_Conv2d_0c_3x3_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_9_concat = torch.cat((
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_9_Branch_0_Conv2d_1x1_Relu,
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_9_Branch_1_Conv2d_0b_3x3_Relu,
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_9_Branch_2_Conv2d_0c_3x3_Relu,),
            1)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_9_Conv2d_1x1_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_9_Conv2d_1x1_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_9_concat)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_9_mul = self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_9_mul_x * InceptionResnetV2_InceptionResnetV2_Repeat_block35_9_Conv2d_1x1_Conv2D
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_9_add = InceptionResnetV2_InceptionResnetV2_Repeat_block35_8_Relu + InceptionResnetV2_InceptionResnetV2_Repeat_block35_9_mul
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_9_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_9_add)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_10_Branch_0_Conv2d_1x1_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_10_Branch_0_Conv2d_1x1_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_9_Relu)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_10_Branch_1_Conv2d_0a_1x1_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_10_Branch_1_Conv2d_0a_1x1_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_9_Relu)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_10_Branch_2_Conv2d_0a_1x1_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_10_Branch_2_Conv2d_0a_1x1_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_9_Relu)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_10_Branch_0_Conv2d_1x1_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_10_Branch_0_Conv2d_1x1_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_10_Branch_0_Conv2d_1x1_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_10_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_10_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_10_Branch_1_Conv2d_0a_1x1_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_10_Branch_2_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_10_Branch_2_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_10_Branch_2_Conv2d_0a_1x1_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_10_Branch_0_Conv2d_1x1_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_10_Branch_0_Conv2d_1x1_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_10_Branch_1_Conv2d_0a_1x1_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_10_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_10_Branch_2_Conv2d_0a_1x1_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_10_Branch_2_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_10_Branch_1_Conv2d_0b_3x3_Conv2D_pad = F.pad(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_10_Branch_1_Conv2d_0a_1x1_Relu, (1, 1, 1, 1))
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_10_Branch_1_Conv2d_0b_3x3_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_10_Branch_1_Conv2d_0b_3x3_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_10_Branch_1_Conv2d_0b_3x3_Conv2D_pad)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_10_Branch_2_Conv2d_0b_3x3_Conv2D_pad = F.pad(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_10_Branch_2_Conv2d_0a_1x1_Relu, (1, 1, 1, 1))
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_10_Branch_2_Conv2d_0b_3x3_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_10_Branch_2_Conv2d_0b_3x3_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_10_Branch_2_Conv2d_0b_3x3_Conv2D_pad)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_10_Branch_1_Conv2d_0b_3x3_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_10_Branch_1_Conv2d_0b_3x3_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_10_Branch_1_Conv2d_0b_3x3_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_10_Branch_2_Conv2d_0b_3x3_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_10_Branch_2_Conv2d_0b_3x3_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_10_Branch_2_Conv2d_0b_3x3_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_10_Branch_1_Conv2d_0b_3x3_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_10_Branch_1_Conv2d_0b_3x3_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_10_Branch_2_Conv2d_0b_3x3_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_10_Branch_2_Conv2d_0b_3x3_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_10_Branch_2_Conv2d_0c_3x3_Conv2D_pad = F.pad(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_10_Branch_2_Conv2d_0b_3x3_Relu, (1, 1, 1, 1))
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_10_Branch_2_Conv2d_0c_3x3_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_10_Branch_2_Conv2d_0c_3x3_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_10_Branch_2_Conv2d_0c_3x3_Conv2D_pad)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_10_Branch_2_Conv2d_0c_3x3_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_10_Branch_2_Conv2d_0c_3x3_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_10_Branch_2_Conv2d_0c_3x3_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_10_Branch_2_Conv2d_0c_3x3_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_10_Branch_2_Conv2d_0c_3x3_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_10_concat = torch.cat((
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_10_Branch_0_Conv2d_1x1_Relu,
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_10_Branch_1_Conv2d_0b_3x3_Relu,
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_10_Branch_2_Conv2d_0c_3x3_Relu,),
            1)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_10_Conv2d_1x1_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_10_Conv2d_1x1_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_10_concat)
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_10_mul = self.InceptionResnetV2_InceptionResnetV2_Repeat_block35_10_mul_x * InceptionResnetV2_InceptionResnetV2_Repeat_block35_10_Conv2d_1x1_Conv2D
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_10_add = InceptionResnetV2_InceptionResnetV2_Repeat_block35_9_Relu + InceptionResnetV2_InceptionResnetV2_Repeat_block35_10_mul
        InceptionResnetV2_InceptionResnetV2_Repeat_block35_10_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_10_add)
        InceptionResnetV2_InceptionResnetV2_Mixed_6a_Branch_0_Conv2d_1a_3x3_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Mixed_6a_Branch_0_Conv2d_1a_3x3_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_10_Relu)
        InceptionResnetV2_InceptionResnetV2_Mixed_6a_Branch_1_Conv2d_0a_1x1_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Mixed_6a_Branch_1_Conv2d_0a_1x1_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_10_Relu)
        InceptionResnetV2_InceptionResnetV2_Mixed_6a_Branch_2_MaxPool_1a_3x3_MaxPool, InceptionResnetV2_InceptionResnetV2_Mixed_6a_Branch_2_MaxPool_1a_3x3_MaxPool_idx = F.max_pool2d(
            InceptionResnetV2_InceptionResnetV2_Repeat_block35_10_Relu, kernel_size=(3, 3), stride=(2, 2), padding=0,
            ceil_mode=False, return_indices=True)
        InceptionResnetV2_InceptionResnetV2_Mixed_6a_Branch_0_Conv2d_1a_3x3_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Mixed_6a_Branch_0_Conv2d_1a_3x3_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Mixed_6a_Branch_0_Conv2d_1a_3x3_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Mixed_6a_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Mixed_6a_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Mixed_6a_Branch_1_Conv2d_0a_1x1_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Mixed_6a_Branch_0_Conv2d_1a_3x3_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Mixed_6a_Branch_0_Conv2d_1a_3x3_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Mixed_6a_Branch_1_Conv2d_0a_1x1_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Mixed_6a_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Mixed_6a_Branch_1_Conv2d_0b_3x3_Conv2D_pad = F.pad(
            InceptionResnetV2_InceptionResnetV2_Mixed_6a_Branch_1_Conv2d_0a_1x1_Relu, (1, 1, 1, 1))
        InceptionResnetV2_InceptionResnetV2_Mixed_6a_Branch_1_Conv2d_0b_3x3_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Mixed_6a_Branch_1_Conv2d_0b_3x3_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Mixed_6a_Branch_1_Conv2d_0b_3x3_Conv2D_pad)
        InceptionResnetV2_InceptionResnetV2_Mixed_6a_Branch_1_Conv2d_0b_3x3_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Mixed_6a_Branch_1_Conv2d_0b_3x3_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Mixed_6a_Branch_1_Conv2d_0b_3x3_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Mixed_6a_Branch_1_Conv2d_0b_3x3_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Mixed_6a_Branch_1_Conv2d_0b_3x3_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Mixed_6a_Branch_1_Conv2d_1a_3x3_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Mixed_6a_Branch_1_Conv2d_1a_3x3_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Mixed_6a_Branch_1_Conv2d_0b_3x3_Relu)
        InceptionResnetV2_InceptionResnetV2_Mixed_6a_Branch_1_Conv2d_1a_3x3_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Mixed_6a_Branch_1_Conv2d_1a_3x3_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Mixed_6a_Branch_1_Conv2d_1a_3x3_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Mixed_6a_Branch_1_Conv2d_1a_3x3_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Mixed_6a_Branch_1_Conv2d_1a_3x3_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Mixed_6a_concat = torch.cat((
            InceptionResnetV2_InceptionResnetV2_Mixed_6a_Branch_0_Conv2d_1a_3x3_Relu,
            InceptionResnetV2_InceptionResnetV2_Mixed_6a_Branch_1_Conv2d_1a_3x3_Relu,
            InceptionResnetV2_InceptionResnetV2_Mixed_6a_Branch_2_MaxPool_1a_3x3_MaxPool,),
            1)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_1_Branch_0_Conv2d_1x1_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_1_Branch_0_Conv2d_1x1_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Mixed_6a_concat)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_1_Branch_1_Conv2d_0a_1x1_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_1_Branch_1_Conv2d_0a_1x1_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Mixed_6a_concat)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_1_Branch_0_Conv2d_1x1_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_1_Branch_0_Conv2d_1x1_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_1_Branch_0_Conv2d_1x1_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_1_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_1_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_1_Branch_1_Conv2d_0a_1x1_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_1_Branch_0_Conv2d_1x1_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_1_Branch_0_Conv2d_1x1_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_1_Branch_1_Conv2d_0a_1x1_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_1_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_1_Branch_1_Conv2d_0b_1x7_Conv2D_pad = F.pad(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_1_Branch_1_Conv2d_0a_1x1_Relu, (3, 3, 0, 0))
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_1_Branch_1_Conv2d_0b_1x7_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_1_Branch_1_Conv2d_0b_1x7_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_1_Branch_1_Conv2d_0b_1x7_Conv2D_pad)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_1_Branch_1_Conv2d_0b_1x7_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_1_Branch_1_Conv2d_0b_1x7_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_1_Branch_1_Conv2d_0b_1x7_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_1_Branch_1_Conv2d_0b_1x7_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_1_Branch_1_Conv2d_0b_1x7_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_1_Branch_1_Conv2d_0c_7x1_Conv2D_pad = F.pad(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_1_Branch_1_Conv2d_0b_1x7_Relu, (0, 0, 3, 3))
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_1_Branch_1_Conv2d_0c_7x1_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_1_Branch_1_Conv2d_0c_7x1_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_1_Branch_1_Conv2d_0c_7x1_Conv2D_pad)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_1_Branch_1_Conv2d_0c_7x1_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_1_Branch_1_Conv2d_0c_7x1_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_1_Branch_1_Conv2d_0c_7x1_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_1_Branch_1_Conv2d_0c_7x1_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_1_Branch_1_Conv2d_0c_7x1_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_1_concat = torch.cat((
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_1_Branch_0_Conv2d_1x1_Relu,
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_1_Branch_1_Conv2d_0c_7x1_Relu,),
            1)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_1_Conv2d_1x1_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_1_Conv2d_1x1_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_1_concat)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_1_mul = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_1_mul_x * InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_1_Conv2d_1x1_Conv2D
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_1_add = InceptionResnetV2_InceptionResnetV2_Mixed_6a_concat + InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_1_mul
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_1_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_1_add)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_2_Branch_0_Conv2d_1x1_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_2_Branch_0_Conv2d_1x1_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_1_Relu)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_2_Branch_1_Conv2d_0a_1x1_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_2_Branch_1_Conv2d_0a_1x1_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_1_Relu)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_2_Branch_0_Conv2d_1x1_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_2_Branch_0_Conv2d_1x1_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_2_Branch_0_Conv2d_1x1_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_2_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_2_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_2_Branch_1_Conv2d_0a_1x1_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_2_Branch_0_Conv2d_1x1_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_2_Branch_0_Conv2d_1x1_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_2_Branch_1_Conv2d_0a_1x1_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_2_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_2_Branch_1_Conv2d_0b_1x7_Conv2D_pad = F.pad(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_2_Branch_1_Conv2d_0a_1x1_Relu, (3, 3, 0, 0))
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_2_Branch_1_Conv2d_0b_1x7_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_2_Branch_1_Conv2d_0b_1x7_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_2_Branch_1_Conv2d_0b_1x7_Conv2D_pad)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_2_Branch_1_Conv2d_0b_1x7_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_2_Branch_1_Conv2d_0b_1x7_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_2_Branch_1_Conv2d_0b_1x7_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_2_Branch_1_Conv2d_0b_1x7_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_2_Branch_1_Conv2d_0b_1x7_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_2_Branch_1_Conv2d_0c_7x1_Conv2D_pad = F.pad(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_2_Branch_1_Conv2d_0b_1x7_Relu, (0, 0, 3, 3))
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_2_Branch_1_Conv2d_0c_7x1_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_2_Branch_1_Conv2d_0c_7x1_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_2_Branch_1_Conv2d_0c_7x1_Conv2D_pad)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_2_Branch_1_Conv2d_0c_7x1_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_2_Branch_1_Conv2d_0c_7x1_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_2_Branch_1_Conv2d_0c_7x1_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_2_Branch_1_Conv2d_0c_7x1_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_2_Branch_1_Conv2d_0c_7x1_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_2_concat = torch.cat((
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_2_Branch_0_Conv2d_1x1_Relu,
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_2_Branch_1_Conv2d_0c_7x1_Relu,),
            1)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_2_Conv2d_1x1_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_2_Conv2d_1x1_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_2_concat)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_2_mul = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_2_mul_x * InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_2_Conv2d_1x1_Conv2D
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_2_add = InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_1_Relu + InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_2_mul
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_2_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_2_add)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_3_Branch_0_Conv2d_1x1_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_3_Branch_0_Conv2d_1x1_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_2_Relu)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_3_Branch_1_Conv2d_0a_1x1_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_3_Branch_1_Conv2d_0a_1x1_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_2_Relu)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_3_Branch_0_Conv2d_1x1_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_3_Branch_0_Conv2d_1x1_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_3_Branch_0_Conv2d_1x1_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_3_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_3_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_3_Branch_1_Conv2d_0a_1x1_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_3_Branch_0_Conv2d_1x1_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_3_Branch_0_Conv2d_1x1_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_3_Branch_1_Conv2d_0a_1x1_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_3_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_3_Branch_1_Conv2d_0b_1x7_Conv2D_pad = F.pad(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_3_Branch_1_Conv2d_0a_1x1_Relu, (3, 3, 0, 0))
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_3_Branch_1_Conv2d_0b_1x7_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_3_Branch_1_Conv2d_0b_1x7_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_3_Branch_1_Conv2d_0b_1x7_Conv2D_pad)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_3_Branch_1_Conv2d_0b_1x7_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_3_Branch_1_Conv2d_0b_1x7_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_3_Branch_1_Conv2d_0b_1x7_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_3_Branch_1_Conv2d_0b_1x7_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_3_Branch_1_Conv2d_0b_1x7_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_3_Branch_1_Conv2d_0c_7x1_Conv2D_pad = F.pad(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_3_Branch_1_Conv2d_0b_1x7_Relu, (0, 0, 3, 3))
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_3_Branch_1_Conv2d_0c_7x1_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_3_Branch_1_Conv2d_0c_7x1_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_3_Branch_1_Conv2d_0c_7x1_Conv2D_pad)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_3_Branch_1_Conv2d_0c_7x1_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_3_Branch_1_Conv2d_0c_7x1_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_3_Branch_1_Conv2d_0c_7x1_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_3_Branch_1_Conv2d_0c_7x1_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_3_Branch_1_Conv2d_0c_7x1_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_3_concat = torch.cat((
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_3_Branch_0_Conv2d_1x1_Relu,
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_3_Branch_1_Conv2d_0c_7x1_Relu,),
            1)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_3_Conv2d_1x1_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_3_Conv2d_1x1_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_3_concat)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_3_mul = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_3_mul_x * InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_3_Conv2d_1x1_Conv2D
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_3_add = InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_2_Relu + InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_3_mul
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_3_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_3_add)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_4_Branch_0_Conv2d_1x1_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_4_Branch_0_Conv2d_1x1_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_3_Relu)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_4_Branch_1_Conv2d_0a_1x1_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_4_Branch_1_Conv2d_0a_1x1_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_3_Relu)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_4_Branch_0_Conv2d_1x1_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_4_Branch_0_Conv2d_1x1_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_4_Branch_0_Conv2d_1x1_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_4_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_4_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_4_Branch_1_Conv2d_0a_1x1_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_4_Branch_0_Conv2d_1x1_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_4_Branch_0_Conv2d_1x1_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_4_Branch_1_Conv2d_0a_1x1_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_4_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_4_Branch_1_Conv2d_0b_1x7_Conv2D_pad = F.pad(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_4_Branch_1_Conv2d_0a_1x1_Relu, (3, 3, 0, 0))
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_4_Branch_1_Conv2d_0b_1x7_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_4_Branch_1_Conv2d_0b_1x7_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_4_Branch_1_Conv2d_0b_1x7_Conv2D_pad)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_4_Branch_1_Conv2d_0b_1x7_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_4_Branch_1_Conv2d_0b_1x7_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_4_Branch_1_Conv2d_0b_1x7_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_4_Branch_1_Conv2d_0b_1x7_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_4_Branch_1_Conv2d_0b_1x7_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_4_Branch_1_Conv2d_0c_7x1_Conv2D_pad = F.pad(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_4_Branch_1_Conv2d_0b_1x7_Relu, (0, 0, 3, 3))
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_4_Branch_1_Conv2d_0c_7x1_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_4_Branch_1_Conv2d_0c_7x1_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_4_Branch_1_Conv2d_0c_7x1_Conv2D_pad)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_4_Branch_1_Conv2d_0c_7x1_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_4_Branch_1_Conv2d_0c_7x1_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_4_Branch_1_Conv2d_0c_7x1_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_4_Branch_1_Conv2d_0c_7x1_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_4_Branch_1_Conv2d_0c_7x1_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_4_concat = torch.cat((
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_4_Branch_0_Conv2d_1x1_Relu,
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_4_Branch_1_Conv2d_0c_7x1_Relu,),
            1)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_4_Conv2d_1x1_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_4_Conv2d_1x1_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_4_concat)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_4_mul = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_4_mul_x * InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_4_Conv2d_1x1_Conv2D
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_4_add = InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_3_Relu + InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_4_mul
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_4_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_4_add)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_5_Branch_0_Conv2d_1x1_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_5_Branch_0_Conv2d_1x1_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_4_Relu)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_5_Branch_1_Conv2d_0a_1x1_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_5_Branch_1_Conv2d_0a_1x1_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_4_Relu)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_5_Branch_0_Conv2d_1x1_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_5_Branch_0_Conv2d_1x1_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_5_Branch_0_Conv2d_1x1_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_5_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_5_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_5_Branch_1_Conv2d_0a_1x1_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_5_Branch_0_Conv2d_1x1_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_5_Branch_0_Conv2d_1x1_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_5_Branch_1_Conv2d_0a_1x1_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_5_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_5_Branch_1_Conv2d_0b_1x7_Conv2D_pad = F.pad(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_5_Branch_1_Conv2d_0a_1x1_Relu, (3, 3, 0, 0))
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_5_Branch_1_Conv2d_0b_1x7_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_5_Branch_1_Conv2d_0b_1x7_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_5_Branch_1_Conv2d_0b_1x7_Conv2D_pad)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_5_Branch_1_Conv2d_0b_1x7_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_5_Branch_1_Conv2d_0b_1x7_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_5_Branch_1_Conv2d_0b_1x7_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_5_Branch_1_Conv2d_0b_1x7_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_5_Branch_1_Conv2d_0b_1x7_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_5_Branch_1_Conv2d_0c_7x1_Conv2D_pad = F.pad(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_5_Branch_1_Conv2d_0b_1x7_Relu, (0, 0, 3, 3))
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_5_Branch_1_Conv2d_0c_7x1_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_5_Branch_1_Conv2d_0c_7x1_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_5_Branch_1_Conv2d_0c_7x1_Conv2D_pad)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_5_Branch_1_Conv2d_0c_7x1_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_5_Branch_1_Conv2d_0c_7x1_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_5_Branch_1_Conv2d_0c_7x1_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_5_Branch_1_Conv2d_0c_7x1_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_5_Branch_1_Conv2d_0c_7x1_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_5_concat = torch.cat((
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_5_Branch_0_Conv2d_1x1_Relu,
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_5_Branch_1_Conv2d_0c_7x1_Relu,),
            1)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_5_Conv2d_1x1_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_5_Conv2d_1x1_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_5_concat)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_5_mul = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_5_mul_x * InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_5_Conv2d_1x1_Conv2D
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_5_add = InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_4_Relu + InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_5_mul
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_5_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_5_add)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_6_Branch_0_Conv2d_1x1_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_6_Branch_0_Conv2d_1x1_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_5_Relu)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_6_Branch_1_Conv2d_0a_1x1_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_6_Branch_1_Conv2d_0a_1x1_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_5_Relu)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_6_Branch_0_Conv2d_1x1_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_6_Branch_0_Conv2d_1x1_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_6_Branch_0_Conv2d_1x1_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_6_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_6_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_6_Branch_1_Conv2d_0a_1x1_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_6_Branch_0_Conv2d_1x1_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_6_Branch_0_Conv2d_1x1_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_6_Branch_1_Conv2d_0a_1x1_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_6_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_6_Branch_1_Conv2d_0b_1x7_Conv2D_pad = F.pad(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_6_Branch_1_Conv2d_0a_1x1_Relu, (3, 3, 0, 0))
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_6_Branch_1_Conv2d_0b_1x7_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_6_Branch_1_Conv2d_0b_1x7_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_6_Branch_1_Conv2d_0b_1x7_Conv2D_pad)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_6_Branch_1_Conv2d_0b_1x7_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_6_Branch_1_Conv2d_0b_1x7_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_6_Branch_1_Conv2d_0b_1x7_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_6_Branch_1_Conv2d_0b_1x7_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_6_Branch_1_Conv2d_0b_1x7_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_6_Branch_1_Conv2d_0c_7x1_Conv2D_pad = F.pad(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_6_Branch_1_Conv2d_0b_1x7_Relu, (0, 0, 3, 3))
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_6_Branch_1_Conv2d_0c_7x1_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_6_Branch_1_Conv2d_0c_7x1_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_6_Branch_1_Conv2d_0c_7x1_Conv2D_pad)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_6_Branch_1_Conv2d_0c_7x1_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_6_Branch_1_Conv2d_0c_7x1_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_6_Branch_1_Conv2d_0c_7x1_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_6_Branch_1_Conv2d_0c_7x1_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_6_Branch_1_Conv2d_0c_7x1_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_6_concat = torch.cat((
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_6_Branch_0_Conv2d_1x1_Relu,
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_6_Branch_1_Conv2d_0c_7x1_Relu,),
            1)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_6_Conv2d_1x1_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_6_Conv2d_1x1_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_6_concat)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_6_mul = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_6_mul_x * InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_6_Conv2d_1x1_Conv2D
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_6_add = InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_5_Relu + InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_6_mul
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_6_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_6_add)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_7_Branch_0_Conv2d_1x1_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_7_Branch_0_Conv2d_1x1_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_6_Relu)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_7_Branch_1_Conv2d_0a_1x1_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_7_Branch_1_Conv2d_0a_1x1_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_6_Relu)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_7_Branch_0_Conv2d_1x1_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_7_Branch_0_Conv2d_1x1_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_7_Branch_0_Conv2d_1x1_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_7_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_7_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_7_Branch_1_Conv2d_0a_1x1_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_7_Branch_0_Conv2d_1x1_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_7_Branch_0_Conv2d_1x1_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_7_Branch_1_Conv2d_0a_1x1_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_7_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_7_Branch_1_Conv2d_0b_1x7_Conv2D_pad = F.pad(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_7_Branch_1_Conv2d_0a_1x1_Relu, (3, 3, 0, 0))
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_7_Branch_1_Conv2d_0b_1x7_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_7_Branch_1_Conv2d_0b_1x7_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_7_Branch_1_Conv2d_0b_1x7_Conv2D_pad)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_7_Branch_1_Conv2d_0b_1x7_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_7_Branch_1_Conv2d_0b_1x7_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_7_Branch_1_Conv2d_0b_1x7_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_7_Branch_1_Conv2d_0b_1x7_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_7_Branch_1_Conv2d_0b_1x7_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_7_Branch_1_Conv2d_0c_7x1_Conv2D_pad = F.pad(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_7_Branch_1_Conv2d_0b_1x7_Relu, (0, 0, 3, 3))
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_7_Branch_1_Conv2d_0c_7x1_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_7_Branch_1_Conv2d_0c_7x1_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_7_Branch_1_Conv2d_0c_7x1_Conv2D_pad)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_7_Branch_1_Conv2d_0c_7x1_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_7_Branch_1_Conv2d_0c_7x1_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_7_Branch_1_Conv2d_0c_7x1_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_7_Branch_1_Conv2d_0c_7x1_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_7_Branch_1_Conv2d_0c_7x1_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_7_concat = torch.cat((
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_7_Branch_0_Conv2d_1x1_Relu,
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_7_Branch_1_Conv2d_0c_7x1_Relu,),
            1)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_7_Conv2d_1x1_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_7_Conv2d_1x1_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_7_concat)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_7_mul = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_7_mul_x * InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_7_Conv2d_1x1_Conv2D
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_7_add = InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_6_Relu + InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_7_mul
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_7_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_7_add)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_8_Branch_0_Conv2d_1x1_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_8_Branch_0_Conv2d_1x1_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_7_Relu)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_8_Branch_1_Conv2d_0a_1x1_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_8_Branch_1_Conv2d_0a_1x1_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_7_Relu)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_8_Branch_0_Conv2d_1x1_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_8_Branch_0_Conv2d_1x1_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_8_Branch_0_Conv2d_1x1_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_8_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_8_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_8_Branch_1_Conv2d_0a_1x1_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_8_Branch_0_Conv2d_1x1_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_8_Branch_0_Conv2d_1x1_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_8_Branch_1_Conv2d_0a_1x1_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_8_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_8_Branch_1_Conv2d_0b_1x7_Conv2D_pad = F.pad(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_8_Branch_1_Conv2d_0a_1x1_Relu, (3, 3, 0, 0))
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_8_Branch_1_Conv2d_0b_1x7_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_8_Branch_1_Conv2d_0b_1x7_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_8_Branch_1_Conv2d_0b_1x7_Conv2D_pad)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_8_Branch_1_Conv2d_0b_1x7_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_8_Branch_1_Conv2d_0b_1x7_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_8_Branch_1_Conv2d_0b_1x7_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_8_Branch_1_Conv2d_0b_1x7_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_8_Branch_1_Conv2d_0b_1x7_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_8_Branch_1_Conv2d_0c_7x1_Conv2D_pad = F.pad(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_8_Branch_1_Conv2d_0b_1x7_Relu, (0, 0, 3, 3))
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_8_Branch_1_Conv2d_0c_7x1_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_8_Branch_1_Conv2d_0c_7x1_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_8_Branch_1_Conv2d_0c_7x1_Conv2D_pad)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_8_Branch_1_Conv2d_0c_7x1_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_8_Branch_1_Conv2d_0c_7x1_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_8_Branch_1_Conv2d_0c_7x1_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_8_Branch_1_Conv2d_0c_7x1_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_8_Branch_1_Conv2d_0c_7x1_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_8_concat = torch.cat((
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_8_Branch_0_Conv2d_1x1_Relu,
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_8_Branch_1_Conv2d_0c_7x1_Relu,),
            1)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_8_Conv2d_1x1_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_8_Conv2d_1x1_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_8_concat)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_8_mul = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_8_mul_x * InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_8_Conv2d_1x1_Conv2D
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_8_add = InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_7_Relu + InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_8_mul
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_8_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_8_add)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_9_Branch_0_Conv2d_1x1_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_9_Branch_0_Conv2d_1x1_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_8_Relu)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_9_Branch_1_Conv2d_0a_1x1_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_9_Branch_1_Conv2d_0a_1x1_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_8_Relu)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_9_Branch_0_Conv2d_1x1_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_9_Branch_0_Conv2d_1x1_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_9_Branch_0_Conv2d_1x1_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_9_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_9_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_9_Branch_1_Conv2d_0a_1x1_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_9_Branch_0_Conv2d_1x1_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_9_Branch_0_Conv2d_1x1_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_9_Branch_1_Conv2d_0a_1x1_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_9_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_9_Branch_1_Conv2d_0b_1x7_Conv2D_pad = F.pad(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_9_Branch_1_Conv2d_0a_1x1_Relu, (3, 3, 0, 0))
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_9_Branch_1_Conv2d_0b_1x7_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_9_Branch_1_Conv2d_0b_1x7_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_9_Branch_1_Conv2d_0b_1x7_Conv2D_pad)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_9_Branch_1_Conv2d_0b_1x7_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_9_Branch_1_Conv2d_0b_1x7_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_9_Branch_1_Conv2d_0b_1x7_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_9_Branch_1_Conv2d_0b_1x7_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_9_Branch_1_Conv2d_0b_1x7_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_9_Branch_1_Conv2d_0c_7x1_Conv2D_pad = F.pad(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_9_Branch_1_Conv2d_0b_1x7_Relu, (0, 0, 3, 3))
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_9_Branch_1_Conv2d_0c_7x1_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_9_Branch_1_Conv2d_0c_7x1_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_9_Branch_1_Conv2d_0c_7x1_Conv2D_pad)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_9_Branch_1_Conv2d_0c_7x1_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_9_Branch_1_Conv2d_0c_7x1_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_9_Branch_1_Conv2d_0c_7x1_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_9_Branch_1_Conv2d_0c_7x1_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_9_Branch_1_Conv2d_0c_7x1_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_9_concat = torch.cat((
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_9_Branch_0_Conv2d_1x1_Relu,
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_9_Branch_1_Conv2d_0c_7x1_Relu,),
            1)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_9_Conv2d_1x1_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_9_Conv2d_1x1_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_9_concat)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_9_mul = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_9_mul_x * InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_9_Conv2d_1x1_Conv2D
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_9_add = InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_8_Relu + InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_9_mul
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_9_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_9_add)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_10_Branch_0_Conv2d_1x1_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_10_Branch_0_Conv2d_1x1_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_9_Relu)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_10_Branch_1_Conv2d_0a_1x1_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_10_Branch_1_Conv2d_0a_1x1_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_9_Relu)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_10_Branch_0_Conv2d_1x1_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_10_Branch_0_Conv2d_1x1_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_10_Branch_0_Conv2d_1x1_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_10_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_10_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_10_Branch_1_Conv2d_0a_1x1_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_10_Branch_0_Conv2d_1x1_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_10_Branch_0_Conv2d_1x1_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_10_Branch_1_Conv2d_0a_1x1_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_10_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_10_Branch_1_Conv2d_0b_1x7_Conv2D_pad = F.pad(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_10_Branch_1_Conv2d_0a_1x1_Relu, (3, 3, 0, 0))
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_10_Branch_1_Conv2d_0b_1x7_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_10_Branch_1_Conv2d_0b_1x7_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_10_Branch_1_Conv2d_0b_1x7_Conv2D_pad)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_10_Branch_1_Conv2d_0b_1x7_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_10_Branch_1_Conv2d_0b_1x7_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_10_Branch_1_Conv2d_0b_1x7_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_10_Branch_1_Conv2d_0b_1x7_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_10_Branch_1_Conv2d_0b_1x7_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_10_Branch_1_Conv2d_0c_7x1_Conv2D_pad = F.pad(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_10_Branch_1_Conv2d_0b_1x7_Relu, (0, 0, 3, 3))
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_10_Branch_1_Conv2d_0c_7x1_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_10_Branch_1_Conv2d_0c_7x1_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_10_Branch_1_Conv2d_0c_7x1_Conv2D_pad)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_10_Branch_1_Conv2d_0c_7x1_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_10_Branch_1_Conv2d_0c_7x1_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_10_Branch_1_Conv2d_0c_7x1_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_10_Branch_1_Conv2d_0c_7x1_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_10_Branch_1_Conv2d_0c_7x1_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_10_concat = torch.cat((
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_10_Branch_0_Conv2d_1x1_Relu,
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_10_Branch_1_Conv2d_0c_7x1_Relu,),
            1)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_10_Conv2d_1x1_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_10_Conv2d_1x1_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_10_concat)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_10_mul = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_10_mul_x * InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_10_Conv2d_1x1_Conv2D
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_10_add = InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_9_Relu + InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_10_mul
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_10_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_10_add)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_11_Branch_0_Conv2d_1x1_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_11_Branch_0_Conv2d_1x1_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_10_Relu)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_11_Branch_1_Conv2d_0a_1x1_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_11_Branch_1_Conv2d_0a_1x1_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_10_Relu)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_11_Branch_0_Conv2d_1x1_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_11_Branch_0_Conv2d_1x1_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_11_Branch_0_Conv2d_1x1_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_11_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_11_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_11_Branch_1_Conv2d_0a_1x1_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_11_Branch_0_Conv2d_1x1_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_11_Branch_0_Conv2d_1x1_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_11_Branch_1_Conv2d_0a_1x1_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_11_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_11_Branch_1_Conv2d_0b_1x7_Conv2D_pad = F.pad(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_11_Branch_1_Conv2d_0a_1x1_Relu, (3, 3, 0, 0))
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_11_Branch_1_Conv2d_0b_1x7_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_11_Branch_1_Conv2d_0b_1x7_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_11_Branch_1_Conv2d_0b_1x7_Conv2D_pad)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_11_Branch_1_Conv2d_0b_1x7_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_11_Branch_1_Conv2d_0b_1x7_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_11_Branch_1_Conv2d_0b_1x7_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_11_Branch_1_Conv2d_0b_1x7_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_11_Branch_1_Conv2d_0b_1x7_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_11_Branch_1_Conv2d_0c_7x1_Conv2D_pad = F.pad(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_11_Branch_1_Conv2d_0b_1x7_Relu, (0, 0, 3, 3))
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_11_Branch_1_Conv2d_0c_7x1_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_11_Branch_1_Conv2d_0c_7x1_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_11_Branch_1_Conv2d_0c_7x1_Conv2D_pad)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_11_Branch_1_Conv2d_0c_7x1_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_11_Branch_1_Conv2d_0c_7x1_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_11_Branch_1_Conv2d_0c_7x1_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_11_Branch_1_Conv2d_0c_7x1_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_11_Branch_1_Conv2d_0c_7x1_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_11_concat = torch.cat((
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_11_Branch_0_Conv2d_1x1_Relu,
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_11_Branch_1_Conv2d_0c_7x1_Relu,),
            1)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_11_Conv2d_1x1_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_11_Conv2d_1x1_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_11_concat)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_11_mul = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_11_mul_x * InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_11_Conv2d_1x1_Conv2D
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_11_add = InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_10_Relu + InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_11_mul
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_11_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_11_add)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_12_Branch_0_Conv2d_1x1_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_12_Branch_0_Conv2d_1x1_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_11_Relu)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_12_Branch_1_Conv2d_0a_1x1_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_12_Branch_1_Conv2d_0a_1x1_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_11_Relu)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_12_Branch_0_Conv2d_1x1_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_12_Branch_0_Conv2d_1x1_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_12_Branch_0_Conv2d_1x1_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_12_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_12_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_12_Branch_1_Conv2d_0a_1x1_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_12_Branch_0_Conv2d_1x1_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_12_Branch_0_Conv2d_1x1_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_12_Branch_1_Conv2d_0a_1x1_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_12_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_12_Branch_1_Conv2d_0b_1x7_Conv2D_pad = F.pad(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_12_Branch_1_Conv2d_0a_1x1_Relu, (3, 3, 0, 0))
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_12_Branch_1_Conv2d_0b_1x7_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_12_Branch_1_Conv2d_0b_1x7_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_12_Branch_1_Conv2d_0b_1x7_Conv2D_pad)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_12_Branch_1_Conv2d_0b_1x7_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_12_Branch_1_Conv2d_0b_1x7_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_12_Branch_1_Conv2d_0b_1x7_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_12_Branch_1_Conv2d_0b_1x7_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_12_Branch_1_Conv2d_0b_1x7_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_12_Branch_1_Conv2d_0c_7x1_Conv2D_pad = F.pad(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_12_Branch_1_Conv2d_0b_1x7_Relu, (0, 0, 3, 3))
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_12_Branch_1_Conv2d_0c_7x1_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_12_Branch_1_Conv2d_0c_7x1_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_12_Branch_1_Conv2d_0c_7x1_Conv2D_pad)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_12_Branch_1_Conv2d_0c_7x1_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_12_Branch_1_Conv2d_0c_7x1_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_12_Branch_1_Conv2d_0c_7x1_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_12_Branch_1_Conv2d_0c_7x1_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_12_Branch_1_Conv2d_0c_7x1_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_12_concat = torch.cat((
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_12_Branch_0_Conv2d_1x1_Relu,
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_12_Branch_1_Conv2d_0c_7x1_Relu,),
            1)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_12_Conv2d_1x1_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_12_Conv2d_1x1_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_12_concat)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_12_mul = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_12_mul_x * InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_12_Conv2d_1x1_Conv2D
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_12_add = InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_11_Relu + InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_12_mul
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_12_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_12_add)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_13_Branch_0_Conv2d_1x1_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_13_Branch_0_Conv2d_1x1_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_12_Relu)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_13_Branch_1_Conv2d_0a_1x1_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_13_Branch_1_Conv2d_0a_1x1_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_12_Relu)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_13_Branch_0_Conv2d_1x1_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_13_Branch_0_Conv2d_1x1_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_13_Branch_0_Conv2d_1x1_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_13_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_13_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_13_Branch_1_Conv2d_0a_1x1_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_13_Branch_0_Conv2d_1x1_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_13_Branch_0_Conv2d_1x1_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_13_Branch_1_Conv2d_0a_1x1_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_13_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_13_Branch_1_Conv2d_0b_1x7_Conv2D_pad = F.pad(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_13_Branch_1_Conv2d_0a_1x1_Relu, (3, 3, 0, 0))
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_13_Branch_1_Conv2d_0b_1x7_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_13_Branch_1_Conv2d_0b_1x7_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_13_Branch_1_Conv2d_0b_1x7_Conv2D_pad)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_13_Branch_1_Conv2d_0b_1x7_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_13_Branch_1_Conv2d_0b_1x7_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_13_Branch_1_Conv2d_0b_1x7_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_13_Branch_1_Conv2d_0b_1x7_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_13_Branch_1_Conv2d_0b_1x7_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_13_Branch_1_Conv2d_0c_7x1_Conv2D_pad = F.pad(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_13_Branch_1_Conv2d_0b_1x7_Relu, (0, 0, 3, 3))
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_13_Branch_1_Conv2d_0c_7x1_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_13_Branch_1_Conv2d_0c_7x1_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_13_Branch_1_Conv2d_0c_7x1_Conv2D_pad)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_13_Branch_1_Conv2d_0c_7x1_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_13_Branch_1_Conv2d_0c_7x1_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_13_Branch_1_Conv2d_0c_7x1_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_13_Branch_1_Conv2d_0c_7x1_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_13_Branch_1_Conv2d_0c_7x1_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_13_concat = torch.cat((
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_13_Branch_0_Conv2d_1x1_Relu,
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_13_Branch_1_Conv2d_0c_7x1_Relu,),
            1)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_13_Conv2d_1x1_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_13_Conv2d_1x1_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_13_concat)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_13_mul = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_13_mul_x * InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_13_Conv2d_1x1_Conv2D
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_13_add = InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_12_Relu + InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_13_mul
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_13_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_13_add)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_14_Branch_0_Conv2d_1x1_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_14_Branch_0_Conv2d_1x1_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_13_Relu)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_14_Branch_1_Conv2d_0a_1x1_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_14_Branch_1_Conv2d_0a_1x1_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_13_Relu)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_14_Branch_0_Conv2d_1x1_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_14_Branch_0_Conv2d_1x1_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_14_Branch_0_Conv2d_1x1_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_14_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_14_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_14_Branch_1_Conv2d_0a_1x1_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_14_Branch_0_Conv2d_1x1_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_14_Branch_0_Conv2d_1x1_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_14_Branch_1_Conv2d_0a_1x1_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_14_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_14_Branch_1_Conv2d_0b_1x7_Conv2D_pad = F.pad(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_14_Branch_1_Conv2d_0a_1x1_Relu, (3, 3, 0, 0))
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_14_Branch_1_Conv2d_0b_1x7_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_14_Branch_1_Conv2d_0b_1x7_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_14_Branch_1_Conv2d_0b_1x7_Conv2D_pad)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_14_Branch_1_Conv2d_0b_1x7_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_14_Branch_1_Conv2d_0b_1x7_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_14_Branch_1_Conv2d_0b_1x7_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_14_Branch_1_Conv2d_0b_1x7_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_14_Branch_1_Conv2d_0b_1x7_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_14_Branch_1_Conv2d_0c_7x1_Conv2D_pad = F.pad(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_14_Branch_1_Conv2d_0b_1x7_Relu, (0, 0, 3, 3))
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_14_Branch_1_Conv2d_0c_7x1_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_14_Branch_1_Conv2d_0c_7x1_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_14_Branch_1_Conv2d_0c_7x1_Conv2D_pad)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_14_Branch_1_Conv2d_0c_7x1_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_14_Branch_1_Conv2d_0c_7x1_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_14_Branch_1_Conv2d_0c_7x1_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_14_Branch_1_Conv2d_0c_7x1_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_14_Branch_1_Conv2d_0c_7x1_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_14_concat = torch.cat((
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_14_Branch_0_Conv2d_1x1_Relu,
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_14_Branch_1_Conv2d_0c_7x1_Relu,),
            1)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_14_Conv2d_1x1_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_14_Conv2d_1x1_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_14_concat)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_14_mul = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_14_mul_x * InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_14_Conv2d_1x1_Conv2D
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_14_add = InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_13_Relu + InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_14_mul
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_14_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_14_add)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_15_Branch_0_Conv2d_1x1_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_15_Branch_0_Conv2d_1x1_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_14_Relu)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_15_Branch_1_Conv2d_0a_1x1_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_15_Branch_1_Conv2d_0a_1x1_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_14_Relu)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_15_Branch_0_Conv2d_1x1_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_15_Branch_0_Conv2d_1x1_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_15_Branch_0_Conv2d_1x1_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_15_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_15_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_15_Branch_1_Conv2d_0a_1x1_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_15_Branch_0_Conv2d_1x1_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_15_Branch_0_Conv2d_1x1_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_15_Branch_1_Conv2d_0a_1x1_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_15_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_15_Branch_1_Conv2d_0b_1x7_Conv2D_pad = F.pad(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_15_Branch_1_Conv2d_0a_1x1_Relu, (3, 3, 0, 0))
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_15_Branch_1_Conv2d_0b_1x7_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_15_Branch_1_Conv2d_0b_1x7_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_15_Branch_1_Conv2d_0b_1x7_Conv2D_pad)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_15_Branch_1_Conv2d_0b_1x7_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_15_Branch_1_Conv2d_0b_1x7_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_15_Branch_1_Conv2d_0b_1x7_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_15_Branch_1_Conv2d_0b_1x7_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_15_Branch_1_Conv2d_0b_1x7_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_15_Branch_1_Conv2d_0c_7x1_Conv2D_pad = F.pad(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_15_Branch_1_Conv2d_0b_1x7_Relu, (0, 0, 3, 3))
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_15_Branch_1_Conv2d_0c_7x1_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_15_Branch_1_Conv2d_0c_7x1_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_15_Branch_1_Conv2d_0c_7x1_Conv2D_pad)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_15_Branch_1_Conv2d_0c_7x1_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_15_Branch_1_Conv2d_0c_7x1_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_15_Branch_1_Conv2d_0c_7x1_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_15_Branch_1_Conv2d_0c_7x1_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_15_Branch_1_Conv2d_0c_7x1_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_15_concat = torch.cat((
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_15_Branch_0_Conv2d_1x1_Relu,
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_15_Branch_1_Conv2d_0c_7x1_Relu,),
            1)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_15_Conv2d_1x1_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_15_Conv2d_1x1_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_15_concat)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_15_mul = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_15_mul_x * InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_15_Conv2d_1x1_Conv2D
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_15_add = InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_14_Relu + InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_15_mul
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_15_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_15_add)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_16_Branch_0_Conv2d_1x1_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_16_Branch_0_Conv2d_1x1_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_15_Relu)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_16_Branch_1_Conv2d_0a_1x1_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_16_Branch_1_Conv2d_0a_1x1_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_15_Relu)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_16_Branch_0_Conv2d_1x1_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_16_Branch_0_Conv2d_1x1_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_16_Branch_0_Conv2d_1x1_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_16_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_16_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_16_Branch_1_Conv2d_0a_1x1_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_16_Branch_0_Conv2d_1x1_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_16_Branch_0_Conv2d_1x1_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_16_Branch_1_Conv2d_0a_1x1_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_16_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_16_Branch_1_Conv2d_0b_1x7_Conv2D_pad = F.pad(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_16_Branch_1_Conv2d_0a_1x1_Relu, (3, 3, 0, 0))
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_16_Branch_1_Conv2d_0b_1x7_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_16_Branch_1_Conv2d_0b_1x7_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_16_Branch_1_Conv2d_0b_1x7_Conv2D_pad)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_16_Branch_1_Conv2d_0b_1x7_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_16_Branch_1_Conv2d_0b_1x7_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_16_Branch_1_Conv2d_0b_1x7_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_16_Branch_1_Conv2d_0b_1x7_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_16_Branch_1_Conv2d_0b_1x7_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_16_Branch_1_Conv2d_0c_7x1_Conv2D_pad = F.pad(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_16_Branch_1_Conv2d_0b_1x7_Relu, (0, 0, 3, 3))
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_16_Branch_1_Conv2d_0c_7x1_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_16_Branch_1_Conv2d_0c_7x1_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_16_Branch_1_Conv2d_0c_7x1_Conv2D_pad)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_16_Branch_1_Conv2d_0c_7x1_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_16_Branch_1_Conv2d_0c_7x1_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_16_Branch_1_Conv2d_0c_7x1_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_16_Branch_1_Conv2d_0c_7x1_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_16_Branch_1_Conv2d_0c_7x1_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_16_concat = torch.cat((
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_16_Branch_0_Conv2d_1x1_Relu,
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_16_Branch_1_Conv2d_0c_7x1_Relu,),
            1)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_16_Conv2d_1x1_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_16_Conv2d_1x1_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_16_concat)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_16_mul = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_16_mul_x * InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_16_Conv2d_1x1_Conv2D
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_16_add = InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_15_Relu + InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_16_mul
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_16_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_16_add)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_17_Branch_0_Conv2d_1x1_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_17_Branch_0_Conv2d_1x1_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_16_Relu)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_17_Branch_1_Conv2d_0a_1x1_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_17_Branch_1_Conv2d_0a_1x1_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_16_Relu)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_17_Branch_0_Conv2d_1x1_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_17_Branch_0_Conv2d_1x1_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_17_Branch_0_Conv2d_1x1_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_17_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_17_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_17_Branch_1_Conv2d_0a_1x1_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_17_Branch_0_Conv2d_1x1_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_17_Branch_0_Conv2d_1x1_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_17_Branch_1_Conv2d_0a_1x1_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_17_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_17_Branch_1_Conv2d_0b_1x7_Conv2D_pad = F.pad(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_17_Branch_1_Conv2d_0a_1x1_Relu, (3, 3, 0, 0))
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_17_Branch_1_Conv2d_0b_1x7_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_17_Branch_1_Conv2d_0b_1x7_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_17_Branch_1_Conv2d_0b_1x7_Conv2D_pad)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_17_Branch_1_Conv2d_0b_1x7_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_17_Branch_1_Conv2d_0b_1x7_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_17_Branch_1_Conv2d_0b_1x7_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_17_Branch_1_Conv2d_0b_1x7_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_17_Branch_1_Conv2d_0b_1x7_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_17_Branch_1_Conv2d_0c_7x1_Conv2D_pad = F.pad(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_17_Branch_1_Conv2d_0b_1x7_Relu, (0, 0, 3, 3))
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_17_Branch_1_Conv2d_0c_7x1_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_17_Branch_1_Conv2d_0c_7x1_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_17_Branch_1_Conv2d_0c_7x1_Conv2D_pad)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_17_Branch_1_Conv2d_0c_7x1_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_17_Branch_1_Conv2d_0c_7x1_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_17_Branch_1_Conv2d_0c_7x1_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_17_Branch_1_Conv2d_0c_7x1_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_17_Branch_1_Conv2d_0c_7x1_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_17_concat = torch.cat((
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_17_Branch_0_Conv2d_1x1_Relu,
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_17_Branch_1_Conv2d_0c_7x1_Relu,),
            1)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_17_Conv2d_1x1_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_17_Conv2d_1x1_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_17_concat)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_17_mul = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_17_mul_x * InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_17_Conv2d_1x1_Conv2D
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_17_add = InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_16_Relu + InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_17_mul
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_17_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_17_add)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_18_Branch_0_Conv2d_1x1_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_18_Branch_0_Conv2d_1x1_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_17_Relu)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_18_Branch_1_Conv2d_0a_1x1_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_18_Branch_1_Conv2d_0a_1x1_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_17_Relu)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_18_Branch_0_Conv2d_1x1_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_18_Branch_0_Conv2d_1x1_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_18_Branch_0_Conv2d_1x1_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_18_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_18_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_18_Branch_1_Conv2d_0a_1x1_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_18_Branch_0_Conv2d_1x1_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_18_Branch_0_Conv2d_1x1_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_18_Branch_1_Conv2d_0a_1x1_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_18_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_18_Branch_1_Conv2d_0b_1x7_Conv2D_pad = F.pad(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_18_Branch_1_Conv2d_0a_1x1_Relu, (3, 3, 0, 0))
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_18_Branch_1_Conv2d_0b_1x7_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_18_Branch_1_Conv2d_0b_1x7_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_18_Branch_1_Conv2d_0b_1x7_Conv2D_pad)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_18_Branch_1_Conv2d_0b_1x7_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_18_Branch_1_Conv2d_0b_1x7_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_18_Branch_1_Conv2d_0b_1x7_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_18_Branch_1_Conv2d_0b_1x7_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_18_Branch_1_Conv2d_0b_1x7_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_18_Branch_1_Conv2d_0c_7x1_Conv2D_pad = F.pad(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_18_Branch_1_Conv2d_0b_1x7_Relu, (0, 0, 3, 3))
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_18_Branch_1_Conv2d_0c_7x1_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_18_Branch_1_Conv2d_0c_7x1_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_18_Branch_1_Conv2d_0c_7x1_Conv2D_pad)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_18_Branch_1_Conv2d_0c_7x1_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_18_Branch_1_Conv2d_0c_7x1_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_18_Branch_1_Conv2d_0c_7x1_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_18_Branch_1_Conv2d_0c_7x1_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_18_Branch_1_Conv2d_0c_7x1_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_18_concat = torch.cat((
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_18_Branch_0_Conv2d_1x1_Relu,
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_18_Branch_1_Conv2d_0c_7x1_Relu,),
            1)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_18_Conv2d_1x1_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_18_Conv2d_1x1_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_18_concat)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_18_mul = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_18_mul_x * InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_18_Conv2d_1x1_Conv2D
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_18_add = InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_17_Relu + InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_18_mul
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_18_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_18_add)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_19_Branch_0_Conv2d_1x1_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_19_Branch_0_Conv2d_1x1_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_18_Relu)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_19_Branch_1_Conv2d_0a_1x1_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_19_Branch_1_Conv2d_0a_1x1_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_18_Relu)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_19_Branch_0_Conv2d_1x1_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_19_Branch_0_Conv2d_1x1_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_19_Branch_0_Conv2d_1x1_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_19_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_19_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_19_Branch_1_Conv2d_0a_1x1_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_19_Branch_0_Conv2d_1x1_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_19_Branch_0_Conv2d_1x1_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_19_Branch_1_Conv2d_0a_1x1_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_19_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_19_Branch_1_Conv2d_0b_1x7_Conv2D_pad = F.pad(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_19_Branch_1_Conv2d_0a_1x1_Relu, (3, 3, 0, 0))
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_19_Branch_1_Conv2d_0b_1x7_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_19_Branch_1_Conv2d_0b_1x7_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_19_Branch_1_Conv2d_0b_1x7_Conv2D_pad)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_19_Branch_1_Conv2d_0b_1x7_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_19_Branch_1_Conv2d_0b_1x7_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_19_Branch_1_Conv2d_0b_1x7_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_19_Branch_1_Conv2d_0b_1x7_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_19_Branch_1_Conv2d_0b_1x7_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_19_Branch_1_Conv2d_0c_7x1_Conv2D_pad = F.pad(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_19_Branch_1_Conv2d_0b_1x7_Relu, (0, 0, 3, 3))
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_19_Branch_1_Conv2d_0c_7x1_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_19_Branch_1_Conv2d_0c_7x1_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_19_Branch_1_Conv2d_0c_7x1_Conv2D_pad)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_19_Branch_1_Conv2d_0c_7x1_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_19_Branch_1_Conv2d_0c_7x1_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_19_Branch_1_Conv2d_0c_7x1_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_19_Branch_1_Conv2d_0c_7x1_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_19_Branch_1_Conv2d_0c_7x1_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_19_concat = torch.cat((
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_19_Branch_0_Conv2d_1x1_Relu,
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_19_Branch_1_Conv2d_0c_7x1_Relu,),
            1)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_19_Conv2d_1x1_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_19_Conv2d_1x1_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_19_concat)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_19_mul = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_19_mul_x * InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_19_Conv2d_1x1_Conv2D
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_19_add = InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_18_Relu + InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_19_mul
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_19_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_19_add)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_20_Branch_0_Conv2d_1x1_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_20_Branch_0_Conv2d_1x1_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_19_Relu)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_20_Branch_1_Conv2d_0a_1x1_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_20_Branch_1_Conv2d_0a_1x1_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_19_Relu)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_20_Branch_0_Conv2d_1x1_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_20_Branch_0_Conv2d_1x1_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_20_Branch_0_Conv2d_1x1_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_20_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_20_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_20_Branch_1_Conv2d_0a_1x1_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_20_Branch_0_Conv2d_1x1_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_20_Branch_0_Conv2d_1x1_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_20_Branch_1_Conv2d_0a_1x1_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_20_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_20_Branch_1_Conv2d_0b_1x7_Conv2D_pad = F.pad(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_20_Branch_1_Conv2d_0a_1x1_Relu, (3, 3, 0, 0))
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_20_Branch_1_Conv2d_0b_1x7_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_20_Branch_1_Conv2d_0b_1x7_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_20_Branch_1_Conv2d_0b_1x7_Conv2D_pad)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_20_Branch_1_Conv2d_0b_1x7_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_20_Branch_1_Conv2d_0b_1x7_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_20_Branch_1_Conv2d_0b_1x7_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_20_Branch_1_Conv2d_0b_1x7_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_20_Branch_1_Conv2d_0b_1x7_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_20_Branch_1_Conv2d_0c_7x1_Conv2D_pad = F.pad(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_20_Branch_1_Conv2d_0b_1x7_Relu, (0, 0, 3, 3))
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_20_Branch_1_Conv2d_0c_7x1_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_20_Branch_1_Conv2d_0c_7x1_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_20_Branch_1_Conv2d_0c_7x1_Conv2D_pad)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_20_Branch_1_Conv2d_0c_7x1_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_20_Branch_1_Conv2d_0c_7x1_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_20_Branch_1_Conv2d_0c_7x1_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_20_Branch_1_Conv2d_0c_7x1_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_20_Branch_1_Conv2d_0c_7x1_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_20_concat = torch.cat((
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_20_Branch_0_Conv2d_1x1_Relu,
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_20_Branch_1_Conv2d_0c_7x1_Relu,),
            1)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_20_Conv2d_1x1_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_20_Conv2d_1x1_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_20_concat)
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_20_mul = self.InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_20_mul_x * InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_20_Conv2d_1x1_Conv2D
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_20_add = InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_19_Relu + InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_20_mul
        InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_20_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_20_add)
        InceptionResnetV2_InceptionResnetV2_Mixed_7a_Branch_0_Conv2d_0a_1x1_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Mixed_7a_Branch_0_Conv2d_0a_1x1_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_20_Relu)
        InceptionResnetV2_InceptionResnetV2_Mixed_7a_Branch_1_Conv2d_0a_1x1_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Mixed_7a_Branch_1_Conv2d_0a_1x1_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_20_Relu)
        InceptionResnetV2_InceptionResnetV2_Mixed_7a_Branch_2_Conv2d_0a_1x1_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Mixed_7a_Branch_2_Conv2d_0a_1x1_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_20_Relu)
        InceptionResnetV2_InceptionResnetV2_Mixed_7a_Branch_3_MaxPool_1a_3x3_MaxPool, InceptionResnetV2_InceptionResnetV2_Mixed_7a_Branch_3_MaxPool_1a_3x3_MaxPool_idx = F.max_pool2d(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_20_Relu, kernel_size=(3, 3), stride=(2, 2), padding=0,
            ceil_mode=False, return_indices=True)
        InceptionResnetV2_AuxLogits_Conv2d_1a_3x3_AvgPool = F.avg_pool2d(
            InceptionResnetV2_InceptionResnetV2_Repeat_1_block17_20_Relu, kernel_size=(5, 5), stride=(3, 3),
            padding=(0,), ceil_mode=False, count_include_pad=False)
        InceptionResnetV2_InceptionResnetV2_Mixed_7a_Branch_0_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Mixed_7a_Branch_0_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Mixed_7a_Branch_0_Conv2d_0a_1x1_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Mixed_7a_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Mixed_7a_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Mixed_7a_Branch_1_Conv2d_0a_1x1_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Mixed_7a_Branch_2_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Mixed_7a_Branch_2_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Mixed_7a_Branch_2_Conv2d_0a_1x1_Conv2D)
        InceptionResnetV2_AuxLogits_Conv2d_1b_1x1_Conv2D = self.InceptionResnetV2_AuxLogits_Conv2d_1b_1x1_Conv2D(
            InceptionResnetV2_AuxLogits_Conv2d_1a_3x3_AvgPool)
        InceptionResnetV2_InceptionResnetV2_Mixed_7a_Branch_0_Conv2d_0a_1x1_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Mixed_7a_Branch_0_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Mixed_7a_Branch_1_Conv2d_0a_1x1_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Mixed_7a_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Mixed_7a_Branch_2_Conv2d_0a_1x1_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Mixed_7a_Branch_2_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_AuxLogits_Conv2d_1b_1x1_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_AuxLogits_Conv2d_1b_1x1_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_AuxLogits_Conv2d_1b_1x1_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Mixed_7a_Branch_0_Conv2d_1a_3x3_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Mixed_7a_Branch_0_Conv2d_1a_3x3_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Mixed_7a_Branch_0_Conv2d_0a_1x1_Relu)
        InceptionResnetV2_InceptionResnetV2_Mixed_7a_Branch_1_Conv2d_1a_3x3_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Mixed_7a_Branch_1_Conv2d_1a_3x3_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Mixed_7a_Branch_1_Conv2d_0a_1x1_Relu)
        InceptionResnetV2_InceptionResnetV2_Mixed_7a_Branch_2_Conv2d_0b_3x3_Conv2D_pad = F.pad(
            InceptionResnetV2_InceptionResnetV2_Mixed_7a_Branch_2_Conv2d_0a_1x1_Relu, (1, 1, 1, 1))
        InceptionResnetV2_InceptionResnetV2_Mixed_7a_Branch_2_Conv2d_0b_3x3_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Mixed_7a_Branch_2_Conv2d_0b_3x3_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Mixed_7a_Branch_2_Conv2d_0b_3x3_Conv2D_pad)
        InceptionResnetV2_AuxLogits_Conv2d_1b_1x1_Relu = F.relu(
            InceptionResnetV2_AuxLogits_Conv2d_1b_1x1_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Mixed_7a_Branch_0_Conv2d_1a_3x3_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Mixed_7a_Branch_0_Conv2d_1a_3x3_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Mixed_7a_Branch_0_Conv2d_1a_3x3_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Mixed_7a_Branch_1_Conv2d_1a_3x3_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Mixed_7a_Branch_1_Conv2d_1a_3x3_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Mixed_7a_Branch_1_Conv2d_1a_3x3_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Mixed_7a_Branch_2_Conv2d_0b_3x3_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Mixed_7a_Branch_2_Conv2d_0b_3x3_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Mixed_7a_Branch_2_Conv2d_0b_3x3_Conv2D)
        InceptionResnetV2_AuxLogits_Conv2d_2a_5x5_Conv2D = self.InceptionResnetV2_AuxLogits_Conv2d_2a_5x5_Conv2D(
            InceptionResnetV2_AuxLogits_Conv2d_1b_1x1_Relu)
        InceptionResnetV2_InceptionResnetV2_Mixed_7a_Branch_0_Conv2d_1a_3x3_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Mixed_7a_Branch_0_Conv2d_1a_3x3_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Mixed_7a_Branch_1_Conv2d_1a_3x3_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Mixed_7a_Branch_1_Conv2d_1a_3x3_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Mixed_7a_Branch_2_Conv2d_0b_3x3_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Mixed_7a_Branch_2_Conv2d_0b_3x3_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_AuxLogits_Conv2d_2a_5x5_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_AuxLogits_Conv2d_2a_5x5_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_AuxLogits_Conv2d_2a_5x5_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Mixed_7a_Branch_2_Conv2d_1a_3x3_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Mixed_7a_Branch_2_Conv2d_1a_3x3_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Mixed_7a_Branch_2_Conv2d_0b_3x3_Relu)
        InceptionResnetV2_AuxLogits_Conv2d_2a_5x5_Relu = F.relu(
            InceptionResnetV2_AuxLogits_Conv2d_2a_5x5_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Mixed_7a_Branch_2_Conv2d_1a_3x3_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Mixed_7a_Branch_2_Conv2d_1a_3x3_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Mixed_7a_Branch_2_Conv2d_1a_3x3_Conv2D)
        InceptionResnetV2_AuxLogits_Flatten_flatten_Shape = torch.Tensor(
            list(InceptionResnetV2_AuxLogits_Conv2d_2a_5x5_Relu.size()))
        InceptionResnetV2_AuxLogits_Flatten_flatten_Reshape = torch.reshape(
            input=InceptionResnetV2_AuxLogits_Conv2d_2a_5x5_Relu, shape=(-1, 768))
        InceptionResnetV2_InceptionResnetV2_Mixed_7a_Branch_2_Conv2d_1a_3x3_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Mixed_7a_Branch_2_Conv2d_1a_3x3_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_AuxLogits_Flatten_flatten_strided_slice = \
            InceptionResnetV2_AuxLogits_Flatten_flatten_Shape[0:1][0]
        InceptionResnetV2_AuxLogits_Logits_MatMul = self.InceptionResnetV2_AuxLogits_Logits_MatMul(
            InceptionResnetV2_AuxLogits_Flatten_flatten_Reshape)
        InceptionResnetV2_InceptionResnetV2_Mixed_7a_concat = torch.cat((
            InceptionResnetV2_InceptionResnetV2_Mixed_7a_Branch_0_Conv2d_1a_3x3_Relu,
            InceptionResnetV2_InceptionResnetV2_Mixed_7a_Branch_1_Conv2d_1a_3x3_Relu,
            InceptionResnetV2_InceptionResnetV2_Mixed_7a_Branch_2_Conv2d_1a_3x3_Relu,
            InceptionResnetV2_InceptionResnetV2_Mixed_7a_Branch_3_MaxPool_1a_3x3_MaxPool,),
            1)
        InceptionResnetV2_AuxLogits_Flatten_flatten_Reshape_shape = [
            InceptionResnetV2_AuxLogits_Flatten_flatten_strided_slice,
            self.InceptionResnetV2_AuxLogits_Flatten_flatten_Reshape_shape_1]
        InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_1_Branch_0_Conv2d_1x1_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_1_Branch_0_Conv2d_1x1_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Mixed_7a_concat)
        InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_1_Branch_1_Conv2d_0a_1x1_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_1_Branch_1_Conv2d_0a_1x1_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Mixed_7a_concat)
        InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_1_Branch_0_Conv2d_1x1_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_1_Branch_0_Conv2d_1x1_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_1_Branch_0_Conv2d_1x1_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_1_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_1_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_1_Branch_1_Conv2d_0a_1x1_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_1_Branch_0_Conv2d_1x1_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_1_Branch_0_Conv2d_1x1_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_1_Branch_1_Conv2d_0a_1x1_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_1_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_1_Branch_1_Conv2d_0b_1x3_Conv2D_pad = F.pad(
            InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_1_Branch_1_Conv2d_0a_1x1_Relu, (1, 1, 0, 0))
        InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_1_Branch_1_Conv2d_0b_1x3_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_1_Branch_1_Conv2d_0b_1x3_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_1_Branch_1_Conv2d_0b_1x3_Conv2D_pad)
        InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_1_Branch_1_Conv2d_0b_1x3_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_1_Branch_1_Conv2d_0b_1x3_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_1_Branch_1_Conv2d_0b_1x3_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_1_Branch_1_Conv2d_0b_1x3_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_1_Branch_1_Conv2d_0b_1x3_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_1_Branch_1_Conv2d_0c_3x1_Conv2D_pad = F.pad(
            InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_1_Branch_1_Conv2d_0b_1x3_Relu, (0, 0, 1, 1))
        InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_1_Branch_1_Conv2d_0c_3x1_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_1_Branch_1_Conv2d_0c_3x1_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_1_Branch_1_Conv2d_0c_3x1_Conv2D_pad)
        InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_1_Branch_1_Conv2d_0c_3x1_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_1_Branch_1_Conv2d_0c_3x1_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_1_Branch_1_Conv2d_0c_3x1_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_1_Branch_1_Conv2d_0c_3x1_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_1_Branch_1_Conv2d_0c_3x1_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_1_concat = torch.cat((
            InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_1_Branch_0_Conv2d_1x1_Relu,
            InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_1_Branch_1_Conv2d_0c_3x1_Relu,),
            1)
        InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_1_Conv2d_1x1_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_1_Conv2d_1x1_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_1_concat)
        InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_1_mul = self.InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_1_mul_x * InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_1_Conv2d_1x1_Conv2D
        InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_1_add = InceptionResnetV2_InceptionResnetV2_Mixed_7a_concat + InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_1_mul
        InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_1_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_1_add)
        InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_2_Branch_0_Conv2d_1x1_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_2_Branch_0_Conv2d_1x1_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_1_Relu)
        InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_2_Branch_1_Conv2d_0a_1x1_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_2_Branch_1_Conv2d_0a_1x1_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_1_Relu)
        InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_2_Branch_0_Conv2d_1x1_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_2_Branch_0_Conv2d_1x1_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_2_Branch_0_Conv2d_1x1_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_2_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_2_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_2_Branch_1_Conv2d_0a_1x1_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_2_Branch_0_Conv2d_1x1_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_2_Branch_0_Conv2d_1x1_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_2_Branch_1_Conv2d_0a_1x1_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_2_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_2_Branch_1_Conv2d_0b_1x3_Conv2D_pad = F.pad(
            InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_2_Branch_1_Conv2d_0a_1x1_Relu, (1, 1, 0, 0))
        InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_2_Branch_1_Conv2d_0b_1x3_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_2_Branch_1_Conv2d_0b_1x3_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_2_Branch_1_Conv2d_0b_1x3_Conv2D_pad)
        InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_2_Branch_1_Conv2d_0b_1x3_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_2_Branch_1_Conv2d_0b_1x3_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_2_Branch_1_Conv2d_0b_1x3_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_2_Branch_1_Conv2d_0b_1x3_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_2_Branch_1_Conv2d_0b_1x3_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_2_Branch_1_Conv2d_0c_3x1_Conv2D_pad = F.pad(
            InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_2_Branch_1_Conv2d_0b_1x3_Relu, (0, 0, 1, 1))
        InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_2_Branch_1_Conv2d_0c_3x1_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_2_Branch_1_Conv2d_0c_3x1_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_2_Branch_1_Conv2d_0c_3x1_Conv2D_pad)
        InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_2_Branch_1_Conv2d_0c_3x1_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_2_Branch_1_Conv2d_0c_3x1_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_2_Branch_1_Conv2d_0c_3x1_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_2_Branch_1_Conv2d_0c_3x1_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_2_Branch_1_Conv2d_0c_3x1_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_2_concat = torch.cat((
            InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_2_Branch_0_Conv2d_1x1_Relu,
            InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_2_Branch_1_Conv2d_0c_3x1_Relu,),
            1)
        InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_2_Conv2d_1x1_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_2_Conv2d_1x1_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_2_concat)
        InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_2_mul = self.InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_2_mul_x * InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_2_Conv2d_1x1_Conv2D
        InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_2_add = InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_1_Relu + InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_2_mul
        InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_2_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_2_add)
        InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_3_Branch_0_Conv2d_1x1_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_3_Branch_0_Conv2d_1x1_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_2_Relu)
        InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_3_Branch_1_Conv2d_0a_1x1_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_3_Branch_1_Conv2d_0a_1x1_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_2_Relu)
        InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_3_Branch_0_Conv2d_1x1_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_3_Branch_0_Conv2d_1x1_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_3_Branch_0_Conv2d_1x1_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_3_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_3_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_3_Branch_1_Conv2d_0a_1x1_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_3_Branch_0_Conv2d_1x1_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_3_Branch_0_Conv2d_1x1_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_3_Branch_1_Conv2d_0a_1x1_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_3_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_3_Branch_1_Conv2d_0b_1x3_Conv2D_pad = F.pad(
            InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_3_Branch_1_Conv2d_0a_1x1_Relu, (1, 1, 0, 0))
        InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_3_Branch_1_Conv2d_0b_1x3_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_3_Branch_1_Conv2d_0b_1x3_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_3_Branch_1_Conv2d_0b_1x3_Conv2D_pad)
        InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_3_Branch_1_Conv2d_0b_1x3_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_3_Branch_1_Conv2d_0b_1x3_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_3_Branch_1_Conv2d_0b_1x3_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_3_Branch_1_Conv2d_0b_1x3_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_3_Branch_1_Conv2d_0b_1x3_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_3_Branch_1_Conv2d_0c_3x1_Conv2D_pad = F.pad(
            InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_3_Branch_1_Conv2d_0b_1x3_Relu, (0, 0, 1, 1))
        InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_3_Branch_1_Conv2d_0c_3x1_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_3_Branch_1_Conv2d_0c_3x1_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_3_Branch_1_Conv2d_0c_3x1_Conv2D_pad)
        InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_3_Branch_1_Conv2d_0c_3x1_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_3_Branch_1_Conv2d_0c_3x1_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_3_Branch_1_Conv2d_0c_3x1_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_3_Branch_1_Conv2d_0c_3x1_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_3_Branch_1_Conv2d_0c_3x1_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_3_concat = torch.cat((
            InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_3_Branch_0_Conv2d_1x1_Relu,
            InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_3_Branch_1_Conv2d_0c_3x1_Relu,),
            1)
        InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_3_Conv2d_1x1_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_3_Conv2d_1x1_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_3_concat)
        InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_3_mul = self.InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_3_mul_x * InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_3_Conv2d_1x1_Conv2D
        InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_3_add = InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_2_Relu + InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_3_mul
        InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_3_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_3_add)
        InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_4_Branch_0_Conv2d_1x1_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_4_Branch_0_Conv2d_1x1_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_3_Relu)
        InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_4_Branch_1_Conv2d_0a_1x1_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_4_Branch_1_Conv2d_0a_1x1_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_3_Relu)
        InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_4_Branch_0_Conv2d_1x1_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_4_Branch_0_Conv2d_1x1_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_4_Branch_0_Conv2d_1x1_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_4_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_4_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_4_Branch_1_Conv2d_0a_1x1_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_4_Branch_0_Conv2d_1x1_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_4_Branch_0_Conv2d_1x1_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_4_Branch_1_Conv2d_0a_1x1_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_4_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_4_Branch_1_Conv2d_0b_1x3_Conv2D_pad = F.pad(
            InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_4_Branch_1_Conv2d_0a_1x1_Relu, (1, 1, 0, 0))
        InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_4_Branch_1_Conv2d_0b_1x3_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_4_Branch_1_Conv2d_0b_1x3_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_4_Branch_1_Conv2d_0b_1x3_Conv2D_pad)
        InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_4_Branch_1_Conv2d_0b_1x3_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_4_Branch_1_Conv2d_0b_1x3_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_4_Branch_1_Conv2d_0b_1x3_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_4_Branch_1_Conv2d_0b_1x3_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_4_Branch_1_Conv2d_0b_1x3_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_4_Branch_1_Conv2d_0c_3x1_Conv2D_pad = F.pad(
            InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_4_Branch_1_Conv2d_0b_1x3_Relu, (0, 0, 1, 1))
        InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_4_Branch_1_Conv2d_0c_3x1_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_4_Branch_1_Conv2d_0c_3x1_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_4_Branch_1_Conv2d_0c_3x1_Conv2D_pad)
        InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_4_Branch_1_Conv2d_0c_3x1_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_4_Branch_1_Conv2d_0c_3x1_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_4_Branch_1_Conv2d_0c_3x1_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_4_Branch_1_Conv2d_0c_3x1_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_4_Branch_1_Conv2d_0c_3x1_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_4_concat = torch.cat((
            InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_4_Branch_0_Conv2d_1x1_Relu,
            InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_4_Branch_1_Conv2d_0c_3x1_Relu,),
            1)
        InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_4_Conv2d_1x1_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_4_Conv2d_1x1_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_4_concat)
        InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_4_mul = self.InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_4_mul_x * InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_4_Conv2d_1x1_Conv2D
        InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_4_add = InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_3_Relu + InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_4_mul
        InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_4_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_4_add)
        InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_5_Branch_0_Conv2d_1x1_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_5_Branch_0_Conv2d_1x1_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_4_Relu)
        InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_5_Branch_1_Conv2d_0a_1x1_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_5_Branch_1_Conv2d_0a_1x1_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_4_Relu)
        InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_5_Branch_0_Conv2d_1x1_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_5_Branch_0_Conv2d_1x1_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_5_Branch_0_Conv2d_1x1_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_5_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_5_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_5_Branch_1_Conv2d_0a_1x1_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_5_Branch_0_Conv2d_1x1_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_5_Branch_0_Conv2d_1x1_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_5_Branch_1_Conv2d_0a_1x1_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_5_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_5_Branch_1_Conv2d_0b_1x3_Conv2D_pad = F.pad(
            InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_5_Branch_1_Conv2d_0a_1x1_Relu, (1, 1, 0, 0))
        InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_5_Branch_1_Conv2d_0b_1x3_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_5_Branch_1_Conv2d_0b_1x3_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_5_Branch_1_Conv2d_0b_1x3_Conv2D_pad)
        InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_5_Branch_1_Conv2d_0b_1x3_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_5_Branch_1_Conv2d_0b_1x3_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_5_Branch_1_Conv2d_0b_1x3_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_5_Branch_1_Conv2d_0b_1x3_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_5_Branch_1_Conv2d_0b_1x3_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_5_Branch_1_Conv2d_0c_3x1_Conv2D_pad = F.pad(
            InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_5_Branch_1_Conv2d_0b_1x3_Relu, (0, 0, 1, 1))
        InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_5_Branch_1_Conv2d_0c_3x1_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_5_Branch_1_Conv2d_0c_3x1_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_5_Branch_1_Conv2d_0c_3x1_Conv2D_pad)
        InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_5_Branch_1_Conv2d_0c_3x1_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_5_Branch_1_Conv2d_0c_3x1_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_5_Branch_1_Conv2d_0c_3x1_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_5_Branch_1_Conv2d_0c_3x1_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_5_Branch_1_Conv2d_0c_3x1_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_5_concat = torch.cat((
            InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_5_Branch_0_Conv2d_1x1_Relu,
            InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_5_Branch_1_Conv2d_0c_3x1_Relu,),
            1)
        InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_5_Conv2d_1x1_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_5_Conv2d_1x1_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_5_concat)
        InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_5_mul = self.InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_5_mul_x * InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_5_Conv2d_1x1_Conv2D
        InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_5_add = InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_4_Relu + InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_5_mul
        InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_5_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_5_add)
        InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_6_Branch_0_Conv2d_1x1_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_6_Branch_0_Conv2d_1x1_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_5_Relu)
        InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_6_Branch_1_Conv2d_0a_1x1_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_6_Branch_1_Conv2d_0a_1x1_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_5_Relu)
        InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_6_Branch_0_Conv2d_1x1_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_6_Branch_0_Conv2d_1x1_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_6_Branch_0_Conv2d_1x1_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_6_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_6_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_6_Branch_1_Conv2d_0a_1x1_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_6_Branch_0_Conv2d_1x1_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_6_Branch_0_Conv2d_1x1_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_6_Branch_1_Conv2d_0a_1x1_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_6_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_6_Branch_1_Conv2d_0b_1x3_Conv2D_pad = F.pad(
            InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_6_Branch_1_Conv2d_0a_1x1_Relu, (1, 1, 0, 0))
        InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_6_Branch_1_Conv2d_0b_1x3_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_6_Branch_1_Conv2d_0b_1x3_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_6_Branch_1_Conv2d_0b_1x3_Conv2D_pad)
        InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_6_Branch_1_Conv2d_0b_1x3_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_6_Branch_1_Conv2d_0b_1x3_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_6_Branch_1_Conv2d_0b_1x3_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_6_Branch_1_Conv2d_0b_1x3_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_6_Branch_1_Conv2d_0b_1x3_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_6_Branch_1_Conv2d_0c_3x1_Conv2D_pad = F.pad(
            InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_6_Branch_1_Conv2d_0b_1x3_Relu, (0, 0, 1, 1))
        InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_6_Branch_1_Conv2d_0c_3x1_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_6_Branch_1_Conv2d_0c_3x1_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_6_Branch_1_Conv2d_0c_3x1_Conv2D_pad)
        InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_6_Branch_1_Conv2d_0c_3x1_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_6_Branch_1_Conv2d_0c_3x1_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_6_Branch_1_Conv2d_0c_3x1_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_6_Branch_1_Conv2d_0c_3x1_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_6_Branch_1_Conv2d_0c_3x1_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_6_concat = torch.cat((
            InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_6_Branch_0_Conv2d_1x1_Relu,
            InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_6_Branch_1_Conv2d_0c_3x1_Relu,),
            1)
        InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_6_Conv2d_1x1_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_6_Conv2d_1x1_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_6_concat)
        InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_6_mul = self.InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_6_mul_x * InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_6_Conv2d_1x1_Conv2D
        InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_6_add = InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_5_Relu + InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_6_mul
        InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_6_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_6_add)
        InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_7_Branch_0_Conv2d_1x1_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_7_Branch_0_Conv2d_1x1_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_6_Relu)
        InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_7_Branch_1_Conv2d_0a_1x1_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_7_Branch_1_Conv2d_0a_1x1_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_6_Relu)
        InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_7_Branch_0_Conv2d_1x1_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_7_Branch_0_Conv2d_1x1_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_7_Branch_0_Conv2d_1x1_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_7_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_7_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_7_Branch_1_Conv2d_0a_1x1_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_7_Branch_0_Conv2d_1x1_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_7_Branch_0_Conv2d_1x1_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_7_Branch_1_Conv2d_0a_1x1_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_7_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_7_Branch_1_Conv2d_0b_1x3_Conv2D_pad = F.pad(
            InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_7_Branch_1_Conv2d_0a_1x1_Relu, (1, 1, 0, 0))
        InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_7_Branch_1_Conv2d_0b_1x3_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_7_Branch_1_Conv2d_0b_1x3_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_7_Branch_1_Conv2d_0b_1x3_Conv2D_pad)
        InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_7_Branch_1_Conv2d_0b_1x3_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_7_Branch_1_Conv2d_0b_1x3_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_7_Branch_1_Conv2d_0b_1x3_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_7_Branch_1_Conv2d_0b_1x3_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_7_Branch_1_Conv2d_0b_1x3_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_7_Branch_1_Conv2d_0c_3x1_Conv2D_pad = F.pad(
            InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_7_Branch_1_Conv2d_0b_1x3_Relu, (0, 0, 1, 1))
        InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_7_Branch_1_Conv2d_0c_3x1_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_7_Branch_1_Conv2d_0c_3x1_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_7_Branch_1_Conv2d_0c_3x1_Conv2D_pad)
        InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_7_Branch_1_Conv2d_0c_3x1_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_7_Branch_1_Conv2d_0c_3x1_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_7_Branch_1_Conv2d_0c_3x1_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_7_Branch_1_Conv2d_0c_3x1_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_7_Branch_1_Conv2d_0c_3x1_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_7_concat = torch.cat((
            InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_7_Branch_0_Conv2d_1x1_Relu,
            InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_7_Branch_1_Conv2d_0c_3x1_Relu,),
            1)
        InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_7_Conv2d_1x1_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_7_Conv2d_1x1_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_7_concat)
        InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_7_mul = self.InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_7_mul_x * InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_7_Conv2d_1x1_Conv2D
        InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_7_add = InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_6_Relu + InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_7_mul
        InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_7_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_7_add)
        InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_8_Branch_0_Conv2d_1x1_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_8_Branch_0_Conv2d_1x1_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_7_Relu)
        InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_8_Branch_1_Conv2d_0a_1x1_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_8_Branch_1_Conv2d_0a_1x1_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_7_Relu)
        InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_8_Branch_0_Conv2d_1x1_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_8_Branch_0_Conv2d_1x1_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_8_Branch_0_Conv2d_1x1_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_8_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_8_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_8_Branch_1_Conv2d_0a_1x1_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_8_Branch_0_Conv2d_1x1_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_8_Branch_0_Conv2d_1x1_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_8_Branch_1_Conv2d_0a_1x1_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_8_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_8_Branch_1_Conv2d_0b_1x3_Conv2D_pad = F.pad(
            InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_8_Branch_1_Conv2d_0a_1x1_Relu, (1, 1, 0, 0))
        InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_8_Branch_1_Conv2d_0b_1x3_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_8_Branch_1_Conv2d_0b_1x3_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_8_Branch_1_Conv2d_0b_1x3_Conv2D_pad)
        InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_8_Branch_1_Conv2d_0b_1x3_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_8_Branch_1_Conv2d_0b_1x3_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_8_Branch_1_Conv2d_0b_1x3_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_8_Branch_1_Conv2d_0b_1x3_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_8_Branch_1_Conv2d_0b_1x3_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_8_Branch_1_Conv2d_0c_3x1_Conv2D_pad = F.pad(
            InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_8_Branch_1_Conv2d_0b_1x3_Relu, (0, 0, 1, 1))
        InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_8_Branch_1_Conv2d_0c_3x1_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_8_Branch_1_Conv2d_0c_3x1_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_8_Branch_1_Conv2d_0c_3x1_Conv2D_pad)
        InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_8_Branch_1_Conv2d_0c_3x1_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_8_Branch_1_Conv2d_0c_3x1_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_8_Branch_1_Conv2d_0c_3x1_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_8_Branch_1_Conv2d_0c_3x1_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_8_Branch_1_Conv2d_0c_3x1_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_8_concat = torch.cat((
            InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_8_Branch_0_Conv2d_1x1_Relu,
            InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_8_Branch_1_Conv2d_0c_3x1_Relu,),
            1)
        InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_8_Conv2d_1x1_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_8_Conv2d_1x1_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_8_concat)
        InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_8_mul = self.InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_8_mul_x * InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_8_Conv2d_1x1_Conv2D
        InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_8_add = InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_7_Relu + InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_8_mul
        InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_8_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_8_add)
        InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_9_Branch_0_Conv2d_1x1_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_9_Branch_0_Conv2d_1x1_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_8_Relu)
        InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_9_Branch_1_Conv2d_0a_1x1_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_9_Branch_1_Conv2d_0a_1x1_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_8_Relu)
        InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_9_Branch_0_Conv2d_1x1_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_9_Branch_0_Conv2d_1x1_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_9_Branch_0_Conv2d_1x1_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_9_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_9_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_9_Branch_1_Conv2d_0a_1x1_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_9_Branch_0_Conv2d_1x1_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_9_Branch_0_Conv2d_1x1_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_9_Branch_1_Conv2d_0a_1x1_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_9_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_9_Branch_1_Conv2d_0b_1x3_Conv2D_pad = F.pad(
            InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_9_Branch_1_Conv2d_0a_1x1_Relu, (1, 1, 0, 0))
        InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_9_Branch_1_Conv2d_0b_1x3_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_9_Branch_1_Conv2d_0b_1x3_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_9_Branch_1_Conv2d_0b_1x3_Conv2D_pad)
        InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_9_Branch_1_Conv2d_0b_1x3_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_9_Branch_1_Conv2d_0b_1x3_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_9_Branch_1_Conv2d_0b_1x3_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_9_Branch_1_Conv2d_0b_1x3_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_9_Branch_1_Conv2d_0b_1x3_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_9_Branch_1_Conv2d_0c_3x1_Conv2D_pad = F.pad(
            InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_9_Branch_1_Conv2d_0b_1x3_Relu, (0, 0, 1, 1))
        InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_9_Branch_1_Conv2d_0c_3x1_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_9_Branch_1_Conv2d_0c_3x1_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_9_Branch_1_Conv2d_0c_3x1_Conv2D_pad)
        InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_9_Branch_1_Conv2d_0c_3x1_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_9_Branch_1_Conv2d_0c_3x1_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_9_Branch_1_Conv2d_0c_3x1_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_9_Branch_1_Conv2d_0c_3x1_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_9_Branch_1_Conv2d_0c_3x1_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_9_concat = torch.cat((
            InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_9_Branch_0_Conv2d_1x1_Relu,
            InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_9_Branch_1_Conv2d_0c_3x1_Relu,),
            1)
        InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_9_Conv2d_1x1_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_9_Conv2d_1x1_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_9_concat)
        InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_9_mul = self.InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_9_mul_x * InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_9_Conv2d_1x1_Conv2D
        InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_9_add = InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_8_Relu + InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_9_mul
        InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_9_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_9_add)
        InceptionResnetV2_InceptionResnetV2_Block8_Branch_0_Conv2d_1x1_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Block8_Branch_0_Conv2d_1x1_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_9_Relu)
        InceptionResnetV2_InceptionResnetV2_Block8_Branch_1_Conv2d_0a_1x1_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Block8_Branch_1_Conv2d_0a_1x1_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_9_Relu)
        InceptionResnetV2_InceptionResnetV2_Block8_Branch_0_Conv2d_1x1_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Block8_Branch_0_Conv2d_1x1_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Block8_Branch_0_Conv2d_1x1_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Block8_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Block8_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Block8_Branch_1_Conv2d_0a_1x1_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Block8_Branch_0_Conv2d_1x1_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Block8_Branch_0_Conv2d_1x1_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Block8_Branch_1_Conv2d_0a_1x1_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Block8_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Block8_Branch_1_Conv2d_0b_1x3_Conv2D_pad = F.pad(
            InceptionResnetV2_InceptionResnetV2_Block8_Branch_1_Conv2d_0a_1x1_Relu, (1, 1, 0, 0))
        InceptionResnetV2_InceptionResnetV2_Block8_Branch_1_Conv2d_0b_1x3_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Block8_Branch_1_Conv2d_0b_1x3_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Block8_Branch_1_Conv2d_0b_1x3_Conv2D_pad)
        InceptionResnetV2_InceptionResnetV2_Block8_Branch_1_Conv2d_0b_1x3_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Block8_Branch_1_Conv2d_0b_1x3_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Block8_Branch_1_Conv2d_0b_1x3_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Block8_Branch_1_Conv2d_0b_1x3_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Block8_Branch_1_Conv2d_0b_1x3_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Block8_Branch_1_Conv2d_0c_3x1_Conv2D_pad = F.pad(
            InceptionResnetV2_InceptionResnetV2_Block8_Branch_1_Conv2d_0b_1x3_Relu, (0, 0, 1, 1))
        InceptionResnetV2_InceptionResnetV2_Block8_Branch_1_Conv2d_0c_3x1_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Block8_Branch_1_Conv2d_0c_3x1_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Block8_Branch_1_Conv2d_0c_3x1_Conv2D_pad)
        InceptionResnetV2_InceptionResnetV2_Block8_Branch_1_Conv2d_0c_3x1_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Block8_Branch_1_Conv2d_0c_3x1_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Block8_Branch_1_Conv2d_0c_3x1_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Block8_Branch_1_Conv2d_0c_3x1_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Block8_Branch_1_Conv2d_0c_3x1_BatchNorm_FusedBatchNorm)
        InceptionResnetV2_InceptionResnetV2_Block8_concat = torch.cat((
            InceptionResnetV2_InceptionResnetV2_Block8_Branch_0_Conv2d_1x1_Relu,
            InceptionResnetV2_InceptionResnetV2_Block8_Branch_1_Conv2d_0c_3x1_Relu,),
            1)
        InceptionResnetV2_InceptionResnetV2_Block8_Conv2d_1x1_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Block8_Conv2d_1x1_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Block8_concat)
        InceptionResnetV2_InceptionResnetV2_Block8_mul = self.InceptionResnetV2_InceptionResnetV2_Block8_mul_x * InceptionResnetV2_InceptionResnetV2_Block8_Conv2d_1x1_Conv2D
        InceptionResnetV2_InceptionResnetV2_Block8_add = InceptionResnetV2_InceptionResnetV2_Repeat_2_block8_9_Relu + InceptionResnetV2_InceptionResnetV2_Block8_mul
        InceptionResnetV2_InceptionResnetV2_Conv2d_7b_1x1_Conv2D = self.InceptionResnetV2_InceptionResnetV2_Conv2d_7b_1x1_Conv2D(
            InceptionResnetV2_InceptionResnetV2_Block8_add)
        InceptionResnetV2_InceptionResnetV2_Conv2d_7b_1x1_BatchNorm_FusedBatchNorm = self.InceptionResnetV2_InceptionResnetV2_Conv2d_7b_1x1_BatchNorm_FusedBatchNorm(
            InceptionResnetV2_InceptionResnetV2_Conv2d_7b_1x1_Conv2D)
        InceptionResnetV2_InceptionResnetV2_Conv2d_7b_1x1_Relu = F.relu(
            InceptionResnetV2_InceptionResnetV2_Conv2d_7b_1x1_BatchNorm_FusedBatchNorm)
        kernel_size = self._reduced_kernel_size_for_small_input(InceptionResnetV2_InceptionResnetV2_Conv2d_7b_1x1_Relu,
                                                                [8, 8])
        InceptionResnetV2_Logits_AvgPool_1a_8x8_AvgPool = F.avg_pool2d(
            InceptionResnetV2_InceptionResnetV2_Conv2d_7b_1x1_Relu, kernel_size=(kernel_size[0], kernel_size[1]),
            stride=(2, 2), padding=(0,), ceil_mode=False, count_include_pad=False)
        InceptionResnetV2_Logits_Flatten_flatten_Shape = torch.Tensor(
            list(InceptionResnetV2_Logits_AvgPool_1a_8x8_AvgPool.size()))
        InceptionResnetV2_Logits_Flatten_flatten_Reshape = torch.reshape(
            input=InceptionResnetV2_Logits_AvgPool_1a_8x8_AvgPool, shape=(-1, 1536))
        InceptionResnetV2_Logits_Flatten_flatten_strided_slice = InceptionResnetV2_Logits_Flatten_flatten_Shape[0:1][0]
        InceptionResnetV2_Logits_Logits_MatMul = self.InceptionResnetV2_Logits_Logits_MatMul(
            InceptionResnetV2_Logits_Flatten_flatten_Reshape)
        InceptionResnetV2_Logits_Flatten_flatten_Reshape_shape = [
            InceptionResnetV2_Logits_Flatten_flatten_strided_slice,
            self.InceptionResnetV2_Logits_Flatten_flatten_Reshape_shape_1]
        MMdnn_Output_input = [InceptionResnetV2_Logits_Logits_MatMul, InceptionResnetV2_AuxLogits_Logits_MatMul]
        result = InceptionResnetV2_Logits_Logits_MatMul[:, 1:]
        return result

    def _reduced_kernel_size_for_small_input(self, input_tensor, kernel_size):
        """Define kernel size which is automatically reduced for small input.

        If the shape of the input images is unknown at graph construction time this
        function assumes that the input images are is large enough.

        Args:
            input_tensor: input tensor of size [batch_size, height, width, channels].
            kernel_size: desired kernel size of length 2: [kernel_height, kernel_width]

        Returns:
            a tensor with the kernel size.

        """
        shape = input_tensor.shape
        if shape[2] is None or shape[3] is None:
            kernel_size_out = kernel_size
        else:
            kernel_size_out = [min(shape[2], kernel_size[0]),
                               min(shape[3], kernel_size[1])]
        return kernel_size_out

    @staticmethod
    def __conv(dim, name, **kwargs):
        if dim == 1:
            layer = nn.Conv1d(**kwargs)
        elif dim == 2:
            layer = nn.Conv2d(**kwargs)
        elif dim == 3:
            layer = nn.Conv3d(**kwargs)
        else:
            raise NotImplementedError()

        layer.state_dict()['weight'].copy_(torch.from_numpy(_weights_dict[name]['weights']))
        if 'bias' in _weights_dict[name]:
            layer.state_dict()['bias'].copy_(torch.from_numpy(_weights_dict[name]['bias']))
        return layer

    @staticmethod
    def __dense(name, **kwargs):
        layer = nn.Linear(**kwargs)
        layer.state_dict()['weight'].copy_(torch.from_numpy(_weights_dict[name]['weights']))
        if 'bias' in _weights_dict[name]:
            layer.state_dict()['bias'].copy_(torch.from_numpy(_weights_dict[name]['bias']))
        return layer

    @staticmethod
    def __batch_normalization(dim, name, **kwargs):
        if dim == 0 or dim == 1:
            layer = nn.BatchNorm1d(**kwargs)
        elif dim == 2:
            layer = nn.BatchNorm2d(**kwargs)
        elif dim == 3:
            layer = nn.BatchNorm3d(**kwargs)
        else:
            raise NotImplementedError()

        if 'scale' in _weights_dict[name]:
            layer.state_dict()['weight'].copy_(torch.from_numpy(_weights_dict[name]['scale']))
        else:
            layer.weight.data.fill_(1)

        if 'bias' in _weights_dict[name]:
            layer.state_dict()['bias'].copy_(torch.from_numpy(_weights_dict[name]['bias']))
        else:
            layer.bias.data.fill_(0)

        layer.state_dict()['running_mean'].copy_(torch.from_numpy(_weights_dict[name]['mean']))
        layer.state_dict()['running_var'].copy_(torch.from_numpy(_weights_dict[name]['var']))
        return layer
