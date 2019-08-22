from functools import reduce
from utils import *
from CalcFeatureVec import FeatureVec
import numpy as np
from LoadCSV import LoadCSV
from CalcFeatureVec import FeatureVec

# 这里要定义一个方法，得到全局的特征向量用于归一化，输入为所有特征向量，最后输出max, min
class FeatureOverall(object):

    def __init__(self, features):

        # 只有一个特征点的
        l1=[]
        # 只有两个特征点的
        l2=[]
        # 有三个及以上特征点的
        l3=[]

        for l in features:
            if l.length == 1:
                l1.append(FeatureVec(l.x_fv, l.y_fv, l.featuredirect).feature_vec_common)
            elif l.length == 2:
                l2.append(FeatureVec(l.x_fv, l.y_fv, l.featuredirect).feature_vec_common)
            else:
                l3.append(FeatureVec(l.x_fv, l.y_fv, l.featuredirect).feature_vec_common)

        result = reduce(vstack, l1+l2+l3)
        self.max = result.max(axis=0)
        result_alt2 = reduce(vstack, l2+l3)[:, 1:3]
        result_alt3 = reduce(vstack, l3)[:, -1]
        self.min = np.zeros((1, 4))
        self.min[0][0] = result[:, 0].min(axis=0)
        self.min[0][1:3] = result_alt2.min(axis = 0)
        self.min[0][-1] = result_alt3.min(axis = 0)


class Normalization(object):
    # 归一化
    def __init__(self, max, min):
        '''
        :param max:最大值向量
        :param min: 最小值向量
        '''
        self.max = max
        self.min = min
        self.delta = max - min

    def normalize(self, matrix):
        norm_matrix = np.zeros((len(matrix), 4))
        # 只有一个特征，特征矩阵只有一行，且只有第一列有数据
        if len(matrix) == 1:
            norm_matrix[:, 0] = (matrix[:, 0] - self.min[:, 0]) / self.delta[:, 0]
        # 只有两个特征，特征矩阵只有两行，且只有前三列有数据
        elif len(matrix) == 2:
            norm_matrix[:, :-1] = (matrix[:, :-1] - self.min[:, :-1]) / self.delta[:, :-1]
        else:
            norm_matrix = (matrix - self.min) / self.delta

        return norm_matrix


