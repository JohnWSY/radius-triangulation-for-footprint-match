import numpy as np

class CalcFeatureDistance(object):
    def __init__(self, feature_mat1, feature_mat2):
        '''
        :param feature_mat1:特征矩阵1
        :param feature_mat2: 特征矩阵2
        '''

        # 计算欧氏距离，是否有其他距离的算法增加，可以对比效果
        '''这里目前做了小改动，不考虑方向，因为鞋印特征的方向没有参考价值，考虑用特征类型'''
        self.distance = np.sqrt(np.sum(pow(feature_mat1[:, 1:]-feature_mat2[:, 1:], 2)))


