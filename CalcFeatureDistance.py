import numpy as np

class CalcFeatureDistance(object):
    def __init__(self, feature_mat1, feature_mat2):

        # 计算欧氏距离，是否有其他距离的算法增加，可以对比效果
        self.distance = np.sqrt(np.sum(pow(feature_mat1-feature_mat2, 2)))


