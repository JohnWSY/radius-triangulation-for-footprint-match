import numpy as np
from utils import *
# 顶点排序(方位角)
# 输入为特征方向，特征位置x坐标，特征位置y坐标，径向剖分中心点x坐标，径向剖分中心点y坐标
# 输出为排序后的特征方向，特征位置x, y坐标
class VertexSort(object):
    def __init__(self, featuredirect, x_fv, y_fv, c_x, c_y):
        '''
        :param featuredirect: 特征方向
        :param x_fv: 特征x
        :param y_fv: 特征y
        :param c_x:中心x
        :param c_y:中心y
        '''
        angle = np.random.uniform(0, 0, (len(x_fv), 1))
        for i in range(len(x_fv)):
            angle[i] = azimuthAngle(c_x, c_y, x_fv[i][0], y_fv[i][0])
        feature_init = np.c_[(featuredirect, x_fv, y_fv, angle)]
        # 根据最后一列方位角进行排序
        feature_sorted = feature_init[np.lexsort(feature_init.T)][:, :3]

        self.featuredirect = feature_sorted[:, 0]
        self.x = feature_sorted[:, 1]
        self.y = feature_sorted[:, 2]
