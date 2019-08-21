import numpy as np
from utils import *
# 顶点排序（有无视凹多边形的方法，这里采用的是方位角）
class VertexSort(object):
    def __init__(self, featuredirect, x_fv, y_fv, c_x, c_y):

        angle = np.random.uniform(0, 0, (len(x_fv), 1))
        for i in range(len(x_fv)):
            angle[i] = azimuthAngle(c_x, c_y, x_fv[i][0], y_fv[i][0])
        feature_init = np.c_[(featuredirect, x_fv, y_fv, angle)]
        # 根据最后一列方位角进行排序
        feature_sorted = feature_init[np.lexsort(feature_init.T)][:, :3]

        self.featuredirect = feature_sorted[:, 0]
        self.x = feature_sorted[:, 1]
        self.y = feature_sorted[:, 2]
