from utils import *
from LoadCSV import LoadCSV
from VertexSort import *

class FeatureVec(object):

    def __init__(self, x_fv, y_fv, featuredirect):
        '''
        :param x_fv: 特征位置x
        :param y_fv: 特征位置y
        :param featuredirect:特征方向
        '''
        '''算法的核心，根据输入的特征位置与方向计算每个鞋印的特征矩阵'''
        # 得到几何中心坐标
        c_x = centriodloc(x_fv, y_fv)[0]
        c_y = centriodloc(x_fv, y_fv)[1]
        # 得到顶点排序后的特征坐标
        feature_sorted = VertexSort(featuredirect, x_fv, y_fv, c_x, c_y)

        # 计算径向长度
        radius_len = np.sqrt(pow(feature_sorted.x - c_x, 2) + pow(feature_sorted.y - c_y, 2))

        # 计算多边形边长
        polyline_len = polygon_len(feature_sorted.x, feature_sorted.y)

        # 计算三角形面积（海伦公式）
        tri_area = triangle_area(radius_len, polyline_len)

        # 为特征向量赋值（特征方向, 径向长度， 多边形边长， 三角形面积）
        self.feature_vec_common = np.random.uniform(0, 0, (len(x_fv), 4))
        self.feature_vec_common[:, 0] = feature_sorted.featuredirect+90
        self.feature_vec_common[:, 1] = radius_len
        self.feature_vec_common[:, 2] = polyline_len[:, 0]
        self.feature_vec_common[:, 3] = tri_area[:, 0]






