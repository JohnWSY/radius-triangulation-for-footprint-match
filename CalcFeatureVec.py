from utils import *
from LoadCSV import LoadCSV

class FeatureVec(object):

    def __init__(self, x_fv, y_fv, featuredirect):
        # 得到几何中心坐标
        c_x = centriodloc(x_fv, y_fv)[0]
        c_y = centriodloc(x_fv, y_fv)[1]
        # 得到顶点排序后的特征坐标
        feature_sorted = minutia_sort(featuredirect, x_fv, y_fv, c_x, c_y)

        x_fv_sorted = feature_sorted[:, 1]
        y_fv_sorted = feature_sorted[:, 2]
        # 此处可添加可视化代码用opencv，然后保存文件

        # 计算径向长度
        radius_len = np.sqrt(pow(x_fv_sorted - c_x, 2) + pow(y_fv_sorted - c_y, 2))

        # 计算多边形边长
        polyline_len = polygon_len(x_fv_sorted, y_fv_sorted)

        # 计算三角形面积（海伦公式）
        tri_area = triangle_area(radius_len, polyline_len)

        # 为特征向量赋值（特征方向, 径向长度， 多边形边长， 三角形面积）
        self.feature_vec_common = np.random.uniform(0, 0, (len(x_fv), 4))
        self.feature_vec_common[:, 0] = feature_sorted[:, 0]+90
        self.feature_vec_common[:, 1] = radius_len
        self.feature_vec_common[:, 2] = polyline_len[:, 0]
        self.feature_vec_common[:, 3] = tri_area[:, 0]






