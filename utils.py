# 根据原代码的main函数提取出有用的数据
import numpy as np
import math
from itertools import combinations

# 得到特征的类别
def featuretype(features):
    feature_type = np.random.uniform(0, 0, (len(features), 1))
    for i in range(len(features)):
        feature_type[i] = features[i].Shape.GeometricClass.value
    # print(featuretype)
    return feature_type

# 得到特征点坐标 (x, y)
def featurelocation(features):
    feature_loc = np.random.uniform(0, 0, (len(features), 2))
    for i in range(len(features)):
        feature_loc[i] = features[i].Location
    # print(feature_loc)
    return feature_loc

# 得到特征点的方向，角度值
def featuredirection(features):
    feature_direct = np.random.uniform(0, 0, (len(features), 1))
    for i in range(len(features)):
        feature_direct[i] = features[i].Direction
    return feature_direct


def centriodloc(x, y):
    c_x = x.mean(axis = 0)
    c_y = y.mean(axis = 0)

    return c_x, c_y

# 计算特征相随对于中心点的方位角
def azimuthAngle(x1, y1, x2, y2):
    angle = 0.0
    dx = x2 - x1
    dy = y2 - y1
    if x2 == x1:
        angle = math.pi / 2.0
        if y2 == y1:
            angle = 0.0
        elif y2 < y1:
            angle = 3.0 * math.pi / 2.0
    elif x2 > x1 and y2 < y1:
        angle = math.atan(-dy / dx)
    elif x2 > x1 and y2 > y1:
        angle = 2*math.pi - math.atan(dy / dx)
    elif x2 < x1 and y2 > y1:
        angle = math.pi + math.atan(-dy / dx)
    elif x2 < x1 and y2 < y1:
        angle = math.pi - math.atan(-dy / -dx)

    result=angle * 180 / math.pi

    return result


def minutia_sort(featuretype, x_fv, y_fv, c_x, c_y):
    angle = np.random.uniform(0, 0, (len(x_fv), 1))
    for i in range(len(x_fv)):
        angle[i] = azimuthAngle(c_x, c_y, x_fv[i][0], y_fv[i][0])
    feature_init = np.c_[(featuretype, x_fv, y_fv, angle)]
    # 根据最后一列方位角进行排序
    feature_sorted = feature_init[np.lexsort(feature_init.T)][:, :3]
    # print(feature_final)
    return feature_sorted


# 计算多边形边长
def polygon_len(x_fv_sorted, y_fv_sorted):
    polygon_len = np.random.uniform(0, 0, (len(x_fv_sorted), 1))
    feature_loc_sort = np.c_[(x_fv_sorted, y_fv_sorted)]
    feature_loc_sort = np.insert(feature_loc_sort, len(feature_loc_sort), values=feature_loc_sort[0, :], axis=0)
    # print(feature_loc_sort)
    for i in range(len(x_fv_sorted)):
        polygon_len[i] = np.sqrt(pow(feature_loc_sort[i][0]-feature_loc_sort[i+1][0], 2)+pow(feature_loc_sort[i][1]-feature_loc_sort[i+1][1], 2))
    # print(polygon_len)
    return polygon_len

# 计算HoleAB的方向,返回的是角度值
def GetDirection(x, y, x_m, y_m):
    cov_matrix = np.zeros((2, 2))
    cov_matrix[0][0] = pow(x - x_m, 2).mean(axis=0)
    cov_matrix[1][1] = pow(y - y_m, 2).mean(axis=0)
    cov_matrix[0][1] = ((x - x_m) * (y - y_m)).mean(axis=0)
    cov_matrix[1][0] = cov_matrix[0][1]

    a, b = np.linalg.eig(cov_matrix)

    vec = [b[0][0], b[0][1]]
    if vec[0] > 0 and vec[1] > 0:
        theta = math.atan(vec[1] / vec[0])
        # print(theta)
    elif vec[0] < 0 and vec[1] > 0:
        theta = - math.atan(vec[1] / -vec[0])
    elif vec[0] < 0 and vec[1] < 0:
        theta = math.atan(vec[1] / vec[0])
    else:
        theta = -math.atan(-vec[1] / vec[0])
    return theta*(180/math.pi)


# 两两枚举方法
def combine(temp_list, n):
    '''根据n获得列表中的所有可能组合（n个元素为一组）'''
    temp_list2 = []
    for c in combinations(temp_list, n):
        temp_list2.append(c)
    return temp_list2

# 这里把两个二维数组列合并, 用于reduce函数
def vstack(a, b):
    return np.vstack((a,b))

# 定义一个函数，输入一个list，输出这个list依次左移的一组list
def roll_list(lt):
    end = []
    for i in range(len(lt)):
        lt.append(lt.pop(0))
        end.append(lt.copy())
    return end


def enumnate_list(end):
    result = []
    for t in end:
        l=list(t)
        result += roll_list(l)

    return result

