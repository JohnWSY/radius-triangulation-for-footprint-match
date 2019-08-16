from scipy.io import loadmat
from glob import glob
import numpy as np
import random
from enum import Enum
from utils import *
import logging
import matplotlib.pyplot as plt
import re
from functools import reduce
import math
import pandas as pd
from scipy.special import perm, comb
import os
from scipy import stats
import seaborn as sns




# 特征类型，以备后期特征向量中要加入
class Shape(Enum):
    POINT = 0
    LINE = 1
    CIRCLE = 2
    TRIANGLE = 3
    IRREGULAR = 4
    ERROR = None

def feature_vec_common(csv_path):
    features_common = pd.read_csv(csv_path, engine='python')

    # 计算径向长度
    # 先取出所有特征的位置坐标
    x_fv = features_common['x'].values.reshape(len(features_common), 1)
    y_fv = features_common['y'].values.reshape(len(features_common), 1)
    featuredirect = features_common['Direction'].values.reshape(len(features_common), 1)

    # 顶点排序（有无视凹多边形的方法，这里采用的是方位角）
    # 得到几何中心坐标
    c_x = centriodloc(x_fv, y_fv)[0]
    c_y = centriodloc(x_fv, y_fv)[1]
    # 得到顶点排序后的特征坐标
    feature_sorted = minutia_sort(featuredirect, x_fv, y_fv, c_x, c_y)

    x_fv_sorted = feature_sorted[:, 1]
    y_fv_sorted = feature_sorted[:, 2]
    # 此处可添加可视化代码用opencv，然后保存文件

    radius_len = np.sqrt(pow(x_fv_sorted - c_x, 2) + pow(y_fv_sorted - c_y, 2))
    # print(radius_len)

    # 计算多边形边长
    polyline_len = polygon_len(x_fv_sorted, y_fv_sorted)
    # print(polyline_len)

    # 计算三角形面积（海伦公式）
    radius_len1 = np.insert(radius_len, len(radius_len),
                            values=radius_len[0], axis=0)
    tri_area = np.random.uniform(0, 0, (len(polyline_len), 1))
    for i in range(len(polyline_len)):
        a = radius_len1[i]
        b = radius_len1[i + 1]
        c = polyline_len[i][0]
        p = (a + b + c) / 2
        tri_area[i] = np.sqrt(p * (p - a) * (p - b) * (p - c))
        if math.isnan(tri_area[i]):
            tri_area[i]=0
    # print(tri_area)

    # 为特征向量赋值（特征方向, 径向长度， 多边形边长， 三角形面积）
    feature_vec_common = np.random.uniform(0, 0, (len(features_common), colum_n))
    feature_vec_common[:, 0] = feature_sorted[:, 0]+90
    feature_vec_common[:, 1] = radius_len
    feature_vec_common[:, 2] = polyline_len[:, 0]
    feature_vec_common[:, 3] = tri_area[:, 0]

    return feature_vec_common



class GetFeatureVec(object):
    def __init__(self, sample_file):
        # 得到用于解析的csv文件
        self.csv_path = sample_file


    # 得到特征向量
    def feature_vec(self, mat_list):
        # 得到所有基准图的特征信息
        features_refer = pd.read_csv(self.csv_path, engine='python')

        # 计算径向长度
        # 先取出所有特征的位置坐标
        x_fv = features_refer['x'].values.reshape(len(features_refer), 1)
        y_fv = features_refer['y'].values.reshape(len(features_refer), 1)
        featuredirect = features_refer['Direction'].values.reshape(len(features_refer), 1)
        x_fv = x_fv[mat_list, :]
        y_fv = y_fv[mat_list, :]
        featuredirect = featuredirect[mat_list, :]

        feature_vec = self.calc_feature_vec(x_fv, y_fv, featuredirect)

        return feature_vec


    # 计算特征向量多边形的主要方法
    def calc_feature_vec(self, x_fv, y_fv, featuredirect):
        # 顶点排序（有无视凹多边形的方法，这里采用的是方位角）
        # 得到几何中心坐标
        c_x = centriodloc(x_fv, y_fv)[0]
        c_y = centriodloc(x_fv, y_fv)[1]
        # 得到顶点排序后的特征坐标
        feature_sorted = minutia_sort(featuredirect, x_fv, y_fv, c_x, c_y)
        # print(feature_sorted)
        x_fv_sorted = feature_sorted[:, 1]
        y_fv_sorted = feature_sorted[:, 2]

        radius_len = np.sqrt(pow(x_fv_sorted - c_x, 2) + pow(y_fv_sorted - c_y, 2))
        # print(radius_len)

        # 计算多边形边长
        polyline_len = polygon_len(x_fv_sorted, y_fv_sorted)
        # print(polyline_len)

        # 计算三角形面积（海伦公式）
        radius_len1 = np.insert(radius_len, len(radius_len),
                                values=radius_len[0], axis=0)
        tri_area = np.random.uniform(0, 0, (len(polyline_len), 1))
        for i in range(len(polyline_len)):
            a = radius_len1[i]
            b = radius_len1[i + 1]
            c = polyline_len[i][0]
            p = (a + b + c) / 2
            tri_area[i] = np.sqrt(p * (p - a) * (p - b) * (p - c))
            if math.isnan(tri_area[i]):
                tri_area[i] = 0
        # print(tri_area)

        # 为特征向量赋值（特征方向, 径向长度， 多边形边长， 三角形面积）
        feature_vec = np.random.uniform(0, 0, (len(x_fv), colum_n))
        feature_vec[:, 0] = feature_sorted[:, 0] + 90
        feature_vec[:, 1] = radius_len
        feature_vec[:, 2] = polyline_len[:, 0]
        feature_vec[:, 3] = tri_area[:, 0]

        return (feature_vec)

# 这里要定义一个方法，得到全局的特征向量用于归一化，输入为所有csv文件列表，最后输出max, min
def featurevec_overall(csv_list):
    res = list(map(feature_vec_common, csv_list))
    # 只有一个特征点的
    l1=[]
    # 只有两个特征点的
    l2=[]
    # 只有三个特征点的
    l3=[]
    for l in res:
        if len(l) == 1:
            l1.append(l)
        elif len(l) == 2:
            l2.append(l)
        else:
            l3.append(l)
    result = reduce(vstack, res)
    max = result.max(axis=0)
    result_alt2 = reduce(vstack, l2+l3)[:, 1:3]
    result_alt3 = reduce(vstack, l3)[:, -1]
    min = np.zeros((1, 4))
    min[0][0] = result[:, 0].min(axis=0)
    min[0][1:3] = result_alt2.min(axis = 0)
    min[0][-1] = result_alt3.min(axis = 0)

    return max,min


# 归一化
def normalize(matrix, max, min):

    delta = max - min
    matrix_normalize = np.zeros((len(matrix), colum_n))
    if len(matrix) == 1:
        matrix_normalize[:, 0] = (matrix[:, 0]-min[:, 0]) / delta[:, 0]
    elif len(matrix) == 2:
        matrix_normalize[:, :-1] = (matrix[:, :-1]-min[:, :-1])/delta[:, :-1]
    else:
        matrix_normalize = (matrix - min) / delta

    return matrix_normalize

# 再定义一个函数，返回的是同源的两个向量，不能拆分。因为有对应关系
def feature_vec_same(csv1, csv2):
    # 得到所有的mat文件路径
    mat_path1 = glob(csv1.replace('.csv', '*.mat'))
    mat_path2 = glob(csv2.replace('.csv', '*.mat'))
    # 读取mat文件中的ID号
    mat_list1 = list(map(mat_read, mat_path1))
    mat_list2 = list(map(mat_read, mat_path2))
    # 筛选出ID号为-1的位置
    l_del = []
    for i, element in enumerate(mat_list1):
        if element == -1:
            l_del.append(i)
    for j, element in enumerate(mat_list2):
        if element == -1:
            if j not in l_del:
                l_del.append(j)

    # 在得到两者共有的-1位置后，将两个列表中的该位置的元素都删除掉
    mat_list1 = [mat_list1[i] for i in range(len(mat_list1)) if (i not in l_del)]
    mat_list2 = [mat_list2[i] for i in range(len(mat_list2)) if (i not in l_del)]

    # 根据列表读取相应的特征
    feature_vec1 = GetFeatureVec(csv1).feature_vec(mat_list1)
    feature_vec2 = GetFeatureVec(csv2).feature_vec(mat_list2)

    return feature_vec1, feature_vec2


# 计算鞋印距离的方法
def feature_distance(feature_mat1, feature_mat2, max, min):

    # 计算欧氏距离，是否有其他距离的算法增加，可以对比效果
    distance = np.sqrt(np.sum(pow(feature_mat1-feature_mat2, 2)))

    return distance


if __name__ == '__main__':
    colum_n = 4  # 特征向量包含的参数个数
    threshold = 1000  # 组合计算阈值
    # 这里同源的数据多了一级目录
    same_source_path=glob('F:\足迹\\footprint\same source\*\*')
    csv_list = []
    for p in same_source_path:
        csv_path = glob(p+'\\*.csv')
        # 生成csv文件路径list
        _csv_list = [csv_path[i] for i in range(len(csv_path))]
        csv_list += _csv_list
    # 这里将全局特征向量的最大最小值保存下来
    result1 = featurevec_overall(csv_list)
    max_s = result1[0]
    min_s = result1[1]
    #
    # 同源的数据比较
    # 先从same_source中挑出所有L和R的分开
    csv_path_L = []
    csv_path_R = []
    for each in same_source_path:
        if 'L' in each:
            csv_path_L.append(each)
        else:
            csv_path_R.append(each)
    distance_isogeny_count=[]
    end_list=[]
    for ind in range(len(csv_path_L)):
        # 生成csv文件路径list
        csv_list_L = glob(csv_path_L[ind]+'\\L*.csv')
        end_list.extend(combine(csv_list_L, 2))
    for innd in range(len(csv_path_R)):
        csv_list_R = glob(csv_path_R[innd]+'\\R*.csv')
        end_list.extend(combine(csv_list_R, 2))
    for i in range(len(end_list)):
        # print(end_list[i])
        feature_vec1, feature_vec2 = feature_vec_same(end_list[i][0], end_list[i][1])
        featurevec1 = normalize(feature_vec1, max_s, min_s)
        featurevec2 = normalize(feature_vec2, max_s, min_s)
        try:
            # 求长度没意义，同源的一定等长
            # 应该直接旋转
            l_s = len(featurevec1)
            matrix = list(range(l_s))
            m = roll_list(matrix)
            fv_modify = np.zeros((l_s, colum_n))
            d_combine = []
            for l in m:
                fv_modify = featurevec2[l, :]
                d = feature_distance(featurevec1, fv_modify, max_s, min_s)
                d_combine.append(d)
            d_combine.sort()
            d = d_combine[0]
            distance_isogeny_count.append(d)
            # 去掉仅有一个特征向量的情况，d=1
        except Exception as e:
            logging.exception(e)
            # 此处数据准备过程中，有一文件人工删除了某一特征，所以无法计算
            print(end_list[i], i)

    print(distance_isogeny_count)
    print(len(distance_isogeny_count))

    # 这里加一个保存数据到txt文件
    text_save('F:\足迹\\footprint\same_source.txt', distance_isogeny_count)

    # 非同源的数据比较
    distance_non_count=[]
    different_source_file_path = glob('F:\足迹\\footprint\different source\*')
    # 这里计算所有非同源的数据最大最小值
    csv_list_L = []
    csv_list_R = []
    # 存储每个文件夹中同源的鞋印组合
    del_list_L = []
    del_list_R = []
    for p in different_source_file_path:
        csv_path_L = glob(p + '\\L*.csv')
        # 生成csv-L文件路径list
        _csv_list_L = [csv_path_L[i] for i in range(len(csv_path_L))]
        del_csv_list_L = combine(_csv_list_L, 2)
        del_list_L += del_csv_list_L
        csv_list_L += _csv_list_L
        # 生成csv-R文件路径list
        csv_path_R = glob(p + '\\R*.csv')
        _csv_list_R = [csv_path_R[i] for i in range(len(csv_path_R))]
        del_csv_list_R = combine(_csv_list_R, 2)
        del_list_R += del_csv_list_R
        csv_list_R += _csv_list_R
    csv_list = csv_list_L + csv_list_R
    del_list = del_list_L + del_list_R
    result2 = featurevec_overall(csv_list)
    max_d = result2[0]
    min_d = result2[1]
    # 得到左右脚中两两组合
    ds_file_L = combine(csv_list_L, 2)
    ds_file_R = combine(csv_list_R, 2)
    ds_file = ds_file_L + ds_file_R
    [ds_file.remove(x) for x in del_list]
    # 从以上结果中要去除同源的左右脚


    for i in range(len(ds_file)):
        # 首先选出两个文件
        file1, file2 = ds_file[i][0], ds_file[i][1]
        feature_vec_s1 = normalize(feature_vec_common(file1), max_d, min_d)
        feature_vec_s2 = normalize(feature_vec_common(file2), max_d, min_d)

        # 选出两个特征向量包含的最少特征
        l_d1 = len(feature_vec_s1)
        l_d2 = len(feature_vec_s2)

        if l_d1 == l_d2:
            d = feature_distance(feature_vec_s1, feature_vec_s2, max_d, min_d)

        else:
            l = [l_d1, l_d2]
            l.sort()
            l_min = l[0]
            l_max = l[1]
            fv_modify = np.zeros((l_min, colum_n))
            d_combine = []
            # 如果组合数超过一万，那么就随机取一万次
            if comb(l_max, l_min) > threshold:
                matrix = random_choice(l_max, l_min, threshold)
                m = enumerate_list(matrix)
            # 如果组合少于一万，就枚举
            else:
                matrix = combine(range(l_max), l_min)
                m = enumerate_list(matrix)

            if l_d1 > l_d2:
                for l in m:
                    fv_modify = feature_vec_s1[l, :]
                    d_d = feature_distance(fv_modify, feature_vec_s2, max_d, min_d)
                    d_combine.append(d_d)
                d_combine.sort()
                d = d_combine[0]

            elif l_d1 < l_d2:
                for l in m:
                    fv_modify = feature_vec_s2[l, :]
                    d_d = feature_distance(fv_modify, feature_vec_s1, max_d, min_d)
                    d_combine.append(d_d)
                d_combine.sort()
                d=d_combine[0]
        distance_non_count.append(d)

    print(distance_non_count)
    print(len(distance_non_count))

    # 这里加一个保存数据到txt文件
    text_save('F:\足迹\\footprint\different_source.txt', distance_non_count)




