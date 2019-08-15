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



    # 得到基准图特征向量（在找到新的特征匹配方法后，基准图和匹配图的特征向量获取可以用同一套方法）
    def feature_vec_refer(self):
        # 得到所有基准图的特征信息
        features_refer = pd.read_csv(self.csv_path, engine='python')

        # 从配准图的.mat文件中找到特征对应关系，这里主要是找到ID打为-1的将其从基准图特征中删除
        dp=[]
        feature_match_ID = []
        mat_path = glob(self.csv_path[:-10]+'\*warped*.mat')
        for ind in range(len(mat_path)):
            fmID = int(loadmat(mat_path[ind])['featureinfo'][3][0][0])
            feature_match_ID.append(fmID)
            if fmID == -1:
                dp.append(ind)


        # 计算径向长度
        # 先取出所有特征的位置坐标
        x_fv = features_refer['x'].values.reshape(len(features_refer), 1)
        y_fv = features_refer['y'].values.reshape(len(features_refer), 1)
        featuredirect = features_refer['Direction'].values.reshape(len(features_refer), 1)
        x_fv = np.delete(x_fv, dp, 0)
        y_fv = np.delete(y_fv, dp, 0)
        featuredirect = np.delete(featuredirect, dp, 0)

        # 顶点排序（有无视凹多边形的方法，这里采用的是方位角）
        # 得到几何中心坐标
        c_x = centriodloc(x_fv, y_fv)[0]
        c_y = centriodloc(x_fv, y_fv)[1]
        # 得到顶点排序后的特征坐标
        feature_sorted = minutia_sort(featuredirect, x_fv, y_fv, c_x, c_y)

        x_fv_sorted = feature_sorted[:, 1]
        y_fv_sorted = feature_sorted[:, 2]
        # 此处可添加可视化代码用opencv，然后保存文件


        radius_len = np.sqrt(pow(x_fv_sorted-c_x, 2)+pow(y_fv_sorted-c_y, 2))
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
        feature_vec_refer = np.random.uniform(0, 0, (len(features_refer)-len(dp), colum_n))
        feature_vec_refer[:, 0] = feature_sorted[:, 0]+90
        feature_vec_refer[:, 1] = radius_len
        feature_vec_refer[:, 2] = polyline_len[:, 0]
        feature_vec_refer[:, 3] = tri_area[:, 0]

        return(feature_vec_refer)

    # 配准图特征向量
    def feature_vec_match(self):
        features_match = pd.read_csv(self.csv_path, engine='python')
        # print(len(features_match))
        match_path = glob(self.csv_path[:-10] + '\*warped*.mat')

        # 计算径向长度
        # 先取出所有特征的位置坐标
        _x = features_match['x'].values.reshape(len(features_match), 1)
        _y = features_match['y'].values.reshape(len(features_match), 1)
        _featuredirect = features_match['Direction'].values.reshape(len(features_match), 1)
        # 做一个对应关系在基准图和匹配图之间
        # 拿到所有的配准图的特征ID  此处的path为csv路径

        feature_match_ID = []
        for ind in range(len(match_path)):
            # 得到特征ID
            fmID = int(loadmat(match_path[ind])['featureinfo'][3][0][0])
            if fmID == -1:
                continue
            feature_match_ID.append(fmID)


        x_fv=_x[feature_match_ID, : ]
        y_fv=_y[feature_match_ID, : ]
        featuredirect=_featuredirect[feature_match_ID, : ]

        # 顶点排序（有无视凹多边形的方法，这里采用的是方位角）
        # 得到几何中心坐标
        c_x = centriodloc(x_fv, y_fv)[0]
        c_y = centriodloc(x_fv, y_fv)[1]
        # 得到顶点排序后的特征坐标
        feature_sorted = minutia_sort(featuredirect, x_fv, y_fv, c_x, c_y)
        # print(feature_sorted)
        x_fv_sorted = feature_sorted[:,1]
        y_fv_sorted = feature_sorted[:,2]


        radius_len = np.sqrt(pow(x_fv_sorted-c_x, 2)+pow(y_fv_sorted-c_y, 2))
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
        feature_vec_match = np.random.uniform(0, 0, (len(feature_match_ID), colum_n))
        feature_vec_match[:, 0] = feature_sorted[:, 0]+90
        feature_vec_match[:, 1] = radius_len
        feature_vec_match[:, 2] = polyline_len[:, 0]
        feature_vec_match[:, 3] = tri_area[:, 0]

        return(feature_vec_match)


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

    delta = max -min
    if len(matrix) == 1:
        matrix_normalize = (matrix[:, 0]-min[:, 0]) / delta[:, 0]
    elif len(matrix) == 2:
        matrix_normalize = (matrix[:, :-1]-min[:, :-1])/delta[:, :-1]
    else:
        matrix_normalize = (matrix - min) / delta

    return matrix_normalize

# 其实同源鞋印的距离，传入的参数应该是向量而不是csv文件
def feature_distance(csv1, csv2, max, min):

    mat = loadmat(re.sub(r'[0-9a-zA-Z]+.csv', r'transformation.mat', csv1))
    refer_path = re.sub(r'[0-9a-zA-Z]+.csv', '', csv1)+mat['trans'][0][0][0].replace('.jpg', '.csv')
    # 判断哪个是基准图，哪个是配准图
    if csv1 == refer_path:
        match_path = csv2
    else:
        match_path = csv1
    feature_vec_reference = GetFeatureVec(refer_path).feature_vec_refer()
    feature_vec_match =  GetFeatureVec(match_path).feature_vec_match()

    # 这一步若两个特征个数不相等无法计算
    distance = np.sqrt(np.sum(pow(normalize(feature_vec_reference, max, min)-normalize(feature_vec_match, max, min), 2)))

    return distance

# 计算非同源鞋印距离的方法
def feature_distance_common(feature_mat1, feature_mat2, max, min):


    # 计算欧氏距离，是否有其他距离的算法增加，可以对比效果
    distance = np.sqrt(np.sum(pow(normalize(feature_mat1, max, min)-normalize(feature_mat2, max ,min), 2)))

    return distance



if __name__ == '__main__':
    colum_n = 4  # 特征向量包含的参数个数
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
        # 将左右脚枚举的同源数据放在一起
        end_list.extend(combine(csv_list_L, 2))
    for innd in range(len(csv_path_R)):
        csv_list_R = glob(csv_path_R[innd]+'\\R*.csv')
        end_list.extend(combine(csv_list_R, 2))
    for i in range(len(end_list)):
        try:
            # 求出两个鞋印间的特征向量欧式距离
            d = feature_distance(end_list[i][0], end_list[i][1], max_s, min_s)
            distance_isogeny_count.append(d)
            # 去掉仅有一个特征向量的情况，d=1
        except Exception as e:
            logging.exception(e)
            # 此处数据准备过程中，有一文件人工删除了某一特征，所以无法计算
            print(end_list[i]+'错误'+'i = %d' %i)
    while 1.0 in distance_isogeny_count:
        distance_isogeny_count.remove(1.0)
    print(distance_isogeny_count)
    print(len(distance_isogeny_count))

    # # 非同源的数据比较
    # distance_non_count=[]
    # # 这里采用testV1.3的数据做非同源的计算
    # different_source_file_path = glob('F:\足迹\\footprint\different source\*')
    # # 这里计算所有非同源的数据最大最小值
    # csv_list = []
    # for p in different_source_file_path:
    #     csv_path = glob(p + '\\*.csv')
    #     # 生成csv文件路径list
    #     _csv_list = [csv_path[i] for i in range(len(csv_path))]
    #     csv_list += _csv_list
    # result2 = featurevec_overall(csv_list)
    # max_d = result2[0]
    # min_d = result2[1]
    # # 得到所有两两组合的文件夹
    # ds_file_L = combine(glob(different_source_file_path + '\\*\\L*.csv'), 2)
    # ds_file_R = combine(glob(different_source_file_path + '\\*\\R*.csv'), 2)
    # ds_file = ds_file_L + ds_file_R
    #
    # for i in range(len(ds_file)):
    #     # 首先选出两个文件夹
    #     file1, file2 = ds_file[i][0], ds_file[i][1]
    #     feature_vec_s1 = feature_vec_common(file1)
    #     feature_vec_s2 = feature_vec_common(file2)
    #
    #     # 选出两个特征向量包含的最少特征
    #     l_s1 = len(feature_vec_s1)
    #     l_s2 = len(feature_vec_s2)
    #
    #     if l_s1 == l_s2:
    #         d = feature_distance_common(feature_vec_s1, feature_vec_s2, max_d, min_d)
    #
    #     elif l_s1 > l_s2:
    #         l = [l_s1, l_s2]
    #         l.sort()
    #         l_min = l[0]
    #         l_max = l[1]
    #         # 在特征数多的特征向量中随机选取相应个特征
    #         matrix = random_choice(l_max, l_min, choice_times=l_max)
    #         m = enumnate_list(matrix)
    #         fv_modify = np.zeros((l_min, colum_n))
    #         d_combine = []
    #         for l in m:
    #             fv_modify = feature_vec_s1[l, :]
    #             d = feature_distance_common(fv_modify, feature_vec_s2, max_d, min_d)
    #             d_combine.append(d)
    #         d_combine.sort()
    #         d = d_combine[0]
    #
    #     elif l_s1 < l_s2:
    #         l = [l_s1, l_s2]
    #         l.sort()
    #         l_min = l[0]
    #         l_max = l[1]
    #         # 在特征数多的特征向量中随机选取相应个特征，加扭转
    #         matrix = random_choice(l_max, l_min, choice_times=l_max)
    #         m = enumnate_list(matrix)
    #         fv_modify = np.zeros((l_min, colum_n))
    #         d_combine = []
    #         for l in m:
    #             fv_modify = feature_vec_s2[l, :]
    #             d = feature_distance_common(fv_modify, feature_vec_s1, max_d, min_d)
    #             d_combine.append(d)
    #         d_combine.sort()
    #         d=d_combine[0]
    #     distance_non_count.append(d)
    #
    # # 删除只有一个特征向量的对比
    # while 1.0 in distance_non_count:
    #     distance_non_count.remove(1.0)
    # print(distance_non_count)
    # print(len(distance_non_count))





