from scipy.io import loadmat
from glob import glob
import numpy as np
import random
from utils import *
import logging
import math
import pandas as pd
from scipy.special import comb
from CalcFeatureVec import *
from Normalization import *
from File_processing import *
from CalcFeatureDistance import *
import re
from DrawFeature import *







if __name__ == '__main__':
    threshold = 1000  # 组合计算阈值
    s_p = r'F:\\footprint\same source'
    d_p = r'F:\\footprint\different source'
    # 这里同源的数据多了一级目录
    same_source_path=glob(s_p + '\*\*')
    csv_list = []
    for p in same_source_path:
        csv_path = glob(p+'\\*.csv')
        # 生成csv文件路径list
        _csv_list = [csv_path[i] for i in range(len(csv_path))]
        csv_list += _csv_list
    # 全局csv文件的解析结果
    features_isogeny = list(map(LoadCSV, csv_list))
    result1 = FeatureOverall(features_isogeny)
    max_s = result1.max
    min_s = result1.min

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
        feature_select1, feature_select2 = feature_vec_same(end_list[i][0], end_list[i][1])

        img = DrawFeature(feature_select1, feature_select2)
        img_path1 = end_list[i][0].replace('.csv', '.jpg')
        img_path2 = end_list[i][1].replace('.csv', '.jpg')
        img.draw(img_path1, img_path2)

        feature_vec1 = FeatureVec(feature_select1.x_fv, feature_select1.y_fv, feature_select1.featuredirect).feature_vec_common
        feature_vec2 = FeatureVec(feature_select2.x_fv, feature_select2.y_fv, feature_select2.featuredirect).feature_vec_common
        l_s1 = len(feature_vec1)
        l_s2 = len(feature_vec2)
        if l_s1<3 or l_s2 <3:
            continue
        featurevec1 = Normalization(feature_vec1, max_s, min_s).normalize
        featurevec2 = Normalization(feature_vec2, max_s, min_s).normalize

        # 应该直接旋转
        l_s = len(featurevec1)
        matrix = list(range(l_s))
        m = roll_list(matrix)
        d_combine = []
        for l in m:
            fv_modify = featurevec2[l, :]
            d = CalcFeatureDistance(featurevec1, fv_modify).distance
            d_combine.append(d)
        d_combine.sort()
        d = d_combine[0]
        distance_isogeny_count.append(d)


    # distance_isogeny_count = normalize(distance_isogeny_count)

    print(distance_isogeny_count)
    print(len(distance_isogeny_count))

    # 这里加一个保存数据到txt文件
    s_s=re.match('.*\\\\', s_p)
    text_save(s_s.group()+'same source.txt', distance_isogeny_count)




    # 非同源的数据比较
    distance_non_count=[]
    different_source_file_path = glob(d_p + '\*')
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
    features_non = list(map(LoadCSV, csv_list))
    result2 = FeatureOverall(features_non)
    max_d = result2.max
    min_d = result2.min
    # 得到左右脚中两两组合
    ds_file_L = combine(csv_list_L, 2)
    ds_file_R = combine(csv_list_R, 2)
    ds_file = ds_file_L + ds_file_R
    # 从以上结果中要去除同源的左右脚
    [ds_file.remove(x) for x in del_list]
    # 选1000组进行测试
    ds_test = random.sample(ds_file, 1000)


    for i in range(len(ds_test)):
        # 首先选出两个文件
        file1, file2 = ds_test[i][0], ds_test[i][1]
        lc1 = LoadCSV(file1)
        lc2 = LoadCSV(file2)

        l_d1 = len(lc1.x_fv)
        l_d2 = len(lc2.x_fv)

        if l_d1<3 or l_d2 <3:
            continue

        if l_d1 == l_d2:
            feature_vec_s1 = Normalization(FeatureVec(lc1.x_fv, lc1.y_fv, lc1.featuredirect).feature_vec_common, max_d, min_d).normalize
            feature_vec_s2 = Normalization(FeatureVec(lc2.x_fv, lc2.y_fv, lc2.featuredirect).feature_vec_common, max_d, min_d).normalize
            d = CalcFeatureDistance(feature_vec_s1, feature_vec_s2).distance



        else:
            l = [l_d1, l_d2]
            l.sort()
            l_min = l[0]
            l_max = l[1]
            d_combine = []
            # 如果组合数多，那么就不重复的随机取
            if comb(l_max, l_min) > threshold:
                matrix = random_choice(l_max, l_min, threshold)
            # 如果组合少，就枚举
            else:
                matrix = combine(range(l_max), l_min)

            if l_d1 > l_d2:
                for l in matrix:
                    # 根据列表读取相应的特征
                    feature_select1 = GetCommonPart(lc1, l)
                    feature_vec1 = Normalization(FeatureVec(feature_select1.x_fv, feature_select1.y_fv, feature_select1.featuredirect).feature_vec_common, max_d, min_d).normalize
                    feature_vec2 = Normalization(FeatureVec(lc2.x_fv, lc2.y_fv, lc2.featuredirect).feature_vec_common, max_d, min_d).normalize
                    # 加入旋转
                    m = roll_list(list(range(len(l))))
                    for r in m:
                        fv1 = feature_vec1[r, :]
                        d_d = CalcFeatureDistance(fv1, feature_vec2).distance
                        d_combine.append(d_d)
                d_combine.sort()
                d = d_combine[0]

            elif l_d1 < l_d2:
                for l in matrix:
                    feature_vec1 = Normalization(FeatureVec(lc1.x_fv, lc1.y_fv, lc1.featuredirect).feature_vec_common, max_d, min_d).normalize
                    feature_select2 = GetCommonPart(lc2, l)
                    feature_vec2 = Normalization(FeatureVec(feature_select2.x_fv, feature_select2.y_fv, feature_select2.featuredirect).feature_vec_common, max_d, min_d).normalize
                    # 加入旋转
                    m = roll_list(list(range(len(l))))
                    for r in m:
                        fv1 = feature_vec1[r, :]
                        d_d = CalcFeatureDistance(fv1, feature_vec2).distance
                        d_combine.append(d_d)
                d_combine.sort()
                d=d_combine[0]
        distance_non_count.append(d)
    # distance_non_count = normalize(distance_non_count)
    print(distance_non_count)
    print(len(distance_non_count))

    # 这里加一个保存数据到txt文件
    s_d = re.match('.*\\\\', d_p)
    text_save(s_d.group() + 'different source.txt', distance_isogeny_count)



