from glob import glob
import numpy as np
import random
from utils import *
from scipy.special import comb
from CalcFeatureVec import *
from Normalization import *
from File_processing import *
from CalcFeatureDistance import *
import re
from DrawFeature import *







if __name__ == '__main__':
    threshold = 1000  # 组合计算阈值
    path = r'../footprint/same source'


    # 同源
    path_all = glob(path + '\*\*')
    # 所有csv路径
    csv_list_all = []
    # 同源的csv路径
    csv_list_same = []
    distance_isogeny_count = []
    for p in path_all:
        csv_path = glob(p+'\\*.csv')
        # 同源csv路径
        csv_list_same += combine(csv_path, 2)
        # 所有csv路径
        csv_list_all += csv_path

    # 全局csv文件的解析结果
    features_isogeny = list(map(LoadCSV, csv_list_all))
    result = FeatureOverall(features_isogeny)
    max = result.max
    min = result.min

    # 遍历所有的组合，计算特征矩阵欧式距离
    for i in range(len(csv_list_same)):
        feature_select1, feature_select2 = feature_vec_same(csv_list_same[i][0], csv_list_same[i][1])
        # 可视化径向剖分结果
        # img = DrawFeature(feature_select1, feature_select2)
        #         # img_path1 = csv_list_same[i][0].replace('.csv', '.jpg')
        #         # img_path2 = csv_list_same[i][1].replace('.csv', '.jpg')
        #         # img.draw(img_path1, img_path2)

        feature_vec1 = FeatureVec(feature_select1.x_fv, feature_select1.y_fv, feature_select1.featuredirect).feature_vec_common
        feature_vec2 = FeatureVec(feature_select2.x_fv, feature_select2.y_fv, feature_select2.featuredirect).feature_vec_common
        l_s1 = len(feature_vec1)
        l_s2 = len(feature_vec2)
        # 剔除掉特征数少于3个的，因为特征数小于3则无法进行径向剖分
        if l_s1<3 or l_s2 <3:
            continue

        # 特征矩阵归一化
        featurevec1 = Normalization(max, min).normalize(feature_vec1)
        featurevec2 = Normalization(max, min).normalize(feature_vec2)

        # 将同源鞋印的特征索引列表，每个元素左移
        l_s = len(featurevec1)
        matrix = list(range(l_s))
        # 移动后的结果列表组合
        m = roll_list(matrix)
        d_combine = []
        for l in m:
            fv_modify = featurevec2[l, :]
            d = CalcFeatureDistance(featurevec1, fv_modify).distance
            d_combine.append(d)
        # 特征扭转的结果排序，取最小值
        d_combine.sort()
        d = d_combine[0]
        # 同源鞋印匹配分数集
        distance_isogeny_count.append(d)

    print(distance_isogeny_count)
    # 有多少个结果
    print(len(distance_isogeny_count))

    # 保存数据到txt文件
    # s_s=re.match('.*/', path)
    # text_save(s_s.group()+'same source.txt', distance_isogeny_count)


    # 非同源
    distance_non_count=[]
    # 非同源鞋印区分左右脚
    csv_different_L = []
    csv_different_R = []
    for n in csv_list_all:
        if 'L' in n:
            csv_different_L.append(n)
        else:
            csv_different_R.append(n)
    csv_list_different = combine(csv_different_L, 2)+combine(csv_different_R, 2)
    # 从以上结果中要去除同源的左右脚
    [csv_list_different.remove(x) for x in csv_list_same]
    # text_save('F:\\footprint\different source file.txt', csv_list_different)
    # 选1000组进行测试
    # ds_test = random.sample(csv_list_different, 1000)
    #
    # for i in range(len(ds_test)):
    #     # 首先选出两个文件
    #     file1, file2 = ds_test[i][0], ds_test[i][1]
    for i in range(len(csv_list_different)):
        # 首先选出两个文件
        file1, file2 = csv_list_different[i][0], csv_list_different[i][1]
        lc1 = LoadCSV(file1)
        lc2 = LoadCSV(file2)

        l_d1 = len(lc1.x_fv)
        l_d2 = len(lc2.x_fv)

        if l_d1<3 or l_d2 <3:
            continue

        if l_d1 == l_d2:
            feature_vec_s1 = Normalization(max, min).normalize(FeatureVec(lc1.x_fv, lc1.y_fv, lc1.featuredirect).feature_vec_common)
            feature_vec_s2 = Normalization(max, min).normalize(FeatureVec(lc2.x_fv, lc2.y_fv, lc2.featuredirect).feature_vec_common)
            d = CalcFeatureDistance(feature_vec_s1, feature_vec_s2).distance

        else:
            l = [l_d1, l_d2]
            l.sort()
            l_min = l[0]
            l_max = l[1]
            d_combine = []
            # 如果组合数超过阈值，那么就不重复的随机取，取的次数暂定为阈值
            if comb(l_max, l_min) > threshold:
                matrix = random_choice(l_max, l_min, threshold)
            # 如果组合数少于阈值，就枚举
            else:
                matrix = combine(range(l_max), l_min)

            if l_d1 > l_d2:
                for l in matrix:
                    # 根据列表读取相应的特征
                    feature_select1 = GetCommonPart(lc1, l)
                    feature_vec1 = Normalization(max, min).normalize(FeatureVec(feature_select1.x_fv, feature_select1.y_fv, feature_select1.featuredirect).feature_vec_common)
                    feature_vec2 = Normalization(max, min).normalize(FeatureVec(lc2.x_fv, lc2.y_fv, lc2.featuredirect).feature_vec_common)

                    # 加入旋转
                    m = roll_list(list(range(len(l))))
                    for r in m:
                        fv1 = feature_vec1[r, :]
                        d_d = CalcFeatureDistance(fv1, feature_vec2).distance
                        # 调试找出非同源鞋印特征矩阵距离小于0.1的，可视化径向剖分结过，分析原因
                        # if d_d < 0.1:
                        #     print(i)
                        #     print(csv_list_different[i])
                        #     img = DrawFeature(feature_select1, lc2)
                        #     img_path1 = csv_list_different[i][0].replace('.csv', '.jpg')
                        #     img_path2 = csv_list_different[i][1].replace('.csv', '.jpg')
                        #     img.draw(img_path1, img_path2)
                        d_combine.append(d_d)
                d_combine.sort()
                d = d_combine[0]

            elif l_d1 < l_d2:
                for l in matrix:
                    feature_vec1 = Normalization(max, min).normalize(FeatureVec(lc1.x_fv, lc1.y_fv, lc1.featuredirect).feature_vec_common)
                    feature_select2 = GetCommonPart(lc2, l)
                    feature_vec2 = Normalization(max, min).normalize(FeatureVec(feature_select2.x_fv, feature_select2.y_fv, feature_select2.featuredirect).feature_vec_common)

                    # 加入旋转
                    m = roll_list(list(range(len(l))))
                    for r in m:
                        fv1 = feature_vec1[r, :]
                        d_d = CalcFeatureDistance(fv1, feature_vec2).distance
                        # 调试找出非同源鞋印特征矩阵距离小于0.1的，可视化径向剖分结过，分析原因
                        # if d_d < 0.1:
                        #     print(i)
                        #     print(csv_list_different[i])
                        #     img = DrawFeature(lc1, feature_select2)
                        #     img_path1 = csv_list_different[i][0].replace('.csv', '.jpg')
                        #     img_path2 = csv_list_different[i][1].replace('.csv', '.jpg')
                        #     img.draw(img_path1, img_path2)
                        d_combine.append(d_d)
                d_combine.sort()
                d=d_combine[0]
        distance_non_count.append(d)
        # print(i)
    print(distance_non_count)
    print(len(distance_non_count))

    # 保存数据到txt文件
    # s_d = re.match('.*/', path)
    # text_save(s_d.group() + 'different source.txt', distance_non_count)



