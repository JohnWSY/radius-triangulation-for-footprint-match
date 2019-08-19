from glob import glob
from utils import *
from CalcFeatureVec import FeatureVec
from LoadCSV import LoadCSV


class GetCommonPart(object):
    def __init__(self, fv, mat_list):
        self.x_fv = fv.x_fv[mat_list]
        self.y_fv = fv.y_fv[mat_list]
        self.featuredirect = fv.featuredirect[mat_list]

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


    fv1 = LoadCSV(csv1)
    fv2 = LoadCSV(csv2)
    # 根据列表读取相应的特征
    feature_select1 = GetCommonPart(fv1, mat_list1)
    feature_select2 = GetCommonPart(fv2, mat_list2)


    feature_vec1 = FeatureVec(feature_select1.x_fv, feature_select1.y_fv, feature_select1.featuredirect).feature_vec_common
    feature_vec2 = FeatureVec(feature_select2.x_fv, feature_select2.y_fv, feature_select2.featuredirect).feature_vec_common

    return feature_vec1, feature_vec2