from glob import glob
from utils import *
from LoadCSV import LoadCSV

# 同源鞋印的特征匹配关系
class GetCommonPart(object):
    def __init__(self, fv, mat_list):
        '''
        :param fv: 传入特征对象
        :param mat_list: 特征匹配关系
        '''
        self.x_fv = fv.x_fv[mat_list]
        self.y_fv = fv.y_fv[mat_list]
        self.featuredirect = fv.featuredirect[mat_list]

# 调用手动匹配同源鞋印ID的函数
def feature_vec_same(csv1, csv2):
    '''
    :param csv1: csv文件路径
    :param csv2: csv文件路径
    :return: 特征匹配后的特征对象
    '''
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

    return feature_select1, feature_select2



