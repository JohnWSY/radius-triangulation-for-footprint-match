import numpy as np
import math
from itertools import combinations
import random
from scipy.io import loadmat

# 计算径向剖分中心点坐标
def centriodloc(x, y):
    '''
    :param x: 特征位置x坐标数组
    :param y: 特征位置y坐标数组
    :return: 径向剖分中心x, y坐标
    '''
    c_x = x.mean(axis = 0)
    c_y = y.mean(axis = 0)

    return c_x, c_y


# 计算多边形边长
def polygon_len(x_fv_sorted, y_fv_sorted):
    '''
    :param x_fv_sorted: 排序后的特征位置x坐标
    :param y_fv_sorted: 排序后的特征位置y坐标
    :return: 多边形长度数组
    '''
    polygon_len = np.random.uniform(0, 0, (len(x_fv_sorted), 1))
    feature_loc_sort = np.c_[(x_fv_sorted, y_fv_sorted)]
    feature_loc_sort = np.insert(feature_loc_sort, len(feature_loc_sort), values=feature_loc_sort[0, :], axis=0)
    # print(feature_loc_sort)
    for i in range(len(x_fv_sorted)):
        polygon_len[i] = np.sqrt(pow(feature_loc_sort[i][0]-feature_loc_sort[i+1][0], 2)+pow(feature_loc_sort[i][1]-feature_loc_sort[i+1][1], 2))
    # print(polygon_len)
    return polygon_len

# 海伦公式计算三角形面积
def triangle_area(radius_len, polyline_len):
    '''
    :param radius_len: 径向长度
    :param polyline_len: 多边形边长
    :return: 三角形面积数组
    '''
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

    return tri_area


# 计算特征相随对于中心点的方位角
def azimuthAngle(x1, y1, x2, y2):
    '''
    :param x1: 特征x坐标
    :param y1: 特征y坐标
    :param x2: 中心x坐标
    :param y2: 中心y坐标
    :return: 角度值
    '''
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


# 计算独立边界HoleAB的方向,返回的是角度值
def GetDirection(x, y, x_m, y_m):
    '''
    :param x: 边界点集x坐标
    :param y: 边界点集y坐标
    :param x_m: 边界中心x坐标
    :param y_m: 边界中心y坐标
    :return: 特征方向角度值[-90，90]之间
    '''
    cov_matrix = np.zeros((2, 2))
    cov_matrix[0][0] = pow(x - x_m, 2).mean(axis=0)
    cov_matrix[1][1] = pow(y - y_m, 2).mean(axis=0)
    cov_matrix[0][1] = ((x - x_m) * (y - y_m)).mean(axis=0)
    cov_matrix[1][0] = cov_matrix[0][1]

    a, b = np.linalg.eig(cov_matrix)

    vec = [b[0][0], b[0][1]]
    if vec[0] > 0 and vec[1] > 0:
        theta = math.atan(vec[1] / vec[0])
    elif vec[0] < 0 and vec[1] > 0:
        theta = - math.atan(vec[1] / -vec[0])
    elif vec[0] < 0 and vec[1] < 0:
        theta = math.atan(vec[1] / vec[0])
    else:
        theta = -math.atan(-vec[1] / vec[0])
    return theta*(180/math.pi)


# 两两枚举方法
def combine(temp_list, n):
    '''
    :param temp_list: 数据列表
    :param n: n个元素为一组
    :return: 组合后的列表
    '''
    temp_list2 = []
    for c in combinations(temp_list, n):
        temp_list2.append(list(c))
    return temp_list2

# 这里把两个二维数组列合并, 用于reduce函数
def vstack(a, b):
    return np.vstack((a,b))

# 定义一个函数，输入一个list，输出这个list元素依次左移的一组list
def roll_list(lt):
    '''
    :param lt: 原列表
    :return: 列表元素依次左移的结果集合列表
    '''
    end = []
    for i in range(len(lt)):
        lt.append(lt.pop(0))
        end.append(lt.copy())
    return end


# 定义一个方法，在非同源鞋印特征数量不等时，较多的特征鞋印中多次选出与较少特征对应的样本
def random_choice(l_max, l_min, choice_times):
    '''
    :param l_max: 特征较多的鞋印特征数
    :param l_min: 特征较少的鞋印特征数
    :param choice_times: 选择的次数
    :return: 返回所有抽取结果的列表
    '''
    l_end=[]
    i=0
    while i <= choice_times:
        l = random.sample(range(l_max), l_min)
        # 这里为了降低时间复杂度们可以使用哈希（注意可哈希对象），代码写过的找不到了...就不改了
        # 这里修改后可提高组合数的上限阈值，也算提升了运行速度
        if l in l_end:
            continue
        l_end.append(l)
        i+=1
    return l_end

# 读取.mat文件中的ID
def mat_read(mat_path):
    '''
    :param mat_path: .mat文件路径
    :return: ID
    '''
    m = loadmat(mat_path)

    return int(m['featureinfo'][3][0][0])


# 存数据的方法
def text_save(filename, data):
    '''
    :param filename: 存储的文件路径
    :param data: 存储的数据（list）
    '''
    file = open(filename,'w')
    for i in range(len(data)):
        s = str(data[i]).replace('[','').replace(']','')
        s = s.replace("'",'').replace(',','') +'\n'
        file.write(s)
    file.close()

# 读取txt的函数
def read_txt(filename):
    '''
    :param filename: 读取文件路径
    :return: 读取的list
    '''
    txt = open(filename, "r")
    txt_list = []
    for line in txt.readlines():
        line = line.strip()
        # 去掉每行头尾空白
        txt_list.append(float(line))
    txt.close()

    return txt_list


