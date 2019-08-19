import numpy as np
import math
from itertools import combinations
import random
from scipy.io import loadmat

def centriodloc(x, y):
    c_x = x.mean(axis = 0)
    c_y = y.mean(axis = 0)

    return c_x, c_y

# 顶点排序（有无视凹多边形的方法，这里采用的是方位角）
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


def triangle_area(radius_len, polyline_len):
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


def enumerate_list(end):
    result = []
    for t in end:
        l=list(t)
        result += roll_list(l)

    return result

# 定义一个方法，在非同源鞋印特征数量不等时，较大的特征数中多次选出与少量特征对应的数量
def random_choice(l_max, l_min, choice_times):
    l_end=[]
    for i in range(choice_times):
        l = random.sample(range(l_max), l_min)
        l_end.append(l)
    return l_end

def mat_read(mat_path):
    m = loadmat(mat_path)

    return int(m['featureinfo'][3][0][0])


# 读存数据的方法
def text_save(filename, data):
    file = open(filename,'w')
    for i in range(len(data)):
        s = str(data[i]).replace('[','').replace(']','') #去除[],这两行按数据不同，可以选择
        s = s.replace("'",'').replace(',','') +'\n'   #去除单引号，逗号，每行末尾追加换行符
        file.write(s)
    file.close()

# 加一个读取txt的函数
def read_txt(filename):
    txt = open(filename, "r")
    txt_list = []
    for line in txt.readlines():
        line = line.strip()
        # 去掉每行头尾空白
        txt_list.append(float(line))
    txt.close()

    return txt_list

def normalize(list):
    list.sort()
    min = list[0]
    max = list[-1]
    end = [(x-min)/(max-min) for x in list]
    return end
