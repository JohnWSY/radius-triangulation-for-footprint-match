import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from utils import *
from scipy import stats



def draw_distribution(same_source_path, different_source_path):
    distance_isogeny_count = read_txt(same_source_path)

    distance_non_count = read_txt(different_source_path)

    # plot画出对比的直方图

    plt.hist(distance_isogeny_count, density=0, bins=len(distance_isogeny_count), facecolor='r',

                                   edgecolor='w', cumulative=False, label='same source')

    plt.hist(distance_non_count, density = 0, bins = 500, facecolor='cyan',

                                   edgecolor='k', alpha = 0.3, cumulative=False, label='different source')

    plt.legend()

    plt.show()



    # seaborn绘图
    # sns.distplot(distance_non_count, kde=False, label='different source', fit=stats.gamma, bins=1000,
    #              hist_kws=dict(edgecolor='cyan'), fit_kws=dict(color='r'))
    #
    # sns.distplot(distance_isogeny_count, kde=False, label='same source', bins=1000,
    #              hist_kws=dict(edgecolor='cyan'), fit_kws=dict(color='r'))
    #
    # plt.show()




if __name__ == '__main__':

    draw_distribution('E:\\footprint\same_source.txt', 'E:\\footprint\different_source.txt')



