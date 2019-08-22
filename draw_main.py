import matplotlib.pyplot as plt
import seaborn as sns
from utils import *
from scipy import stats



def draw_plot(distance_isogeny_count, distance_non_count):

    # plot画出对比的直方图

    plt.hist(distance_isogeny_count, density=1, bins=500, facecolor='r', cumulative=False, label='same source')

    plt.hist(distance_non_count, density=1, bins=500, facecolor='cyan', alpha = 0.8, cumulative=False, label='different source')

    plt.legend()
    # 保存图片格式
    plt.savefig('./result_plot.jpg', bbox_inches='tight', dpi=500)

    plt.show()



def draw_seaborn(distance_isogeny_count, distance_non_count):

    # seaborn绘图
    sns.set(style='white', palette="muted", color_codes=True)

    sns.distplot(distance_isogeny_count, kde=False, bins=500, color='r', label=['same source'])

    sns.distplot(distance_non_count, kde=False, bins=500, color='cyan', label=['different source'])

    plt.show()
    # 带宽按照bw=n^(-1/5)计算
    t1=len(distance_isogeny_count)**(-0.2)
    t2=len(distance_non_count)**(-0.2)

    sns.kdeplot(distance_isogeny_count, label=['same source'], bw=t1, shade=True, color='r')

    sns.kdeplot(distance_non_count, label=['different source'], bw=t2, shade=True, color='cyan')

    # plt.savefig('../result_seaborn_kde.jpg', bbox_inches='tight', dpi=500)

    plt.show()

    sns.kdeplot(distance_isogeny_count, label=['same source'], bw=t1, color='r', cumulative=True)

    sns.kdeplot(distance_non_count, label=['different source'], bw=t2, color='cyan', cumulative=True)

    # plt.savefig('../result_seaborn_cumulative.jpg', bbox_inches='tight', dpi=500)

    plt.show()


if __name__ == '__main__':
    same_source_path = r'../same source(without direct).txt'

    different_source_path = r'../different source(without direct).txt'

    distance_isogeny_count = read_txt(same_source_path)

    distance_non_count = read_txt(different_source_path)

    # draw_plot(distance_isogeny_count, distance_non_count)

    draw_seaborn(distance_isogeny_count, distance_non_count)



