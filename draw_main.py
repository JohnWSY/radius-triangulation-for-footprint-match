import matplotlib.pyplot as plt
import seaborn as sns
from utils import *
from scipy import stats



def draw_plot(distance_isogeny_count, distance_non_count):



    # plot画出对比的直方图

    # plt.hist(distance_isogeny_count, density=0, bins=500, facecolor='r', cumulative=False, label='same source')
    #
    # plt.hist(distance_non_count, density=0, bins=500, facecolor='cyan', alpha=0.8, cumulative=False, label='different source')

    plt.hist(distance_isogeny_count, density=1, bins=500, facecolor='r', cumulative=False, label='same source')

    plt.hist(distance_non_count, density=1, bins=500, facecolor='cyan', alpha = 0.8, cumulative=False, label='different source')

    plt.legend()

    plt.savefig('F:\\footprint\\result_kde.jpg', bbox_inches='tight', dpi=500)

    plt.show()



def draw_seaborn(distance_isogeny_count, distance_non_count):

    # seaborn绘图
    sns.set(style='white', palette="muted", color_codes=True)

    sns.distplot(distance_isogeny_count, kde=False, bins=500, color='r', label=['same source'])

    sns.distplot(distance_non_count, kde=False, bins=500, color='cyan', label=['different source'])

    plt.show()

    t1=len(distance_isogeny_count)**(-0.2)
    t2=len(distance_non_count)**(-0.2)

    sns.kdeplot(distance_isogeny_count, label=['same source'], bw=t1, shade=True, color='r')

    sns.kdeplot(distance_non_count, label=['different source'], bw=t2, shade=True, color='cyan')

    # sns.rugplot(distance_isogeny_count, color='r')
    #
    # sns.rugplot(distance_non_count, color='cyan')

    plt.show()

    sns.kdeplot(distance_isogeny_count, label=['same source'], bw=t1, color='r', cumulative=True)

    sns.kdeplot(distance_non_count, label=['different source'], bw=t2, color='cyan', cumulative=True)

    plt.show()


if __name__ == '__main__':
    same_source_path = r'F:\\footprint\same source(without direct).txt'

    different_source_path = r'F:\\footprint\different source(without direct).txt'

    distance_isogeny_count = read_txt(same_source_path)

    distance_non_count = read_txt(different_source_path)

    # draw_plot(distance_isogeny_count, distance_non_count)

    draw_seaborn(distance_isogeny_count, distance_non_count)



