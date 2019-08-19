import matplotlib.pyplot as plt
import seaborn as sns
from utils import *
from scipy import stats



def draw_distribution(same_source_path, different_source_path):

    distance_isogeny_count = read_txt(same_source_path)

    distance_non_count = read_txt(different_source_path)

    # plot画出对比的直方图

    plt.hist(distance_isogeny_count, density=0, bins=500, facecolor='r', cumulative=False, label='same source')

    plt.hist(distance_non_count, density=0, bins=500, facecolor='cyan', alpha=0.3, cumulative=False,
             label='different source')

    plt.legend()

    plt.show()



    # seaborn绘图
    sns.set(style='white', palette="muted", color_codes=True)

    fig, axes = plt.subplots(2, 2)

    sns.distplot(distance_isogeny_count, ax=axes[0, 0], kde=False, bins=500, color='r')

    sns.distplot(distance_non_count, ax=axes[0, 1], kde=False, bins=500, color='cyan')

    sns.distplot(distance_isogeny_count, label=['same source'], ax=axes[1, 0], bins=500, kde=True, color='r')

    sns.distplot(distance_non_count, label=['different source'], ax=axes[1, 1], bins=500, kde=True, color='cyan')

    plt.show()


if __name__ == '__main__':

    draw_distribution('F:\足迹\\footprint\same_source.txt', 'F:\足迹\\footprint\different_source.txt')



