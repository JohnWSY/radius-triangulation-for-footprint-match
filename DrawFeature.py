import cv2
from VertexSort import *

class DrawFeature(object):


    def __init__(self, feature1, feature2):
        '''
        :param feature1: 特征对象1
        :param feature2: 特征对象2
        '''
        self.feature1 = feature1
        self.feature2 = feature2

    def draw(self, img_path1, img_path2, r=50, color_poly = (0, 0, 255), color_radius = (0, 255, 0), color_cen = (100, 100, 200), color_direct = (0, 255, 255)):
        '''
        :param img_path1: 图1路径
        :param img_path2: 图2路径
        :param r: 特征方向可视化的延长半径
        :param color_poly: 多边形线颜色
        :param color_radius: 径向线颜色
        :param color_cen: 径向剖分中心点颜色
        :param color_direct: 特征方向颜色
        :return:
        '''
        img1=cv2.imread(img_path1, 1)
        img2=cv2.imread(img_path2, 1)

        c1_x, c1_y = centriodloc(self.feature1.x_fv, self.feature1.y_fv)
        c2_x, c2_y = centriodloc(self.feature2.x_fv, self.feature2.y_fv)

        fv1 = VertexSort(self.feature1.featuredirect, self.feature1.x_fv, self.feature1.y_fv, c1_x, c1_y)
        fv2 = VertexSort(self.feature2.featuredirect, self.feature2.x_fv, self.feature2.y_fv, c2_x, c2_y)


        pts1 = [[fv1.x[i], fv1.y[i]] for i in range(len(fv1.x))]
        pts1 = np.array(pts1, dtype = np.int32).reshape(-1, 1, 2)

        pts2 = [[fv2.x[i], fv2.y[i]] for i in range(len(fv2.x))]
        pts2 = np.array(pts2, dtype = np.int32).reshape(-1, 1, 2)

        # 画出多边形
        cv2.polylines(img1, [pts1], True, thickness=5, color=color_poly, lineType=cv2.LINE_AA)
        cv2.polylines(img2, [pts2], True, thickness=5, color=color_poly, lineType=cv2.LINE_AA)

        # 画出中心圆
        cv2.circle(img1, (c1_x, c1_y), 1, color_cen, thickness=50)
        cv2.circle(img2, (c2_x, c2_y), 1, color_cen, thickness=50)

        for i in range(len(fv1.x)):
            #  radius line
            cv2.line(img1, (int(c1_x), int(c1_y)), (int(fv1.x[i]), int(fv1.y[i])), color=color_radius, thickness=2, lineType=cv2.LINE_AA)
            cv2.line(img2, (int(c2_x), int(c2_y)), (int(fv2.x[i]), int(fv2.y[i])), color=color_radius, thickness=2, lineType=cv2.LINE_AA)

            x1 = int(fv1.x[i])
            y1 = int(fv1.y[i])
            x_end1 = int(x1+r*math.cos(fv1.featuredirect[i]))
            y_end1 = int(y1+r*math.sin(fv1.featuredirect[i]))

            x2 = int(fv2.x[i])
            y2 = int(fv2.y[i])
            x_end2 = int(x2 + r * math.cos(fv2.featuredirect[i]))
            y_end2 = int(y2 + r * math.sin(fv2.featuredirect[i]))
            # feature direct
            cv2.line(img1, (x1, y1), (x_end1, y_end1), color=color_direct, thickness=10, lineType=cv2.LINE_AA)
            cv2.line(img2, (x2, y2), (x_end2, y_end2), color=color_direct, thickness=10, lineType=cv2.LINE_AA)



        cv2.namedWindow('image1', cv2.WINDOW_NORMAL)
        cv2.namedWindow('image2', cv2.WINDOW_NORMAL)
        cv2.imshow('image1', img1)
        cv2.imshow('image2', img2)

        cv2.waitKey(0)

# if __name__ == '__main__':
#     csv1 = 'F:\\footprint\same source\ES005S1\ES005S1-L\\L01S1.csv'
#     csv2 = 'F:\\footprint\same source\ES005S1\ES005S1-L\\L03S1.csv'
    #
    # # 同源计算
    # lc1, lc2 = feature_vec_same(csv1, csv2)
    #
    # # 非同源计算根据选完特征的lc计算
    #
    # img_path1 = csv1.replace('.csv', '.jpg')
    # img_path2 = csv2.replace('.csv', '.jpg')

    # img = DrawFeature(lc1, lc2)
    # img.draw(img_path1, img_path2)

