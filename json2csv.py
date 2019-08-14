# import GetMinutiaInfo
import pandas as pd
from utils import *
from glob import glob
import os


def json2csv(json_path):
    a = GetMinutiaInfo(json_path).get_minutia_info().features
    direct=[]
    x=[]
    featuretype=[]
    y=[]
    for i in range(len(a)):
        if a[i].__class__.__name__ == 'HoleAB':
            try:
                direct.append(a[i].Direction)
                x.append(a[i].Location[0])
                y.append(a[i].Location[1])
                featuretype.append(a[i].__class__.__name__)
            except Exception as e:
                featuretype.append(a[i].__class__.__name__)
                points = []
                for j in range(len(a[i].holes)):
                    points+=a[i].holes[j].data
                x_h = np.zeros((len(points), 1))
                y_h = np.zeros((len(points), 1))
                for k in range(len(points)):
                    x_h[k] = points[k][0]
                    y_h[k] = points[k][1]
                # 得到Location
                c_x = x_h.mean(axis=0)
                c_y = y_h.mean(axis=0)
                x.append(c_x[0])
                y.append(c_y[0])
                # 得到Direction
                direct.append(GetDirection(x_h, y_h, c_x, c_y))
        elif a[i].__class__.__name__ == 'Shcallamach':
            featuretype.append(a[i].__class__.__name__)
            direct.append(a[i].Direction)
            points=a[i].data
            x_s = np.zeros((len(points), 1))
            y_s = np.zeros((len(points), 1))
            for j in range(len(points)):
                x_s[j] = points[j][0]
                y_s[j] = points[j][1]
            # 得到Location
            c_x = x_s.mean(axis=0)
            c_y = y_s.mean(axis=0)
            x.append(c_x[0])
            y.append(c_y[0])
        else:
            featuretype.append(a[i].__class__.__name__)
            direct.append(a[i].Direction)
            x.append(a[i].Location[0])
            y.append(a[i].Location[1])


    df = pd.DataFrame({'featuretype':featuretype, 'Direction':direct, 'x':x, 'y':y})
    csv_path = json_path.replace('.json', '.csv')
    df.to_csv('%s' %csv_path, index=False)


if __name__ == '__main__':
    # 同源
    path = glob('F:\足迹\\footprint\same source\*\*\*.json')
    # 非同源
    # path = glob('F:\足迹\\footprint\different source\*\*.json')
    list(map(json2csv, path))
    # 删除csv文件
    # csv_path = glob('F:\足迹\\footprint\same source\*\*\*.csv')
    # list(map(os.remove, csv_path))

