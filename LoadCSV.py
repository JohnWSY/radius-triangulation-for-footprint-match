import pandas as pd

class LoadCSV(object):

    def __init__(self, csv_path):

        features_init = pd.read_csv(csv_path, engine='python')

        # 先取出所有特征的信息
        self.featuretype = features_init['featuretype'].values.reshape(len(features_init), 1)
        self.x_fv = features_init['x'].values.reshape(len(features_init), 1)
        self.y_fv = features_init['y'].values.reshape(len(features_init), 1)
        self.featuredirect = features_init['Direction'].values.reshape(len(features_init), 1)
        self.length = len(self.x_fv)