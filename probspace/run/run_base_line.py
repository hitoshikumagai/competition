# -- coding: utf-8 --
#base lineモデル

import pandas as pd
import feather
import numpy as np
import re as re
import argparse
import json
import sys
import os
sys.path.append(os.getcwd())
from function import data, model_train, submit

parser = argparse.ArgumentParser()
parser.add_argument('--config', default='./configs/default.json')
options = parser.parse_args()
config = json.load(open(options.config))


#データ取得
train, X_test = data.get_base_line_dataframe()
y_train = train[config["target_name"]]
X_train = train.drop([config["target_name"]],axis=1)

#データ削除
"""
X_train = X_train.drop(["payment_8","payment_7","payment_6","payment_5","payment_4"],axis=1)
X_test = X_test.drop(["payment_8","payment_7","payment_6","payment_5","payment_4"],axis=1)
"""

#base lineモデル
y_train_pred, y_test_pred, models, score = model_train.run_lgbm(X_train, y_train, X_test, comment = "base_line",save=False)


#提出
submit.submit_csv(y_test_pred, score)
