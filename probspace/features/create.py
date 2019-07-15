# -- coding: utf-8 --
import pandas as pd
import feather
import numpy as np
import re as re
import argparse
from sklearn.model_selection import KFold,StratifiedKFold
import json
import sys
import os
sys.path.append(os.getcwd())
sys.path.append(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath( __file__ )), '..')))
from base import Feature, get_arguments, generate_features, generate_dataframe, get_features_to_json
from function import data, config_json
from tqdm import tqdm
import scipy.stats

Feature.dir = 'features'
""""
class one_hot_encoding(Feature):
    def create_features(self):
        train_oht, test_oht = data.category_columns_to_one_hot_train_test(X_train_origin, test_origin)

        data.check_columns_size(train_oht, test_oht)
        self.train = train_oht
        self.test = test_oht

#数値カラムのNanフラグ
class Nan_flag(Feature):
    def create_features(self):
        train_nan, test_nan = data.get_nan_flag(X_train_origin, test_origin)

        data.check_columns_size(train_nan, test_nan)
        self.train = train_nan
        self.test = test_nan

#数値カラムの0カウントエンコーディング
class Zero_percent(Feature):
    def create_features(self):
        df_all_zero_per = data.get_zero_percent(df_all_origin)
        train_zero_per = df_all_zero_per[:train_test_split_index]
        test_zero_per = df_all_zero_per[train_test_split_index:]
        test_zero_per = test_zero_per.reset_index(drop=True)

        data.check_columns_size(train_zero_per, test_zero_per)
        self.train = train_zero_per
        self.test = test_zero_per

#0以下のフラグ立て
class  minus_flag(Feature):
    def create_features(self):
        df_new = pd.DataFrame(index=df_all.index, columns=[])
        for column in df_all.columns:
            df_new["{}_is_underZero".format(column)] = df_all[column].apply(lambda x: 1 if x <= 0 else 0)

        self.train = df_new[:train_test_split_index]
        self.test = df_new[train_test_split_index:].reset_index(drop=True)
"""

class  gender_marriage(Feature):
    def create_features(self):
        df_new = pd.DataFrame(index=df_all.index, columns=[])
        column="gender_marraige"
        fil = (df_all.gender == 1) | (df_all.marriage == 1)
        df_new.loc[fil, column] = 1
        fil = (df_all.gender == 1) | (df_all.marriage == 2)
        df_new.loc[fil, column] = 2
        fil = (df_all.gender == 1) | (df_all.marriage == 3)
        df_new.loc[fil, column] = 3
        fil = (df_all.gender == 2) | (df_all.marriage == 1)
        df_new.loc[fil, column] = 4
        fil = (df_all.gender == 2) | (df_all.marriage == 2)
        df_new.loc[fil, column] = 5
        fil = (df_all.gender == 2) | (df_all.marriage == 3)
        df_new.loc[fil, column] = 6

        self.train = df_new[:train_test_split_index]
        self.test = df_new[train_test_split_index:].reset_index(drop=True)

class  gender_education(Feature):
    def  create_features(self):
        df_new = pd.DataFrame(index=df_all.index, columns=[])
        column = "gender_education"
        fil = (df_all.gender == 1) | (df_all.education == 1)
        df_new.loc[fil, column] = 1
        fil = (df_all.gender == 1) | (df_all.education == 2)
        df_new.loc[fil, column] = 2
        fil = (df_all.gender == 1) | (df_all.education == 3)
        df_new.loc[fil, column] = 3
        fil = (df_all.gender == 1) | (df_all.education == 4)
        df_new.loc[fil, column] = 4
        fil = (df_all.gender == 2) | (df_all.education == 1)
        df_new.loc[fil, column] = 5
        fil = (df_all.gender == 2) | (df_all.education == 2)
        df_new.loc[fil, column] = 6
        fil = (df_all.gender == 2) | (df_all.education == 3)
        df_new.loc[fil, column] = 7
        fil = (df_all.gender == 2) | (df_all.education == 4)
        df_new.loc[fil, column] = 8

        self.train = df_new[:train_test_split_index]
        self.test = df_new[train_test_split_index:].reset_index(drop=True)


class marriage_education(Feature):
    def create_features(self):
        df_new = pd.DataFrame(index=df_all.index, columns=[])
        column = "marriage_education"
        fil = (df_all.marriage == 1) | (df_all.education == 1)
        df_new.loc[fil, column] = 1
        fil = (df_all.marriage == 1) | (df_all.education == 2)
        df_new.loc[fil, column] = 2
        fil = (df_all.marriage == 1) | (df_all.education == 3)
        df_new.loc[fil, column] = 3
        fil = (df_all.marriage == 1) | (df_all.education == 4)
        df_new.loc[fil, column] = 4
        fil = (df_all.marriage == 2) | (df_all.education == 1)
        df_new.loc[fil, column] = 5
        fil = (df_all.marriage == 2) | (df_all.education == 2)
        df_new.loc[fil, column] = 6
        fil = (df_all.marriage == 2) | (df_all.education == 3)
        df_new.loc[fil, column] = 7
        fil = (df_all.marriage == 2) | (df_all.education == 4)
        df_new.loc[fil, column] = 8
        fil = (df_all.marriage == 3) | (df_all.education == 1)
        df_new.loc[fil, column] = 9
        fil = (df_all.marriage == 3) | (df_all.education == 2)
        df_new.loc[fil, column] = 10
        fil = (df_all.marriage == 3) | (df_all.education == 3)
        df_new.loc[fil, column] = 11
        fil = (df_all.marriage == 3) | (df_all.education == 4)
        df_new.loc[fil, column] = 12

        self.train = df_new[:train_test_split_index]
        self.test = df_new[train_test_split_index:].reset_index(drop=True)

class  credit_divide_age(Feature):
    def create_features(self):
        df_new = pd.DataFrame(index=df_all.index, columns=[])
        column = "credit_divide_age"
        df_new[column] = df_all["credit"]/df_all["age"]

        self.train = df_new[:train_test_split_index]
        self.test = df_new[train_test_split_index:].reset_index(drop=True)

class  age_bin(Feature):
    def create_features(self):
        df_new = pd.DataFrame(index=df_all.index, columns=[])
        df_new['ageBin'] = 0 #creates a column of 0
        df_new.loc[((df_all['age'] > 20) & (df_all['age'] < 30)) , 'ageBin'] = 1
        df_new.loc[((df_all['age'] >= 30) & (df_all['age'] < 40)) , 'ageBin'] = 2
        df_new.loc[((df_all['age'] >= 40) & (df_all['age'] < 50)) , 'ageBin'] = 3
        df_new.loc[((df_all['age'] >= 50) & (df_all['age'] < 60)) , 'ageBin'] = 4
        df_new.loc[((df_all['age'] >= 60) & (df_all['age'] < 70)) , 'ageBin'] = 5
        df_new.loc[((df_all['age'] >= 70) & (df_all['age'] < 81)) , 'ageBin'] = 6

        self.train = df_new[:train_test_split_index]
        self.test = df_new[train_test_split_index:].reset_index(drop=True)

class  age_gender(Feature):
    def create_features(self):
        df = pd.DataFrame(index=df_all.index, columns=[])
        df['age_gender'] = 0
        df.loc[((df_all.gender == 1) & (df_feature_all.ageBin == 1)), 'age_gender'] = 1  # man in 20's
        df.loc[((df_all.gender == 1) & (df_feature_all.ageBin == 2)), 'age_gender'] = 2  # man in 30's
        df.loc[((df_all.gender == 1) & (df_feature_all.ageBin == 3)), 'age_gender'] = 3  # man in 40's
        df.loc[((df_all.gender == 1) & (df_feature_all.ageBin == 4)), 'age_gender'] = 4  # man in 50's
        df.loc[((df_all.gender == 1) & (df_feature_all.ageBin == 5)), 'age_gender'] = 5  # man in 60's and above
        df.loc[((df_all.gender == 1) & (df_feature_all.ageBin == 6)), 'age_gender'] = 6  # man in 70's and above
        df.loc[((df_all.gender == 2) & (df_feature_all.ageBin == 1)), 'age_gender'] = 7  # woman in 20's
        df.loc[((df_all.gender == 2) & (df_feature_all.ageBin == 2)), 'age_gender'] = 8  # woman in 30's
        df.loc[((df_all.gender == 2) & (df_feature_all.ageBin == 3)), 'age_gender'] = 9  # woman in 40's
        df.loc[((df_all.gender == 2) & (df_feature_all.ageBin == 4)), 'age_gender'] = 10  # woman in 50's
        df.loc[((df_all.gender == 2) & (df_feature_all.ageBin == 5)), 'age_gender'] = 11  # woman in 60's and above
        df.loc[((df_all.gender == 2) & (df_feature_all.ageBin == 6)), 'age_gender'] = 12  # man in 60's and above

        self.train = df[:train_test_split_index]
        self.test = df[train_test_split_index:].reset_index(drop=True)


class  client(Feature):
    def create_features(self):
        df = pd.DataFrame(index=df_all.index, columns=[])
        df['Client_9'] = 1
        df['Client_8'] = 1
        df['Client_7'] = 1
        df['Client_6'] = 1
        df['Client_5'] = 1
        df['Client_4'] = 1
        df.loc[((df_all.payment_9 == 0) & (df_all.claim_9 == 0) & (df_all.advance_9 == 0)) , 'Client_9'] = 0
        df.loc[((df_all.payment_8 == 0) & (df_all.claim_8 == 0) & (df_all.advance_8 == 0)) , 'Client_8'] = 0
        df.loc[((df_all.payment_7 == 0) & (df_all.claim_7 == 0) & (df_all.advance_7 == 0)) , 'Client_7'] = 0
        df.loc[((df_all.payment_6 == 0) & (df_all.claim_6 == 0) & (df_all.advance_6 == 0)) , 'Client_6'] = 0
        df.loc[((df_all.payment_5 == 0) & (df_all.claim_5 == 0) & (df_all.advance_5 == 0)) , 'Client_5'] = 0
        df.loc[((df_all.payment_4 == 0) & (df_all.claim_4 == 0) & (df_all.advance_4 == 0)) , 'Client_4'] = 0

        self.train = df[:train_test_split_index]
        self.test = df[train_test_split_index:].reset_index(drop=True)

class  payment_divide_credit(Feature):
    def create_features(self):
        df = pd.DataFrame(index=df_all.index, columns=[])
        df["payment_4_divide_credit"] = (1+df_all["payment_4"]*100000)/df_all["credit"]
        df["payment_5_divide_credit"] = (1 + df_all["payment_5"]*100000) / df_all["credit"]
        df["payment_6_divide_credit"] = (1 + df_all["payment_6"]*100000) / df_all["credit"]
        df["payment_7_divide_credit"] = (1 + df_all["payment_7"]*100000) / df_all["credit"]
        df["payment_8_divide_credit"] = (1 + df_all["payment_8"]*100000) / df_all["credit"]
        df["payment_9_divide_credit"] = (1 + df_all["payment_9"]*100000) / df_all["credit"]


        self.train = df[:train_test_split_index]
        self.test = df[train_test_split_index:].reset_index(drop=True)

class  payment(Feature):
    def create_features(self):
        df = pd.DataFrame(index=df_all.index, columns=[])
        df["payment_sum"] = df_all[["payment_4", "payment_5", "payment_6", "payment_7", "payment_8", "payment_9"]].sum(axis=1)
        df["payment_non_zeros"] = (df_all[["payment_4","payment_5","payment_6","payment_7","payment_8","payment_9"]]>0).sum(axis=1)
        df["payment_var"] = df_all[["payment_4", "payment_5", "payment_6", "payment_7", "payment_8", "payment_9"]].var(axis=1)

        self.train = df[:train_test_split_index]
        self.test = df[train_test_split_index:].reset_index(drop=True)

class  advance(Feature):
    def create_features(self):
        df = pd.DataFrame(index=df_all.index, columns=[])
        df["advance_sum"] = df_all[["advance_4", "advance_5", "advance_6", "advance_7", "advance_8", "advance_9"]].sum(axis=1)
        df["advance_non_zeros"] = (df_all[["advance_4", "advance_5", "advance_6", "advance_7", "advance_8", "advance_9"]]>0).sum(axis=1)
        df["advance_var"] = df_all[["advance_4", "advance_5", "advance_6", "advance_7", "advance_8", "advance_9"]].var(axis=1)

        self.train = df[:train_test_split_index]
        self.test = df[train_test_split_index:].reset_index(drop=True)

class  claim(Feature):
    def create_features(self):
        df = pd.DataFrame(index=df_all.index, columns=[])
        df["claim_sum"] = df_all[["claim_4", "claim_5", "claim_6", "claim_7", "claim_8", "claim_9"]].sum(axis=1)
        df["claim_non_zeros"] = (df_all[["claim_4", "claim_5", "claim_6", "claim_7", "claim_8", "claim_9"]]>0).sum(axis=1)
        df["claim_var"] = df_all[["claim_4", "claim_5", "claim_6", "claim_7", "claim_8", "claim_9"]].var(axis=1)

        self.train = df[:train_test_split_index]
        self.test = df[train_test_split_index:].reset_index(drop=True)

""""
class payment_left(Feature):
    def create_features(self):
        df = pd.DataFrame(index=df_all.index, columns=[["payment_left_{}".format(i) for i in range(4,10)]])
        for key, row in tqdm(df_all.iterrows()):
            for i in range(4, 10):
                df["payment_left_{}".format(i)].ix[key] = row[["claim_{}".format(k) for k in range(i - row["payment_{}".format(i)], i)]].sum(axis=0) if i - row["payment_{}".format(i)] >= 4 else (4 - i + row["payment_{}".format(i)]) * row[["claim_{}".format(k) for k in range(4, 10)]].mean(axis=0) + row[["claim_{}".format(k) for k in range(4, i)]].sum(axis=0)

        self.train = df[:train_test_split_index]
        self.test = df[train_test_split_index:].reset_index(drop=True)
"""

class payment_1mounth(Feature):
    def create_features(self):
        df = pd.DataFrame(index=df_all.index, columns=[])
        df["payment_4_5"] = df_all["payment_5"]-df_all["payment_4"]
        df["payment_5_6"] = df_all["payment_6"] - df_all["payment_5"]
        df["payment_6_7"] = df_all["payment_7"] - df_all["payment_6"]
        df["payment_7_8"] = df_all["payment_8"] - df_all["payment_7"]
        df["payment_8_9"] = df_all["payment_9"] - df_all["payment_8"]

        self.train = df[:train_test_split_index]
        self.test = df[train_test_split_index:].reset_index(drop=True)


class claim_1mounth(Feature):
    def create_features(self):
        df = pd.DataFrame(index=df_all.index, columns=[])
        df["claim_4_5"] = df_all["claim_5"] - df_all["claim_4"]
        df["claim_5_6"] = df_all["claim_6"] - df_all["claim_5"]
        df["claim_6_7"] = df_all["claim_7"] - df_all["claim_6"]
        df["claim_7_8"] = df_all["claim_8"] - df_all["claim_7"]
        df["claim_8_9"] = df_all["claim_9"] - df_all["claim_8"]

        self.train = df[:train_test_split_index]
        self.test = df[train_test_split_index:].reset_index(drop=True)


class advance_1mounth(Feature):
    def create_features(self):
        df = pd.DataFrame(index=df_all.index, columns=[])
        df["advance_4_5"] = df_all["advance_5"] - df_all["advance_4"]
        df["advance_5_6"] = df_all["advance_6"] - df_all["advance_5"]
        df["advance_6_7"] = df_all["advance_7"] - df_all["advance_6"]
        df["advance_7_8"] = df_all["advance_8"] - df_all["advance_7"]
        df["advance_8_9"] = df_all["advance_9"] - df_all["advance_8"]

        self.train = df[:train_test_split_index]
        self.test = df[train_test_split_index:].reset_index(drop=True)

class payment_4_9(Feature):
    def create_features(self):
        df = pd.DataFrame(index=df_all.index, columns=[])
        df["payment_4_9"] = df_all["payment_9"]-df_all["payment_4"]

        self.train = df[:train_test_split_index]
        self.test = df[train_test_split_index:].reset_index(drop=True)


class claim_4_9(Feature):
    def create_features(self):
        df = pd.DataFrame(index=df_all.index, columns=[])
        df["claim_4_9"] = df_all["claim_9"] - df_all["claim_4"]

        self.train = df[:train_test_split_index]
        self.test = df[train_test_split_index:].reset_index(drop=True)


class advance_4_9(Feature):
    def create_features(self):
        df = pd.DataFrame(index=df_all.index, columns=[])
        df["advance_4_9"] = df_all["advance_9"] - df_all["advance_4"]

        self.train = df[:train_test_split_index]
        self.test = df[train_test_split_index:].reset_index(drop=True)

class advance_claim(Feature):
    def create_features(self):
        df = pd.DataFrame(index=df_all.index, columns=[])
        df["advance_claim_4_5"] = df_all["advance_4"] - df_all["claim_5"]
        df["advance_claim_5_6"] = df_all["advance_5"] - df_all["claim_6"]
        df["advance_claim_6_7"] = df_all["advance_6"] - df_all["claim_7"]
        df["advance_claim_7_8"] = df_all["advance_7"] - df_all["claim_8"]
        df["advance_claim_8_9"] = df_all["advance_8"] - df_all["claim_9"]

        self.train = df[:train_test_split_index]
        self.test = df[train_test_split_index:].reset_index(drop=True)

class advance_claim_divide_credit(Feature):
    def create_features(self):
        df = pd.DataFrame(index=df_all.index, columns=[])
        df["advance_claim_divide_credit_4_5"] = (df_all["advance_4"] - df_all["claim_5"])/df_all["credit"]
        df["advance_claim_divide_credit_5_6"] = (df_all["advance_5"] - df_all["claim_6"])/df_all["credit"]
        df["advance_claim_divide_credit_6_7"] = (df_all["advance_6"] - df_all["claim_7"])/df_all["credit"]
        df["advance_claim_divide_credit_7_8"] = (df_all["advance_7"] - df_all["claim_8"])/df_all["credit"]
        df["advance_claim_divide_credit_8_9"] = (df_all["advance_8"] - df_all["claim_9"])/df_all["credit"]

        self.train = df[:train_test_split_index]
        self.test = df[train_test_split_index:].reset_index(drop=True)

class payment_times_claim(Feature):
    def create_features(self):
        df = pd.DataFrame(index=df_all.index, columns=[])
        df["payment_times_claim_4"] = df_all["payment_4"] * df_all["claim_4"]
        df["payment_times_claim_5"] = df_all["payment_5"] * df_all["claim_5"]
        df["payment_times_claim_6"] = df_all["payment_6"] * df_all["claim_6"]
        df["payment_times_claim_7"] = df_all["payment_7"] * df_all["claim_7"]
        df["payment_times_claim_8"] = df_all["payment_8"] * df_all["claim_8"]
        df["payment_times_claim_9"] = df_all["payment_9"] * df_all["claim_9"]

        self.train = df[:train_test_split_index]
        self.test = df[train_test_split_index:].reset_index(drop=True)

class payment_2month(Feature):
    def create_features(self):
        df = pd.DataFrame(index=df_all.index, columns=[])
        df["payment_4_2month"] = df_all["payment_4"]+df_all["payment_5"]+df_all["payment_6"]
        df["payment_5_2month"] = df_all["payment_5"] + df_all["payment_6"] + df_all["payment_7"]
        df["payment_6_2month"] = df_all["payment_6"] + df_all["payment_7"] + df_all["payment_8"]
        df["payment_7_2month"] = df_all["payment_7"] + df_all["payment_8"] + df_all["payment_9"]

        self.train = df[:train_test_split_index]
        self.test = df[train_test_split_index:].reset_index(drop=True)


class claim_2month(Feature):
    def create_features(self):
        df = pd.DataFrame(index=df_all.index, columns=[])
        df["claim_4_2month"] = df_all["claim_4"]+df_all["claim_5"]+df_all["claim_6"]
        df["claim_5_2month"] = df_all["claim_5"] + df_all["claim_6"] + df_all["claim_7"]
        df["claim_6_2month"] = df_all["claim_6"] + df_all["claim_7"] + df_all["claim_8"]
        df["claim_7_2month"] = df_all["claim_7"] + df_all["claim_8"] + df_all["claim_9"]

        self.train = df[:train_test_split_index]
        self.test = df[train_test_split_index:].reset_index(drop=True)


class advance_4_2month(Feature):
    def create_features(self):
        df = pd.DataFrame(index=df_all.index, columns=[])
        df["advance_4_2month"] = df_all["advance_4"]+df_all["advance_5"]+df_all["advance_6"]
        df["advance_5_2month"] = df_all["advance_5"] + df_all["advance_6"] + df_all["advance_7"]
        df["advance_6_2month"] = df_all["advance_6"] + df_all["advance_7"] + df_all["advance_8"]
        df["advance_7_2month"] = df_all["advance_7"] + df_all["advance_8"] + df_all["advance_9"]

        self.train = df[:train_test_split_index]
        self.test = df[train_test_split_index:].reset_index(drop=True)

""""
#コサイン類似度
def cos_sim(v1, v2):
    #エラー回避
    if  np.linalg.norm(v2)==0:
        return -999
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

class payment_cos_sim(Feature):
    def create_features(self):
        df = pd.DataFrame(index=df_all.index, columns=["payment_cos_sim_normal","payment_cos_sim_default"])
        data_std = scipy.stats.zscore(train_all[["payment_{}".format(i) for i in range(4,10)]], axis=1)
        data_std = pd.DataFrame(data_std, index=train_all.index,columns=[["payment_{}".format(i) for i in range(4,10)]])
        data_std = data_std.fillna(0)
        normal_cos_mean = data_std[train_all["y"]== 0].mean(axis=0)
        default_cos_mean = data_std[train_all["y"] == 1].mean(axis=0)
        for i in range(len(df_all)):
            df["payment_cos_sim_normal"].ix[i]=cos_sim(normal_cos_mean,df_all[["payment_{}".format(i) for i in range(4,10)]].loc[i])
            df["payment_cos_sim_default"].ix[i] = cos_sim(default_cos_mean,
                                                  df_all[["payment_{}".format(i) for i in range(4, 10)]].loc[i])

            df["payment_cos_sim_normal"]=df["payment_cos_sim_normal"].apply(lambda x: 1 if x == -999 else x)
            df["payment_cos_sim_default"] = df["payment_cos_sim_default"].apply(lambda x: 0 if x == -999 else x)
        self.train = df[:train_test_split_index]
        self.test = df[train_test_split_index:].reset_index(drop=True)

class advance_cos_sim(Feature):
    def create_features(self):
        df = pd.DataFrame(index=df_all.index, columns=["advance_cos_sim_normal","advance_cos_sim_default"])
        data_std = scipy.stats.zscore(train_all[["advance_{}".format(i) for i in range(4,10)]], axis=1)
        data_std = pd.DataFrame(data_std, index=train_all.index,columns=[["advance_{}".format(i) for i in range(4,10)]])
        data_std = data_std.fillna(0)
        normal_cos_mean = data_std[train_all["y"]== 0].mean(axis=0)
        default_cos_mean = data_std[train_all["y"] == 1].mean(axis=0)
        for i in range(len(df_all)):
            df["advance_cos_sim_normal"].ix[i]=cos_sim(normal_cos_mean,df_all[["advance_{}".format(i) for i in range(4,10)]].loc[i])
            df["advance_cos_sim_default"].ix[i] = cos_sim(default_cos_mean,
                                                  df_all[["advance_{}".format(i) for i in range(4, 10)]].loc[i])

            df["advance_cos_sim_normal"]=df["advance_cos_sim_normal"].apply(lambda x: 1 if x == -999 else x)
            df["advance_cos_sim_default"] = df["advance_cos_sim_default"].apply(lambda x: 0 if x == -999 else x)
        self.train = df[:train_test_split_index]
        self.test = df[train_test_split_index:].reset_index(drop=True)

class claim_cos_sim(Feature):
    def create_features(self):
        df = pd.DataFrame(index=df_all.index, columns=["claim_cos_sim_normal","claim_cos_sim_default"])
        data_std = scipy.stats.zscore(train_all[["claim_{}".format(i) for i in range(4,10)]], axis=1)
        data_std = pd.DataFrame(data_std, index=train_all.index,columns=[["claim_{}".format(i) for i in range(4,10)]])
        data_std = data_std.fillna(0)
        normal_cos_mean = data_std[train_all["y"]== 0].mean(axis=0)
        default_cos_mean = data_std[train_all["y"] == 1].mean(axis=0)
        for i in range(len(df_all)):
            df["claim_cos_sim_normal"].ix[i]=cos_sim(normal_cos_mean,df_all[["claim_{}".format(i) for i in range(4,10)]].loc[i])
            df["claim_cos_sim_default"].ix[i] = cos_sim(default_cos_mean,
                                                  df_all[["claim_{}".format(i) for i in range(4, 10)]].loc[i])

            df["claim_cos_sim_normal"]=df["claim_cos_sim_normal"].apply(lambda x: 1 if x == -999 else x)
            df["claim_cos_sim_default"] = df["claim_cos_sim_default"].apply(lambda x: 0 if x == -999 else x)
        self.train = df[:train_test_split_index]
        self.test = df[train_test_split_index:].reset_index(drop=True)
"""

class payment_cont_0(Feature):
    def create_features(self):
        df = pd.DataFrame(index=df_all.index, columns=["payment_contCount_0"])
        for  key, row in df_all[["payment_{}".format(i) for i in range(4,10)]].iterrows():
            count=0
            count_max=0
            for i in range(0,5):
                if row[i]==0 and row[i+1]==0:
                    if count==1:count+=1
                    count+=1
                    count_max=max(count_max,count)
                else:
                    count=0
            df["payment_contCount_0"][key]=count_max

        self.train = df[:train_test_split_index]
        self.test = df[train_test_split_index:].reset_index(drop=True)

class advance_cont_0(Feature):
    def create_features(self):
        df = pd.DataFrame(index=df_all.index, columns=["advance_contCount_0"])
        for  key, row in df_all[["advance_{}".format(i) for i in range(4,10)]].iterrows():
            count=0
            count_max=0
            for i in range(0,5):
                if row[i]==0 and row[i+1]==0:
                    if count==1:count+=1
                    count+=1
                    count_max=max(count_max,count)
                else:
                    count=0
            df["advance_contCount_0"][key]=count_max

        self.train = df[:train_test_split_index]
        self.test = df[train_test_split_index:].reset_index(drop=True)

class claim_cont_0(Feature):
    def create_features(self):
        df = pd.DataFrame(index=df_all.index, columns=["claim_contCount_0"])
        for  key, row in df_all[["claim_{}".format(i) for i in range(4,10)]].iterrows():
            count=0
            count_max=0
            for i in range(0,5):
                if row[i]==0 and row[i+1]==0:
                    if count==1:count+=1
                    count+=1
                    count_max=max(count_max,count)
                else:
                    count=0
            df["claim_contCount_0"][key]=count_max

        self.train = df[:train_test_split_index]
        self.test = df[train_test_split_index:].reset_index(drop=True)

class payment_left(Feature):
    def create_features(self):
        df = pd.DataFrame(index=df_all.index)

        flag = (df_all["payment_4"] == 0)
        df.loc[flag,"pay_4"] = df_all["claim_4"][flag]
        flag = df_all["payment_4"] == 1
        df.loc[flag, "pay_4"] = df_all["claim_4"] + (1/6) * (df_all["claim_4"] + df_all["claim_5"][flag] + df_all["claim_6"][flag] + df_all["claim_7"][flag] + df_all["claim_8"][flag] +
                df_all["claim_9"][flag])
        flag = df_all["payment_4"] == 2
        df.loc[flag, "pay_4"] = df_all["claim_4"] + (2/6) * (df_all["claim_4"] + df_all["claim_5"][flag] + df_all["claim_6"][flag] + df_all["claim_7"][flag] + df_all["claim_8"][flag] +
                df_all["claim_9"][flag])
        flag = df_all["payment_4"] == 3
        df.loc[flag, "pay_4"] = df_all["claim_4"] + (3/6) *  (df_all["claim_4"] + df_all["claim_5"][flag] + df_all["claim_6"][flag] + df_all["claim_7"][flag] + df_all["claim_8"][flag] +
                df_all["claim_9"][flag])
        flag = df_all["payment_4"] == 4
        df.loc[flag, "pay_4"] = df_all["claim_4"] + (4/6) * (df_all["claim_4"] + df_all["claim_5"][flag] + df_all["claim_6"][flag] + df_all["claim_7"][flag] + df_all["claim_8"][flag] +
                df_all["claim_9"][flag])
        flag = df_all["payment_4"] == 5
        df.loc[flag, "pay_4"] = df_all["claim_4"] + (5/6) * (df_all["claim_4"] + df_all["claim_5"][flag] + df_all["claim_6"][flag] + df_all["claim_7"][flag] + df_all["claim_8"][flag] +
                df_all["claim_9"][flag])
        flag = df_all["payment_4"] == 6
        df.loc[flag, "pay_4"] = (df_all["claim_5"][flag] + df_all["claim_6"][flag] + df_all["claim_7"][flag] + df_all["claim_8"][flag] +
                 df_all["claim_9"][flag])
        df.loc[flag, "pay_4"] = df_all["claim_4"] + (6/6) * (df_all["claim_4"] + df_all["claim_5"][flag] + df_all["claim_6"][flag] + df_all["claim_7"][flag] + df_all["claim_8"][flag] +
                df_all["claim_9"][flag])
        flag = df_all["payment_4"] == 7
        df.loc[flag, "pay_4"] = df_all["claim_4"] + (7/6) * (df_all["claim_4"] + df_all["claim_5"][flag] + df_all["claim_6"][flag] + df_all["claim_7"][flag] + df_all["claim_8"][flag] +
                df_all["claim_9"][flag])
        flag = df_all["payment_4"] == 8
        df.loc[flag, "pay_4"] = df_all["claim_4"] + (8/6) * (df_all["claim_4"] + df_all["claim_5"][flag] + df_all["claim_6"][flag] + df_all["claim_7"][flag] + df_all["claim_8"][flag] +
                df_all["claim_9"][flag])

        flag = df_all["payment_9"] == 0
        df.loc[flag, "pay_9"] = df_all["claim_9"][flag]
        flag = df_all["payment_9"] == 1
        df.loc[flag, "pay_9"] = (df_all["claim_8"][flag] + df_all["claim_9"][flag])
        flag = df_all["payment_9"] == 2
        df.loc[flag, "pay_9"] = ( df_all["claim_7"][flag] + df_all["claim_8"][flag] + df_all["claim_9"][flag])
        flag = df_all["payment_9"] == 3
        df.loc[flag, "pay_9"] = (df_all["claim_6"][flag] + df_all["claim_7"][flag] + df_all["claim_8"][flag] + df_all["claim_9"][flag])
        flag = df_all["payment_9"] == 4
        df.loc[flag, "pay_9"] = (df_all["claim_5"][flag] + df_all["claim_6"][flag] + df_all["claim_7"][flag] + df_all["claim_8"][flag] + df_all["claim_9"][flag])
        flag = df_all["payment_9"] == 5
        df.loc[flag, "pay_9"] = (df_all["claim_4"][flag] + df_all["claim_5"][flag] + df_all["claim_6"][flag] + df_all["claim_7"][flag] + df_all["claim_8"][flag] + df_all["claim_9"][flag])
        flag = df_all["payment_9"] == 6
        df.loc[flag, "pay_9"] = (1/6)*(df_all["claim_4"][flag] + df_all["claim_5"][flag] + df_all["claim_6"][flag] + df_all["claim_7"][flag] + df_all["claim_8"][flag] + df_all["claim_9"][flag])+(df_all["claim_4"][flag] + df_all["claim_5"][flag] + df_all["claim_6"][flag] + df_all["claim_7"][flag] + df_all["claim_8"][flag] + df_all["claim_9"][flag])
        flag = df_all["payment_9"] == 7
        df.loc[flag, "pay_9"] =  (2/6)*(df_all["claim_4"][flag] + df_all["claim_5"][flag] + df_all["claim_6"][flag] + df_all["claim_7"][flag] + df_all["claim_8"][flag] + df_all["claim_9"][flag])+(df_all["claim_4"][flag] + df_all["claim_5"][flag] + df_all["claim_6"][flag] + df_all["claim_7"][flag] + df_all["claim_8"][flag] + df_all["claim_9"][flag])
        flag = df_all["payment_9"] == 8
        df.loc[flag, "pay_9"] = (3/6)*(df_all["claim_4"][flag] + df_all["claim_5"][flag] + df_all["claim_6"][flag] + df_all["claim_7"][flag] + df_all["claim_8"][flag] + df_all["claim_9"][flag])+(df_all["claim_4"][flag] + df_all["claim_5"][flag] + df_all["claim_6"][flag] + df_all["claim_7"][flag] + df_all["claim_8"][flag] + df_all["claim_9"][flag])

        df = df.fillna(0)

        self.train = df[:train_test_split_index]
        self.test = df[train_test_split_index:].reset_index(drop=True)

"""
class payment_left(Feature):
    def create_features(self):
        df = pd.DataFrame(columns=[["payment_left_{}".format(i) for i in range(4,10)]],index=df_all.index)
        for  key, row in df_all.iterrows():
            for i in range(4,10):
                df["payment_left_{}".format(i)].ix[key]= row[["claim_{}".format(k) for k in range(i-row["payment_{}".format(i)],i)]].sum(axis=0) if i-row["payment_{}".format(i)]>=4 else (4-i+row["payment_{}".format(i)])*row[["claim_{}".format(k) for k in range(4,10)]].mean(axis=0)+ row[["claim_{}".format(k) for k in range(4,i)]].sum(axis=0)

        self.train = df[:train_test_split_index]
        self.test = df[train_test_split_index:].reset_index(drop=True)

class payment_0_count(Feature):
    def create_features(self):
        df = pd.DataFrame(index=df_all.index, columns=["payment_0_count"])
        df["payment_0_count"] = (df_all[["payment_{}".format(i) for i in range(4,10)]]==0).sum(axis=1)

        self.train = df[:train_test_split_index]
        self.test = df[train_test_split_index:].reset_index(drop=True)

class advance_0_count(Feature):
    def create_features(self):
        df = pd.DataFrame(index=df_all.index, columns=["advance_0_count"])
        df["advance_0_count"] = (df_all[["advance_{}".format(i) for i in range(4, 10)]]==0).sum(axis=1)

        self.train = df[:train_test_split_index]
        self.test = df[train_test_split_index:].reset_index(drop=True)

class claim_0_count(Feature):
    def create_features(self):
        df = pd.DataFrame(index=df_all.index, columns=["claim_0_count"])
        df["claim_0_count"] = (df_all[["claim_{}".format(i) for i in range(4, 10)]]==0).sum(axis=1)

        self.train = df[:train_test_split_index]
        self.test = df[train_test_split_index:].reset_index(drop=True)
"""


if __name__ == '__main__':
    args = get_arguments()
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/default.json')
    options = parser.parse_args()
    config = json.load(open(options.config))

    cv = config["cv"]
    if cv == "kfold":
        kf = KFold(**config["kfold"])
    elif cv == "skfold":
        kf = StratifiedKFold(**config["skfold"])

    #元データ
    train_origin, test_origin = data.get_origin_dataframe()
    # ラベル書き換え
    columns = ["id", "credit", "gender", "education", "marriage", "age"]
    columns += [f"payment_{i}" for i in range(9, 3, -1)]
    columns += [f"claim_{i}" for i in range(9, 3, -1)]
    columns += [f"advance_{i}" for i in range(9, 3, -1)]
    train_origin.columns = columns + ["y"]
    test_origin.columns = columns

    #none→nan
    data.none_to_nan(train_origin)
    data.none_to_nan(test_origin)

    #df_all_origin : train_origin(-target) + test_origin
    df_all_origin, Y_train_origin, train_test_split_index = data.get_dataframe_set(train_origin, test_origin)
    #X_train_origin : train_origin(-target)
    X_train_origin = df_all_origin[:train_test_split_index]


    #前処理済みデータ
    X_train, Y_train, test =data.get_preprocessing_dataframe()

    #X_train + Y_train
    train_all = pd.concat([X_train, Y_train], axis=1)

    #X_train + test
    df_all = pd.concat([X_train, test],sort=False, ignore_index=True)

    train_feature, X_test_feature = data.get_base_line_dataframe()
    X_train_feature = train_feature.drop([config["target_name"]], axis=1)
    df_feature_all = pd.concat([X_train_feature, X_test_feature]).reset_index(drop=True)


    #特徴量作成
    generate_features(globals(), args.force)

    #作成した特徴量でDataFrame作成
    df_train, df_test = generate_dataframe(globals())

    #作成した特徴量をconfigに追加
    get_features_to_json(globals(),True)

