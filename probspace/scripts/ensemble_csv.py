import pandas as pd
import datetime

now = datetime.datetime.now()

a=pd.read_csv("data/output/sub_2019042819:1853_0.42552998130288994.csv", encoding="utf-8")
b=pd.read_csv("data/output/sub_2019042819:2004_0.42497908247260846.csv", encoding="utf-8")
c=pd.read_csv("data/output/sub_2019042819:2051_0.4254333252977684.csv", encoding="utf-8")
d=pd.read_csv("data/output/sub_2019042819:2137_0.4257718442951776.csv", encoding="utf-8")
e=pd.read_csv("data/output/sub_2019042819:2523_0.42608413248071714.csv", encoding="utf-8")
f=pd.read_csv("data/output/sub_2019042819:3044_0.42568074625815416.csv", encoding="utf-8")
g=pd.read_csv("data/output/sub_2019042819:3140_0.4252821581436168.csv", encoding="utf-8")
h=pd.read_csv("data/output/sub_2019042819:3243_0.42610028800109667.csv", encoding="utf-8")
i=pd.read_csv("data/output/sub_2019042819:3347_0.4256315994751148.csv", encoding="utf-8")
j=pd.read_csv("data/output/sub_2019042915:5537_0.42538124357068763.csv",encoding="utf-8")
k=pd.read_csv("data/output/sub_2019042916:0610_0.4254753322842629.csv",encoding="utf-8")
l=pd.read_csv("data/output/sub_2019042916:1436_0.42608602069605755.csv",encoding="utf-8")
m=pd.read_csv("data/output/sub_2019042916:1756_0.4252546966353918.csv",encoding="utf-8")

target = a.columns[-1]

ans = (a[target]+b[target]+c[target]+d[target]+e[target]+f[target]+g[target]+h[target]+i[target]+j[target]+k[target]+l[target]+m[target]).apply(lambda x : 0 if x < 6 else 1)

#ID_name = config['ID_name']
ID_name = "ID"
sub = pd.DataFrame(pd.read_csv('./data/input/test.csv')[ID_name])
sub["Y"] = (ans.values).astype(int)
sub.to_csv(
    './data/output/sub_{0:%Y%m%d%H:%M%S}_ensemble.csv'.format(now),
    index=False
)
