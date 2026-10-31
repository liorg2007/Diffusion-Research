import sys,os; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from common import *
import pandas as pd, json
D=data(); d=D["df"]; print(d.columns.tolist())
print(d.iloc[0].to_dict())
for c in ("source","dataset","pool","kind","src_path"):
    if c in d: print(c, d[c].astype(str).str.slice(0,60).value_counts().head(12).to_dict())
d["top"]=d.src_path.astype(str).str.split("/").str[:5].str.join("/")
print(pd.crosstab(d.top,d.generator))
fam=d.groupby("grp").generator.nunique(); print("family size dist",fam.value_counts().sort_index().to_dict())
d["strat"]=[RR.r20().strata_of(g) for g in d.grp]; print(pd.crosstab(d.strat,d.generator))
# do sd3/flux share families with synthbuster
s=d.groupby("grp").generator.apply(set)
sb={'dalle2','dalle3','firefly','midjourney-v5','stable-diffusion-2','stable-diffusion-xl'}
print("families with sb only",sum(x<=sb for x in s),"with sd3/flux only",sum(not (x&sb) for x in s),"mixed",sum(bool(x&sb) and bool(x-sb) for x in s))
