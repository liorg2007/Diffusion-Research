"""RAID replication at every block + manipulations (independent image set, 4 anonymous generators, n from 020 ha1_raid; T-mix = clean, jpeg(75), jpeg(90))."""
import sys,os; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from common import *
import c2_extract as X
from PIL import Image
from perturb import apply_condition
from concurrent.futures import ThreadPoolExecutor
r=RR.r20()
df,u=r.load_universe("fov224"); fj=json.loads((ROOT/"experiments/020-attr-foundation/results/folds_ha1_raid.json").read_text())
pos={s:i for i,s in enumerate(df.sha256)}; d=df.iloc[[pos[s] for s in fj["rows"]]].reset_index(drop=True)
fold=np.array(fj["fold"]); cl=["d3_gen_0","d3_gen_1","d3_gen_2","d3_gen_3"]; y=np.array([cl.index(g) for g in d.generator]); print("n",len(y),np.bincount(y),"folds",np.bincount(fold))
conds=["clean","jpeg(75)","jpeg(90)"]; prim="jpeg(75)"
emb=RR.Embedder(batch=64)
def base(c):
    def one(i):
        with Image.open(d.path[i]) as im:
            im.load(); return np.asarray(RR.crop224(apply_condition(im,c,key=d.sha256[i],seed=0)))
    with ThreadPoolExecutor(12) as ex: return np.stack(list(ex.map(one,range(len(d)))))
Bs={c:base(c) for c in conds}
ms=["none","shuf32","shuf14","eqgray","blur3","blur6","shuf14_blur15_eqgray"]
F={}
for m in ms:
    for c in conds:
        with ThreadPoolExecutor(8) as ex: A=list(ex.map(lambda i:X.MA[m](Bs[c][i].astype(np.float32),d.sha256[i]),range(len(d))))
        F[(m,c)]=emb.embed([Image.fromarray(np.clip(np.rint(a),0,255).astype(np.uint8)) for a in A])
    log(m)
print("block profile none, T-mix, C=.01, jpeg(75) (020: b4 .703 b8 .645 b12 .525; chance .25):")
prof=[]
for b in range(12):
    p=cv_pred([F[("none",c)][:,b] for c in conds],y,fold,{prim:F[("none",prim)][:,b]},C=0.01); prof.append(round(bacc(y,p[prim]),3))
print(prof,flush=True)
for m in ms:
    row=[]
    for b in (3,5,7,11):
        p=cv_pred([F[(m,c)][:,b] for c in conds],y,fold,{prim:F[(m,prim)][:,b]},C=0.01); row.append(round(bacc(y,p[prim]),3))
    print(m,"b4,b6,b8,b12",row,flush=True)
# gen1-3 restricted at b6 fresh 3 way (no unique-size class)
p=cv_pred([F[("none",c)][:,5] for c in conds],y,fold,{prim:F[("none",prim)][:,5]},C=0.01,classes=[1,2,3]); m=y>0; print("3-way gen1-3 b6",round(bacc(y[m],p[prim][m]),3))
