"""Statistics: refit noise across fold seeds; capacity of the last block (LayerNorm, higher C, nonlinear probe); paired CLIP vs CommFor; R4 alternative curve forms."""
import sys,os; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from common import *
import pickle
D=data(); y,fold,g=D["y"],D["fold"],D["groups"]; W=boot(D); n=len(y); r=RR.r20()
def F(c,b): return getblk("none",c,b)
print("## A refit noise: 6 new caption-family fold seeds, block 6 and 12, C=.01")
out={6:[],12:[]}
for s in range(1,7):
    fs=r.make_folds(y,g,5,1000+s)
    for b in (6,12):
        p=cv_pred([F(c,b) for c in TMIX],y,fs,{PRIM:F(PRIM,b)},C=0.01); out[b].append(bacc(y,p[PRIM]))
    print(s,round(out[6][-1],3),round(out[12][-1],3),flush=True)
print("b6 mean/sd/range",np.mean(out[6]).round(3),np.std(out[6],ddof=1).round(4),min(out[6]),max(out[6]));print("b12",np.mean(out[12]).round(3),np.std(out[12],ddof=1).round(4))
print("gap per seed",np.round(np.array(out[6])-np.array(out[12]),3))
print("\n## B last-block capacity at primary (T-mix)")
for C in (0.001,0.01,0.1,1,10):
    p=cv_pred([F(c,12) for c in TMIX],y,fold,{PRIM:F(PRIM,12)},C=C); print("b12 LR C",C,round(bacc(y,p[PRIM]),3),flush=True)
for C in (0.01,0.1,1):
    p=cv_pred([F(c,6) for c in TMIX],y,fold,{PRIM:F(PRIM,6)},C=C); print("b6 LR C",C,round(bacc(y,p[PRIM]),3),flush=True)
# CLS vs mean
for nm,sl in (("CLS",slice(0,768)),("mean",slice(768,1536))):
    for b in (6,12):
        p=cv_pred([F(c,b)[:,sl] for c in TMIX],y,fold,{PRIM:F(PRIM,b)[:,sl]},C=0.01); print(f"b{b} {nm}",round(bacc(y,p[PRIM]),3),flush=True)
# post-LN CLS (standard use of DINOv2 last layer)
import torch
from transformers import AutoModel
try:
    from features import load_backbone
    bb=load_backbone(RR.CFG["backbone"],input_size=224,layers=(12,),use_cls=True,use_mean=True)
    ln=bb.model.layernorm.eval()
    def LN(X):
        with torch.no_grad(): return ln(torch.from_numpy(X)).numpy()
    Xs=[LN(F(c,12)[:,:768]) for c in TMIX]; Xp=LN(F(PRIM,12)[:,:768])
    for C in (0.01,0.1,1):
        p=cv_pred(Xs,y,fold,{PRIM:Xp},C=C); print("b12 post-LN CLS C",C,round(bacc(y,p[PRIM]),3),flush=True)
except Exception as e: print("LN failed",e)
# nonlinear
from sklearn.decomposition import PCA; from sklearn.preprocessing import StandardScaler; from sklearn.svm import SVC; from sklearn.neural_network import MLPClassifier
from concurrent.futures import ProcessPoolExecutor
def nl_task(a):
    b,f,kind=a
    tr,te=np.where(fold!=f)[0],np.where(fold==f)[0]
    Xtr=np.concatenate([F(c,b)[tr] for c in TMIX]); ytr=np.tile(y[tr],3); sc=StandardScaler().fit(Xtr); pc=PCA(200,random_state=0).fit(sc.transform(Xtr))
    Z=pc.transform(sc.transform(Xtr)); Zt=pc.transform(sc.transform(F(PRIM,b)[te]))
    if kind=="rbf": clf=SVC(C=10,gamma="scale",class_weight="balanced").fit(Z,ytr)
    else: clf=MLPClassifier((512,),alpha=1e-2,max_iter=200,early_stopping=True,random_state=0).fit(Z,ytr)
    return te,clf.predict(Zt)
for b in (12,6):
    for kind in ("rbf","mlp"):
        with ProcessPoolExecutor(5) as ex: rs=list(ex.map(nl_task,[(b,f,kind) for f in range(5)]))
        p=np.zeros(n,int)
        for te,pp in rs: p[te]=pp
        print(f"b{b} PCA200+{kind}",round(bacc(y,p),3),flush=True)
print("\n## C paired backbones (nested over mid blocks, jpeg75_resize05 + held-out)")
RES=json.load(open(FR/"r1r4/results/results.json"))
for bb,d in RES["backbones"].items():
    print(bb,{c:round(d["nested_mid"]["bacc"][c][0],3) for c in d["nested_mid"]["bacc"]},"last",{c:round(d["gaps"][c]["last_bacc"][0],3) for c in d["gaps"]})
BB={}
for bb in RES["backbones"]:
    res=pickle.load(open(FR/f"r1r4/cache/probe_{bb}.pkl","rb")); L=RES["backbones"][bb]["L"]
    mids=[l for l in range(L) if .25-1e-9<=(l+1)/L<=.75+1e-9]
    BB[bb]={}
    for c in (PRIM,"screenshot_sim","mavali_medium"):
        lp=np.zeros((n,8))
        for f in range(5):
            best=max(mids,key=lambda l:(round(res[(l,f)]["inner_best"],12),-l)); t=res[(best,f)]; lp[t["te"]]=t["logp"][c]
        BB[bb][c]=r.conf_stats(r.boot_conf(W,y,lp.argmax(1),8))["bacc"]
    BB[bb]["last"]={c:None for c in ()}
for a,b in (("clip_openai","commfor"),("clip_laion","commfor"),("clip_openai","bfree"),("commfor","bfree"),("dinov2","commfor"),("clip_openai","clip_laion")):
    print(a,"-",b,{c:[round(v,3) for v in RR.ci(BB[a][c]-BB[b][c])] for c in (PRIM,"screenshot_sim","mavali_medium")})
print("\n## D R4 alternative curves")
from scipy.optimize import curve_fit
R4=json.load(open(FR/"r1r4/results/r4.json"))
for key,d in R4["models"].items():
    ns=np.array(d["n_eff"]); e=1-np.array([p["bacc_mean"] for p in d["points"]])
    print(key,d["backbone"],d["block_1based"],"n",ns,"bacc",np.round(1-e,3))
    def pl(n,a,b,c): return a*n**(-b)+c
    def p0(n,a,b): return a*n**(-b)
    def lg(n,a,b): return a-b*np.log(n)
    def ex(n,a,b,c): return a*np.exp(-n/b)+c
    forms={"power+c":(pl,(1,.6,.05),([0,0,0],[1e4,5,1])),"pure power c=0":(p0,(1,.5),([0,0],[1e4,5])),"exp+c":(ex,(.5,30,.13),([0,1,0],[5,1e4,1]))}
    for nm,(fn,pp,bd) in forms.items():
        po,_=curve_fit(fn,ns,e,p0=pp,bounds=bd,maxfev=20000); sse=float(((fn(ns,*po)-e)**2).sum())
        print("  ",nm,"params",np.round(po,3),"rmse",round((sse/len(ns))**.5,4),"bacc@90",round(1-float(fn(90.,*po)),3),"@1000",round(1-float(fn(1000.,*po)),3),"@10000",round(1-float(fn(1e4,*po)),3))
    # truncated-data forecasting: fit power+c on first 3 points is exactly determined; first 4 then predict all
    for k in (3,4):
        po,_=curve_fit(pl,ns[:k],e[:k],p0=(1,.6,.05),bounds=([0,0,0],[1e4,5,1]),maxfev=20000)
        print(f"   fit first {k} pts: b={po[1]:.3f} c={po[2]:.3f}; predicts bacc@102 {1-pl(102.,*po):.3f} (obs {1-e[-1]:.3f}); @1000 {1-pl(1000.,*po):.3f}")
    print("   slope per doubling: ",[round(float(np.log2(ns[i+1]/ns[i])**-1*(e[i]-e[i+1])),3) for i in range(4)])
