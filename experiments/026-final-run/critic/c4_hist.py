"""Is the signal a global tone/colour histogram or simple texture statistic of the 224 model input? T-mix training, primary test, same folds.
C for LR is chosen on the TEST folds (max over grid): optimistic, deliberately in favour of the trivial explanation. HGB = default settings."""
import sys,os; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from common import *
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from scipy.ndimage import laplace, sobel
D=data(); y,fold=D["y"],D["fold"]; W=boot(D); n=len(y)
conds=TMIX+["screenshot_sim"]
B={c:np.load(CACHE/f"base_{RR.safe(c)}.npy",mmap_mode="r") for c in conds}
def lbp(g):
    c=g[1:-1,1:-1]; nb=[g[:-2,:-2],g[:-2,1:-1],g[:-2,2:],g[1:-1,2:],g[2:,2:],g[2:,1:-1],g[2:,:-2],g[1:-1,:-2]]
    return np.bincount((sum((x>=c).astype(np.int32)<<i for i,x in enumerate(nb))).ravel(),minlength=256)/c.size
fy=np.fft.fftfreq(224)[:,None]; fx=np.fft.fftfreq(224)[None,:]; RAD=np.minimum((np.sqrt(fx**2+fy**2)/0.5*32).astype(int),31)
def fe(a):
    a=a.astype(np.float32); g=a@np.array([.299,.587,.114],dtype=np.float32); o={}
    hs=[np.histogram(a[...,i],32,(0,255))[0]/a[...,i].size for i in range(3)]
    o["rgb_hist"]=np.concatenate(hs); o["luma_hist"]=np.histogram(g,64,(0,255))[0]/g.size
    x=a.reshape(-1,3); mu=x.mean(0); sd=x.std(0)+1e-3; z=(x-mu)/sd
    pct=np.percentile(x,[1,5,25,50,75,95,99],axis=0).ravel()
    mx,mn=x.max(1),x.min(1); sat=(mx-mn)/(mx+1e-3)
    o["moments"]=np.concatenate([mu,sd,(z**3).mean(0),(z**4).mean(0),pct,[sat.mean(),sat.std()],np.corrcoef(x.T)[np.triu_indices(3,1)]])
    q=(a//32).astype(int); o["rgb3d"]=np.bincount((q[...,0]*64+q[...,1]*8+q[...,2]).ravel(),minlength=512)/q[...,0].size
    gx,gy=sobel(g,1),sobel(g,0); m=np.hypot(gx,gy); o["grad"]=np.concatenate([np.histogram(np.log1p(m),32,(0,7))[0]/m.size,np.histogram(np.arctan2(gy,gx),16,(-np.pi,np.pi),weights=m)[0]/(m.sum()+1e-3)])
    L=laplace(g); o["lap"]=np.histogram(np.sign(L)*np.log1p(np.abs(L)),32,(-7,7))[0]/L.size
    o["lbp"]=lbp(g.astype(np.int32))
    P=np.abs(np.fft.fft2(g-g.mean()))**2; o["psd"]=np.log(np.bincount(RAD.ravel(),P.ravel(),32)/np.bincount(RAD.ravel(),minlength=32)+1e-3)
    return o
fp=CACHE/"hist_feats.npz"
if fp.exists(): Z=dict(np.load(fp))
else:
    Z={}
    for c in conds:
        with ThreadPoolExecutor(14) as ex: R=list(ex.map(lambda i:fe(np.asarray(B[c][i])),range(n)))
        for k in R[0]: Z[f"{k}|{RR.safe(c)}"]=np.stack([r[k] for r in R])
        log("features",c)
    np.savez(fp,**Z)
def X(keys,c): return np.concatenate([Z[f"{k}|{RR.safe(c)}"] for k in keys],1)
SETS={"mean RGB + std only":("MS",None),"luma_hist":["luma_hist"],"rgb_hist":["rgb_hist"],"moments":["moments"],"rgb3d":["rgb3d"],"TONE all (rgb,luma,moments,3d)":["rgb_hist","luma_hist","moments","rgb3d"],
 "grad+lap":["grad","lap"],"lbp":["lbp"],"psd":["psd"],"TEXTURE all (grad,lap,lbp,psd)":["grad","lap","lbp","psd"],"TONE+TEXTURE":["rgb_hist","luma_hist","moments","rgb3d","grad","lap","lbp","psd"]}
def xs(v,c):
    if v[0]=="MS": return Z[f"moments|{RR.safe(c)}"][:,:6]
    return X(v,c)
def hgbt(a):
    f,Xt,yt,Xe=a
    from sklearn.ensemble import HistGradientBoostingClassifier as H
    return H(random_state=0,class_weight="balanced",max_iter=150).fit(Xt,yt).predict(Xe)
print("feature set | dim | LR best C(optimistic) bacc@primary [CI] | HGB bacc@primary | LR@screenshot")
for nm,v in SETS.items():
    Xl=[xs(v,c) for c in TMIX]; ev={PRIM:xs(v,PRIM),"shot":xs(v,"screenshot_sim")}
    best=None
    for C in (0.01,0.1,1,10):
        p=cv_pred(Xl,y,fold,ev,C=C); b=bacc(y,p[PRIM])
        if best is None or b>best[0]: best=(b,C,p)
    ci_,_=bacc_ci(W,y,best[2][PRIM])
    pg=np.zeros(n,int)
    tasks=[]
    for f in range(5):
        tr=np.where(fold!=f)[0]; te=np.where(fold==f)[0]
        tasks.append((f,np.concatenate([x[tr] for x in Xl]),np.tile(y[tr],3),ev[PRIM][te]))
    os.environ["OMP_NUM_THREADS"]="3"
    with ProcessPoolExecutor(5) as ex: rr=list(ex.map(hgbt,tasks))
    for f,pp in enumerate(rr): pg[fold==f]=pp
    print(f"{nm} | {Xl[0].shape[1]} | {ci_[0]:.3f} [{ci_[1]:.3f},{ci_[2]:.3f}] C={best[1]} | {bacc(y,pg):.3f} | {bacc(y,best[2]['shot']):.3f}",flush=True)
    if nm.startswith("TONE all"):
        print("  recall per class",{CLASSES[c]:round((best[2][PRIM][y==c]==c).mean(),3) for c in range(8)})
M=Z[f"moments|{RR.safe(PRIM)}"]
print("\nper-generator mean[R,G,B], std of luminance-ish (mean per-channel std), mean saturation (primary):")
for c in range(8):
    m=y==c; print(CLASSES[c],np.round(M[m,:3].mean(0),1),"sd",np.round(M[m,3:6].mean(),1),"between-image sd of mean",np.round(M[m,:3].mean(1).std(),1),"sat",round(M[m,-5].mean(),3))
