"""Cached-feature checks, DINOv2-base block 6, C=0.01 fixed (the two values the nested run chose were 0.01/0.1; no test tuning)."""
import sys,os; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from common import *
import pickle
D=data(); y,fold=D["y"],D["fold"]; W=boot(D); df=D["df"]; n=len(y)
B=6
def F(c,m="none",b=B): return getblk(m,c,b)
# r1r4 cache: mavali
import importlib.util as iu
def r14(bb,cond):
    store={}
    for p in sorted((FR/"r1r4/cache"/bb/RR.safe(cond)).glob("part_*.npz")):
        z=np.load(p,allow_pickle=False); store.update(zip(z["keys"].tolist(),z["feats"]))
    return np.stack([store[s] for s in D["shas"]])
X_mav=r14("dinov2","mavali_medium"); X_mav=X_mav[:,B-1].reshape(n,-1)
log("mavali loaded",X_mav.shape)
evals={"clean":F("clean"),PRIM:F(PRIM),"social_sim":F("social_sim"),"screenshot_sim":F("screenshot_sim"),"mavali_medium":X_mav}
for c in ["jr(95,0.5)","jr(75,1.0)","jr(45,1.0)","jr(30,1.0)","jr(30,0.5)","jr(95,0.35)","jr(75,0.35)","jr(75,0.25)","jr(95,0.25)","jr(95,1.0)","jr(95,0.75)","jr(75,0.75)"]: evals[c]=F(c)
TR={"T-mix (reference)":["clean",PRIM,"social_sim"],
    "clean only":["clean"],
    "clean+JPEG-only (q75,q45 f=1)":["clean","jr(75,1.0)","jr(45,1.0)"],
    "clean+resize-mostly (q95; f=0.75,0.5)":["clean","jr(95,0.75)","jr(95,0.5)"],
    "clean+screenshot_sim (resample, no JPEG)":["clean","screenshot_sim"],
    "clean+social_sim (no 0.5x)":["clean","social_sim"],
    "clean+j75r05":["clean",PRIM],
    "j75r05 only":[PRIM],
    "T-mix+ screenshot+mav (all 5)":["clean",PRIM,"social_sim","screenshot_sim","MAV"],}
keys=[PRIM,"screenshot_sim","mavali_medium","clean","jr(95,0.5)","jr(75,1.0)","jr(45,1.0)","jr(30,1.0)","jr(30,0.5)","jr(95,0.35)","jr(75,0.25)"]
res={}
print("## A. leave-laundering-family-out, block 6, bacc per eval (train cells marked by the training-set name)")
print("train | "+" | ".join(keys))
for nm,cs in TR.items():
    Xl=[evals[c] if c in evals else X_mav for c in cs]
    p=cv_pred(Xl,y,fold,{k:evals[k] for k in keys},C=0.01)
    res[nm]={k:bacc(y,p[k]) for k in keys}
    print(nm,"|"," | ".join(f"{res[nm][k]:.3f}" for k in keys),flush=True)
    if nm.startswith(("T-mix (ref","clean+JPEG","clean+resize")):
        for k in ("screenshot_sim","mavali_medium",PRIM):
            c,_=bacc_ci(W,y,p[k]); print("   CI",k,[round(v,3) for v in c])
    if nm.startswith("T-mix (ref"): pref=p
pickle.dump(pref,open(CACHE/"pref_b6.pkl","wb"))
json.dump(res,open(HERE/"c1_launder.json","w"),indent=1)
# ---- B cross-manipulation transfer
print("\n## B. cross-manipulation: T-mix probe on row manipulation -> tested on column manipulation at primary (retrain diag vs frozen off-diag)")
ms=["none","a_shuffle","b_blur","c_highpass","d_gray"]
ev={m:F(PRIM,m) for m in ms}
print("train\\test | "+" | ".join(ms))
for m in ms:
    p=cv_pred([F(c,m) for c in TMIX],y,fold,ev,C=0.01)
    print(m,"|"," | ".join(f"{bacc(y,p[k]):.3f}" for k in ms),flush=True)
# ---- C confusion, within-batch
pr=pref[PRIM]; 
print("\n## C. confusion (8-way, block 6, jpeg75_resize05, rows true)")
cm=np.zeros((8,8),int); np.add.at(cm,(y,pr),1)
print(CLASSES); print(cm); print("recall",dict(zip(CLASSES,np.round(cm.diagonal()/cm.sum(1),3))))
sb=[0,1,2,3,4,5]; gr=[6,7]; 
is_sb=np.isin(y,sb)
print("Across-batch errors (true SB->pred GRIP or vice versa) share of errors:",((np.isin(y,sb)&np.isin(pr,gr))|(np.isin(y,gr)&np.isin(pr,sb))).sum()/ (pr!=y).sum())
exp=(cm.sum(1,keepdims=True)-cm.diagonal()[:,None]); 
print("sd3->flux",cm[6,7],"flux->sd3",cm[7,6],"sd3 err",cm[6].sum()-cm[6,6],"flux err",cm[7].sum()-cm[7,7])
def restr(cl):
    lg=None
    return cl
Xl=[F(c) for c in TMIX]
p6=cv_pred(Xl,y,fold,{PRIM:F(PRIM),"screenshot_sim":F("screenshot_sim"),"mavali":X_mav},C=0.01,classes=sb)
m=is_sb
print("Synthbuster-only 6-way (fresh probe, chance .167):",{k:round(bacc(y[m],p6[k][m]),3) for k in p6})
print("per-class recall 6-way",{CLASSES[c]:round((p6[PRIM][y==c]==c).mean(),3) for c in sb})
m2=np.isin(y,gr); p2=cv_pred(Xl,y,fold,{PRIM:F(PRIM),"screenshot_sim":F("screenshot_sim")},C=0.01,classes=gr)
print("sd3 vs flux 2-way (GRIP batch only, chance .5):",{k:round(bacc(y[m2],p2[k][m2]),3) for k in p2})
# batch separation: 2-way synthbuster vs GRIP, using only generators with the same size 1024^2 dalle2 vs sd3+flux etc.
yb=(np.isin(y,gr)).astype(int); pb=cv_pred(Xl,yb,fold,{PRIM:F(PRIM)},C=0.01)
print("batch detector SB vs GRIP bacc:",round(bacc(yb,pb[PRIM]),3))
# leave one synthbuster generator out then is it called GRIP? i.e. is 'GRIP-ness' a generic pipeline cue? train batch detector w/o class c, test on class c
for held in sb:
    tm=y!=held
    from sklearn.preprocessing import StandardScaler; from sklearn.linear_model import LogisticRegression
    out=np.zeros(n,int)
    for f in range(5):
        tr=np.where((fold!=f)&tm)[0]; te=np.where((fold==f)&(y==held))[0]
        Xtr=np.concatenate([X[tr] for X in Xl]); ytr=np.tile(yb[tr],3); sc,clf=fit_lr(Xtr,ytr,0.01)
        out[te]=clf.predict(sc.transform(F(PRIM)[te]))
    print(f"  batch detector trained w/o {CLASSES[held]}: share of that generator called GRIP = {out[y==held].mean():.3f}")
# hold out GRIP generator too
for held in gr:
    tm=y!=held; out=np.zeros(n,int)
    for f in range(5):
        tr=np.where((fold!=f)&tm)[0]; te=np.where((fold==f)&(y==held))[0]
        Xtr=np.concatenate([X[tr] for X in Xl]); ytr=np.tile(yb[tr],3); sc,clf=fit_lr(Xtr,ytr,0.01)
        out[te]=clf.predict(sc.transform(F(PRIM)[te]))
    print(f"  batch detector trained w/o {CLASSES[held]}: share of that generator called GRIP = {out[y==held].mean():.3f}")
# ---- D per-class mid-last gap
p12=cv_pred([F(c,b=12) for c in TMIX],y,fold,{PRIM:F(PRIM,b=12)},C=0.01)
print("\n## D. per-class recall b6 vs b12:",{CLASSES[c]:(round((pr[y==c]==c).mean(),3),round((p12[PRIM][y==c]==c).mean(),3)) for c in range(8)})
c1,_a=bacc_ci(W,y,pr); c2,_b=bacc_ci(W,y,p12[PRIM]); print("b6",np.round(c1,3),"b12",np.round(c2,3),"gap",np.round(RR.ci(_a-_b),3))
