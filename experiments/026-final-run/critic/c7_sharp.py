"""Follow-up: why do mild pre-laundering blur and unsharp collapse the frozen T-mix probe to 0.52? Dose-response, order of operations, attraction classes, and leave-one-out augmentation."""
import sys,os; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from common import *
import c2_extract as X
from PIL import Image, ImageFilter
from scipy.ndimage import gaussian_filter
from concurrent.futures import ThreadPoolExecutor
from perturb import jpeg, resize, jpeg75_resize05
D=data(); y,fold=D["y"],D["fold"]; W=boot(D); paths,shas=D["paths"],D["shas"]; n=len(y)
pre=lambda s:(lambda im: jpeg75_resize05(im.filter(ImageFilter.GaussianBlur(s))))
uns=lambda p:(lambda im: jpeg75_resize05(im.filter(ImageFilter.UnsharpMask(radius=2,percent=p,threshold=0))))
def mid(s): return lambda im: jpeg(resize(im,.5).filter(ImageFilter.GaussianBlur(s)),75)
def post(s): return lambda im: jpeg75_resize05(im).filter(ImageFilter.GaussianBlur(s))
def gpost(s):
    def f(im):
        a=np.asarray(jpeg75_resize05(im)).astype(np.float32); return Image.fromarray(np.clip(np.rint(gaussian_filter(a,sigma=(s,s,0),truncate=4.0)),0,255).astype(np.uint8))
    return f
OPS={"pre_blur_0.5":pre(.5),"pre_blur_1":pre(1),"pre_blur_2":pre(2),"pre_blur_3":pre(3),"unsharp_50":uns(50),"unsharp_100":uns(100),"unsharp_150":uns(150),
     "mid_blur_0.5(256 scale, before JPEG)":mid(.5),"mid_blur_1(256 scale, before JPEG)":mid(1),"post_blur_0.5(after JPEG,PIL)":post(.5),"post_blur_1(after JPEG,PIL)":post(1),"post_blur_1.5(after JPEG,PIL)":post(1.5),"post_blur_1.5(scipy)":gpost(1.5)}
emb=RR.Embedder(batch=64)
def get(nm):
    f=CACHE/("s7_"+RR.safe(nm)+".npy")
    if f.exists(): return np.load(f)
    def one(i):
        with Image.open(paths[i]) as im:
            im.load(); return RR.crop224(OPS[nm](im.convert("RGB")))
    with ThreadPoolExecutor(12) as ex: imgs=list(ex.map(one,range(n)))
    x=emb.embed(imgs)[:,5,:].astype(np.float32); np.save(f,x); return x
Fx={k:get(k) for k in OPS}; log("features done")
Xt=[getblk("none",c,6) for c in TMIX]; ev=dict(Fx); ev[PRIM]=getblk("none",PRIM,6)
p=cv_pred(Xt,y,fold,ev,C=0.01)
print("op | frozen T-mix b6 bacc | top predicted class share")
for k in ev:
    c=np.bincount(p[k],minlength=8)/n; t=c.argmax()
    print(f"{k} | {bacc(y,p[k]):.3f} | {CLASSES[t]} {c[t]:.2f}",flush=True)
print("per-class recall under pre_blur_2:",{CLASSES[c]:round((p['pre_blur_2'][y==c]==c).mean(),2) for c in range(8)})
print("per-class recall under unsharp_150:",{CLASSES[c]:round((p['unsharp_150'][y==c]==c).mean(),2) for c in range(8)})
print("predicted-share pre_blur_2:",dict(zip(CLASSES,np.round(np.bincount(p['pre_blur_2'],minlength=8)/n,2))))
print("predicted-share unsharp_150:",dict(zip(CLASSES,np.round(np.bincount(p['unsharp_150'],minlength=8)/n,2))))
# augmentation leave-one-out in the sharpness family
for trk,tek in (("pre_blur_2","unsharp_150"),("unsharp_150","pre_blur_2"),("pre_blur_1","pre_blur_3"),("unsharp_50","unsharp_150")):
    q=cv_pred(Xt+[Fx[trk]],y,fold,{tek:Fx[tek],trk:Fx[trk],PRIM:ev[PRIM]},C=0.01)
    print(f"T-mix+{trk}: {tek} {bacc(y,q[tek]):.3f}, {trk} {bacc(y,q[trk]):.3f}, primary {bacc(y,q[PRIM]):.3f}",flush=True)
q=cv_pred(Xt+[Fx["pre_blur_2"],Fx["unsharp_150"]],y,fold,{"pre_blur_1":Fx["pre_blur_1"],"unsharp_100":Fx["unsharp_100"],PRIM:ev[PRIM]},C=0.01)
print("T-mix+blur2+unsharp150:",{k:round(bacc(y,q[k]),3) for k in q})
