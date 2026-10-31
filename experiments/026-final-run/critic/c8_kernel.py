"""Why PIL GaussianBlur(1.5) and scipy gaussian_filter(1.5) differ on the frozen probe: measured blur strength on 40 laundered images."""
import sys,os; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from common import *
from PIL import Image, ImageFilter
from scipy.ndimage import gaussian_filter, laplace
B=np.load(CACHE/"base_jpeg75_resize05.npy",mmap_mode="r")
idx=np.linspace(0,1019,40).astype(int)
def lapvar(a): return float(np.mean([laplace(a[...,i].astype(np.float32)).var() for i in range(3)]))
imp=np.zeros(201,np.uint8); 
# impulse response std (empirical sigma)
I=np.zeros((101,101,3),np.uint8); I[50,50]=255
pi=np.asarray(Image.fromarray(I).filter(ImageFilter.GaussianBlur(1.5))).astype(float)[...,0]
print("PIL impulse peak/sum",pi.max(),pi.sum())
big=np.zeros((101,101),np.float32); big[50,50]=1; sc=gaussian_filter(big,1.5,truncate=4.0)
yy,xx=np.mgrid[:101,:101]
print("empirical sigma scipy",float(np.sqrt((sc*(xx-50)**2).sum()/sc.sum())))
I2=np.zeros((101,101,3),np.uint8); I2[45:56,45:56]=255
a=np.asarray(Image.fromarray(I2).filter(ImageFilter.GaussianBlur(1.5))).astype(np.float32)[...,0]; b=gaussian_filter(I2[...,0].astype(np.float32),1.5,truncate=4.0)
print("edge-profile MAE between kernels (0-255) on a square",float(np.abs(a-b).mean()),"max",float(np.abs(a-b).max()))
print("mean Laplacian variance: raw",np.mean([lapvar(B[i]) for i in idx]),"PIL",np.mean([lapvar(np.asarray(Image.fromarray(B[i]).filter(ImageFilter.GaussianBlur(1.5)))) for i in idx]),"scipy",np.mean([lapvar(np.clip(np.rint(gaussian_filter(B[i].astype(np.float32),(1.5,1.5,0),truncate=4.0)),0,255)) for i in idx]))
d=[np.abs(np.asarray(Image.fromarray(B[i]).filter(ImageFilter.GaussianBlur(1.5))).astype(float)-np.clip(np.rint(gaussian_filter(B[i].astype(np.float32),(1.5,1.5,0),truncate=4.0)),0,255)).mean() for i in idx]
print("mean |PIL-scipy| on images",float(np.mean(d)))
