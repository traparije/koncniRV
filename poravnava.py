from funkcije import transformImage,transAffine2D,showImage
#from hornSchunk import *
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.pyplot import hist2d
from plots_and_reads import quiverOnImage
import numpy as np
from horn_schunck_piramida import HSpiramida
#oPar=[1,2]
#imgFix = np.array(Image.open('C:/RV/KoncniProjekt/koncniRV/Frames/ulica.jpg').convert('L'), dtype=np.float32) #sivinska slika
#iImgMov = transformImage(imgFix, transAffine2D(iTrans = oPar))
#showImage(imgFix)
#showImage(iImgMov)
#u,v=HornSchunck(imgFix,iImgMov,0.2,9)
imgFix = np.array(Image.open('C:/RV/KoncniProjekt/koncniRV/Frames/a.png').convert('L'), dtype=np.float32)
iImgMov = np.array(Image.open('C:/RV/KoncniProjekt/koncniRV/Frames/b.png').convert('L'), dtype=np.float32)
u,v=HSpiramida(imgFix,iImgMov,alpha=100,eps=0.0001,nj=0.5,nScales=5,nWarps=50,maxiter=150)
print('u',u,u.shape)
#quiverOnImage(u,v,imgFix)
print(((np.amin(v),np.amax(v)),(np.amin(u),np.amax(u))))
#hist2d(v.flatten(),u.flatten(),bins=(100,100), range=((-5,5),(-5,5)))
#plt.show()
print('end')