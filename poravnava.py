from funkcije import transformImage,transAffine2D,showImage,saveImage
from hornSchunk import HornSchunck
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.pyplot import hist2d
from plots_and_reads import quiverOnImage,optFlowColorVisualisation
import numpy as np
from horn_schunck_piramida import HSpiramida
import scipy as si
oPar=[-3,-3]
#imgFix = np.array(Image.open('C:/RV/KoncniProjekt/koncniRV/Frames/a.png').convert('L'), dtype=np.float32) #sivinska slika
#iImgMov = transformImage(imgFix, transAffine2D(iTrans = oPar))

#imgFix = np.array(Image.open('C:/RV/KoncniProjekt/koncniRV/Frames/a.png').convert('L'), dtype=np.float32) #sivinska slika
#iImgMov = np.array(Image.open('C:/RV/KoncniProjekt/koncniRV/Frames/b.png').convert('L'), dtype=np.float32) #sivinska slika

#showImage(imgFix)
#showImage(iImgMov)
#u,v=HornSchunck(imgFix,iImgMov,0.2,9)
imgFixColor=np.array(Image.open('C:/koncniRV/Frames/img0001.png'))
iImgMovColor=np.array(Image.open('C:/koncniRV/Frames/img0002.png'))
imgFix = np.array(Image.open('C:/koncniRV/Frames/img0001.png').convert('L'), dtype=np.float32)
iImgMov = np.array(Image.open('C:/koncniRV/Frames/img0002.png').convert('L'), dtype=np.float32)
#print(np.array(iImgMov))
#u,v=HornSchunck(imgFix,iImgMov,0.1,100)
#parInv=(si.median(u),si.median(v)) #parametri inverzne preslikave, mediana ni ok
im1=imgFix
im2=iImgMov
from piramidna_poskus2 import  piramidna2,bicubicInterpolateWarp

#u,v=piramidna2(imgFix,iImgMov,alpha=0.3,eps=0.0001,nj=0.5,nScales=5,nWarps=10,maxiter=150)
u,v=piramidna2(imgFix,iImgMov,7,0.001,0.5,150,10,5)
print('u',u,u.shape)

quiverOnImage(u,v,imgFix,scale=1,step=10)
optFlowColorVisualisation(u,v,imgFix) #še ne dela ok.
nbins=1000
razpon=(-5,5)
h,xe,ye,m=hist2d(u.flatten(),v.flatten(),bins=(nbins,nbins), range=(razpon,razpon))
najskup=np.argmax(h)
najy,najx=int(najskup%nbins),int(najskup//nbins)
parInv=[ razpon[0] + (razpon[1]-razpon[0])/nbins*(najx-1), razpon[0] + (razpon[1]-razpon[0])/nbins*(najy-1) ]
iBack=transformImage(iImgMov,transAffine2D(iTrans = (-parInv[0],-parInv[1])))
plt.imshow(iBack,cmap='gray')
plt.show()
plt.imshow(imgFix-iBack,cmap='gray')
plt.show()
#iBackColor=transformImage(iImgMovColor,transAffine2D(iTrans = (-parInv[0],-parInv[1])))
saveImage('C:/koncniRV/Frames/2v1',iBack,'png')
saveImage('C:/koncniRV/Frames/1',im1,'png')
saveImage('C:/koncniRV/Frames/2',im2,'png')
saveImage('C:/koncniRV/Frames/2v1warp',bicubicInterpolateWarp(im2,u,v),'png')

#print('end')
#?