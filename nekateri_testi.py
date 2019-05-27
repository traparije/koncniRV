from funkcije import transformImage,transAffine2D,showImage,saveImage
from hornSchunk import HornSchunck
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.pyplot import hist2d
from plots_and_reads import quiverOnImage,optFlowColorVisualisation,genImgsIntoArray
import numpy as np
from HS_piramidna import HornSchunckPiramidna,bicubicInterpolateWarp

I0=np.array(Image.open('C:/koncniRV/Rezultati/avto/img0018.png').convert('L'),dtype=np.float32)
I1=np.array(Image.open('C:/koncniRV/Rezultati/avto/img0019.png').convert('L'),dtype=np.float32)

#u,v=HornSchunck(I0,I1,0.1,100)
#showImage(I0)
#showImage(I1)
#u,v=HornSchunckPiramidna(I0,I1,7,0.0001,0.5,150,10,5)
u,v=HornSchunckPiramidna(I0,I1,7,0.01,0.5,150,15,5)
# print(np.median(u.flatten())) #večina pikslov se verjetno premakne za toliko po x
# print(np.median(v.flatten())) #večina pikslov se verjerno premakne za toliko po y
quiverOnImage(u,v,I0,scale=3,step=20)
optFlowColorVisualisation(u,v,I0) 

#nbins=1000
#razpon=(-15,15)
#h,xe,ye,m=hist2d(uskup.flatten(),vskup.flatten(),bins=(nbins,nbins), range=(razpon,razpon))
#najskup=np.argmax(h)
#najy,najx=int(najskup%nbins),int(najskup//nbins)
#parInv += np.array([ razpon[0] + (razpon[1]-razpon[0])/nbins*(najx-1), razpon[0] + (razpon[1]-razpon[0])/nbins*(najy-1) ])
#iBack=transformImage(I1,transAffine2D(iTrans = (-parInv[0],-parInv[1])))
#uMed +=np.median(u.flatten())
#vMed +=np.median(v.flatten())
#UFix=uMed*np.ones(I1.shape)
#VFix=vMed*np.ones(I1.shape)
#saveImage('C:/VideosRV/imgfixed{}'.format(i+1),bicubicInterpolateWarp(I1,uskup,vskup),'png')
#saveImage('C:/VideosRV/imgfixed{}'.format(i+1),bicubicInterpolateWarp(I1,UFix,VFix),'png')
#saveImage('C:/VideosRV/imgfixed{}'.format(i+1),iBack,'png')
I0=I1
