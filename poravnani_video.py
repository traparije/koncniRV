from funkcije import transformImage,transAffine2D,showImage,saveImage
from hornSchunk import HornSchunck
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.pyplot import hist2d
from plots_and_reads import quiverOnImage,optFlowColorVisualisation,genImgsIntoArray
import numpy as np
import scipy as si
from HS_piramidna import HornSchunckPiramidna,bicubicInterpolateWarp
invPar=np.array([0,0])

gen=genImgsIntoArray('C:/VideosRV/img','png',31)
I0=next(gen)
uskup=np.zeros(I0.shape,dtype=np.float32)
vskup=np.zeros(I0.shape,dtype=np.float32)
i=0
parInv=np.array((0,0),dtype=np.float32)
uMed=0
vMed=0
for img in gen:
    i+=1
    print(i)
    I1=img
    #u,v=HornSchunck(I0,I1,0.1,100)
    #showImage(I0)
    #showImage(I1)
    u,v=HornSchunckPiramidna(I0,I1,7,0.001,0.5,150,10,5)
    # print(np.median(u.flatten())) #večina pikslov se verjetno premakne za toliko po x
    # print(np.median(v.flatten())) #večina pikslov se verjerno premakne za toliko po y
    quiverOnImage(u,v,I0,scale=1,step=10)
    optFlowColorVisualisation(u,v,I0) 

    #nbins=1000
    #razpon=(-15,15)
    #h,xe,ye,m=hist2d(uskup.flatten(),vskup.flatten(),bins=(nbins,nbins), range=(razpon,razpon))
    #najskup=np.argmax(h)
    #najy,najx=int(najskup%nbins),int(najskup//nbins)
    #parInv += np.array([ razpon[0] + (razpon[1]-razpon[0])/nbins*(najx-1), razpon[0] + (razpon[1]-razpon[0])/nbins*(najy-1) ])
    #iBack=transformImage(I1,transAffine2D(iTrans = (-parInv[0],-parInv[1])))
    uMed +=np.median(u.flatten())
    vMed +=np.median(v.flatten())
    UFix=uMed*np.ones(I1.shape)
    VFix=vMed*np.ones(I1.shape)
    #saveImage('C:/VideosRV/imgfixed{}'.format(i+1),bicubicInterpolateWarp(I1,uskup,vskup),'png')
    #saveImage('C:/VideosRV/imgfixed{}'.format(i+1),bicubicInterpolateWarp(I1,UFix,VFix),'png')
    #saveImage('C:/VideosRV/imgfixed{}'.format(i+1),iBack,'png')
    I0=I1

    

