from funkcije import transformImage,transAffine2D,showImage,saveImage
from hornSchunk import HornSchunck
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.pyplot import hist2d
from plots_and_reads import quiverOnImage,optFlowColorVisualisation,genImgsIntoArray
import numpy as np
from horn_schunck_piramida import HSpiramida
import scipy as si
from piramidna_poskus2 import piramidna2,bicubicInterpolateWarp
invPar=np.array([0,0])

gen=genImgsIntoArray('C:/koncniRV/Frames/img','png',64)
I0=next(gen)
uskup=np.zeros(I0.shape,dtype=np.float32)
vskup=np.zeros(I0.shape,dtype=np.float32)
i=0
for img in gen:
    i+=1
    print(i)
    I1=img
    #u,v=HornSchunck(I0,I1,0.1,100)
    #showImage(I0)
    #showImage(I1)
    u,v=piramidna2(I0,I1,7,0.001,0.5,150,10,5)
    #quiverOnImage(u,v,I0,scale=1,step=10)
    #optFlowColorVisualisation(u,v,I0) 
    #nbins=1000
    #razpon=(-10,10)
    #h,xe,ye,m=hist2d(u.flatten(),v.flatten(),bins=(nbins,nbins), range=(razpon,razpon))
    #najskup=np.argmax(h)
    #najy,najx=int(najskup%nbins),int(najskup//nbins)
    #parInv=[ razpon[0] + (razpon[1]-razpon[0])/nbins*(najx-1), razpon[0] + (razpon[1]-razpon[0])/nbins*(najy-1) ]
    #iBack=transformImage(iImgMov,transAffine2D(iTrans = (-parInv[0],-parInv[1])))
    uskup+=u
    vskup+=v
    saveImage('C:/koncniRV/Frames/imgfixed{}'.format(i+1),bicubicInterpolateWarp(I1,uskup,vskup),'png')
    I0=I1

    

