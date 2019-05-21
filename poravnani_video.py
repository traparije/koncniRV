from funkcije import transformImage,transAffine2D,showImage
from hornSchunk import HornSchunck
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.pyplot import hist2d
from plots_and_reads import quiverOnImage,optFlowColorVisualisation,genImgsIntoArray
import numpy as np
from horn_schunck_piramida import HSpiramida
import scipy as si
invPar=np.array([0,0])

gen=genImgsIntoArray('C:/koncniRV/Frames/img','png',6)
I0=next(gen)
i=0
for img in gen:
    i+=1
    print(i)
    I1=img
    u,v=HornSchunck(I0,I1,0.1,100)
    quiverOnImage(u,v,I0,scale=1,step=10)
    nbins=1000
    razpon=(-10,10)
    h,xe,ye,m=hist2d(u.flatten(),v.flatten(),bins=(nbins,nbins), range=(razpon,razpon))
    najskup=np.argmax(h)
    najy,najx=int(najskup%nbins),int(najskup//nbins)
    parInv=[ razpon[0] + (razpon[1]-razpon[0])/nbins*(najx-1), razpon[0] + (razpon[1]-razpon[0])/nbins*(najy-1) ]
    print(parInv)


    I0=I1

    

