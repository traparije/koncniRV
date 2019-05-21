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

gen=genImgsIntoArray('C:/koncniRV/Frames/img','png',64)
I0=next(gen).shape
i=1
for img in gen:
    I1=img


    I0=I1

    

