from funkcije import transformImage,transAffine2D,showImage,saveImage
from hornSchunk import HornSchunck
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.pyplot import hist2d
from plots_and_reads import quiverOnImage,optFlowColorVisualisation
import numpy as np
from horn_schunck_piramida import HSpiramida
import scipy as si
import cv2
#program za slike-> video in obratno

fourcc=cv2.VideoWriter_fourcc(*'XVID')
out=cv2.VideoWriter('C:/VideosRV/out_neobdelan.avi',fourcc,10,(720,480))
N=31
path_with_name='C:/VideosRV/img'
filetype='png'
for znj in range(1,N+1):
        p="{}{:04}.{}".format(path_with_name,znj,filetype)
        slika = np.array(Image.open(p))
        plt.imshow(slika)
        #slika=cv2.cvtColor(slika,cv2.COLOR_BGR2GRAY) #sivinska slika
        #out.write(slika)

out.release()