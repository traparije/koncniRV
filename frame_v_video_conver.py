from hornSchunk import HornSchunck
from PIL import Image
from plots_and_reads import quiverOnImage,optFlowColorVisualisation
import numpy as np
import cv2
#program za slike-> video in obratno

fourcc=cv2.VideoWriter_fourcc(*'XVID')
out=cv2.VideoWriter('C:/VideosRV/out_obdelan.avi',fourcc,10,(720,480))
N=31
path_with_name='C:/VideosRV/img'
filetype='png'
for znj in range(1,N+1):
        p="{}{}.{}".format(path_with_name,znj,filetype)
        slika = np.array(Image.open(p))
        out.write(slika)
out.release()