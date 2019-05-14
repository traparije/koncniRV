from funkcije import *
from hornSchunk import *
from PIL import Image
from matplotlib.pyplot import hist2d
from plots_and_reads import quiverOnImage
import numpy as np
oPar=[0,10]
imgFix = np.array(Image.open('C:/RV/KoncniProjekt/koncniRV/Frames/ulica.jpg').convert('L'), dtype=np.float32) #sivinska slika
iImgMov = transformImage(imgFix, transAffine2D(iTrans = oPar))
showImage(imgFix)
showImage(iImgMov)
u,v=HornSchunck(imgFix,iImgMov,0.2,9)
quiverOnImage(u,v,imgFix)
print(((np.amin(v),np.amax(v)),(np.amin(u),np.amax(u))))
hist2d(v.flatten(),u.flatten(),bins=(100,100), range=((-5,5),(-5,5)))
plt.show()
