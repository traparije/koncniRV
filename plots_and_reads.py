from PIL import Image
import re
import numpy
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image as im
from funkcije import saveImage
def convertToGray(image):
    dtype = image.dtype
    rgb = np.array(image, dtype='float')
    gray = rgb[:, :, 0]*0.299 + rgb[:, :, 1]*0.587 + rgb[:, :, 2]*0.114
    return gray.astype(dtype)
'''
def read_pgm(filename, byteorder='>'): #vir stackoverflow https://stackoverflow.com/questions/7368739/numpy-and-16-bit-pgm
    """Return image data from a raw PGM file as numpy array.

    Format specification: http://netpbm.sourceforge.net/doc/pgm.html

    """
    with open(filename, 'rb') as f:
        buffer = f.read()
    try:
        header, width, height, maxval = re.search(
            b"(^P5\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n]\s)*)", buffer).groups()
    except AttributeError:
        raise ValueError("Not a raw PGM file: '%s'" % filename)

    return convertToGray(np.array(numpy.frombuffer(buffer,
                            dtype='u1' if int(maxval) < 256 else byteorder+'u2',
                            count=int(width)*int(height),
                            offset=len(header)
                            ).reshape((int(height), int(width))), dtype='float'))
'''

def showImage(iImage, iTitle=''):
    '''
    Prikaže sliko iImage in jo naslovi z iTitle
    
    Parameters
    ----------
    iImage : numpy.ndarray
        Vhodna slika 
    iTitle : str 
        Naslov za sliko
    
    Returns
    ---------
    Nothing
    
    
    '''
    plt.figure() # odpri novo prikazno okno
    
    if iImage.ndim == 3 and iImage.shape[0] == 3:
        iImage = np.transpose(iImage,[1,2,0])

    plt.imshow(iImage, cmap = 'gray') # prikazi sliko v novem oknu
    plt.suptitle(iTitle) # nastavi naslov slike
    plt.xlabel('x')
    plt.ylabel('y')

def genImgsIntoArray(path_with_name,filetype,N):
    '''
    Generator za slike z diska  
    '''
    for znj in range(1,N+1):
        p="{}{:04}.{}".format(path_with_name,znj,filetype)
        slika = Image.open(p).convert('L') #sivinska slika
        yield np.array(slika,dtype=np.float32)


def quiverOnImage(u, v, iImage, scale=3, step=5,showOrSave='show',path=''):
    """
    na novo napisan quiver plot, da lahko rišem bolj na redko
    """
    u=-u #
    v=-v #
    ax = plt.figure().gca()
    ax.imshow(iImage, cmap='gray', origin='upper')
    for i in range(0, u.shape[0], step):
        for j in range(0, v.shape[1], step):
            ax.arrow(j,i, u[i, j]*scale, v[i, j]*scale, color='red',
                     head_width=1.5, head_length=1) #navadni quiverplot mi ni sluzil dobro, zato sem ga spisal na roke
    plt.draw()
    if showOrSave=='show':   
        plt.show()
    elif showOrSave=='save':
        plt.savefig(path+'.png')

def convertHSV2RGB( iImage):
    #pretvorba barvne slike iz barvnega prostora HSV v RGB
    iImage = iImage.astype('float') #vrača uint8 sliko
    h = iImage[:,:,0]
    s = iImage[:,:,1]
    v = iImage[:,:,2]
    
    C = v * s
    X = C * (1.0 - np.abs( ( (h/60.0) % 2.0 ) -1 ) )
    m = v - C
    
    r = np.zeros_like( h )
    g = np.zeros_like( h )
    b = np.zeros_like( h )
    
    r[ (h>=0.0) * (h<60.0) ] = C[ (h>=0.0) * (h<60.0) ]
    g[ (h>=0.0) * (h<60.0) ] = X[ (h>=0.0) * (h<60.0) ]
    
    r[ (h>=60.0) * (h<120.0) ] = X[ (h>=60.0) * (h<120.0) ]
    g[ (h>=60.0) * (h<120.0) ] = C[ (h>=60.0) * (h<120.0) ]
    
    g[ (h>=120.0) * (h<180.0) ] = C[ (h>=120.0) * (h<180.0) ]
    b[ (h>=120.0) * (h<180.0) ] = X[ (h>=120.0) * (h<180.0) ]
    
    g[ (h>=180.0) * (h<240.0) ] = X[ (h>=180.0) * (h<240.0) ]
    b[ (h>=180.0) * (h<240.0) ] = C[ (h>=180.0) * (h<240.0) ]

    r[ (h>=240.0) * (h<300.0) ] = X[ (h>=240.0) * (h<300.0) ]
    b[ (h>=240.0) * (h<300.0) ] = C[ (h>=240.0) * (h<300.0) ]   
    
    r[ (h>=300.0) * (h<360.0) ] = C[ (h>=300.0) * (h<360.0) ]
    b[ (h>=300.0) * (h<360.0) ] = X[ (h>=300.0) * (h<360.0) ]  
    
    r = r + m
    g = g + m
    b = b + m
    
    #ustvari izhodno sliko
    oImage = np.zeros_like( iImage )
    oImage[:,:,0] = r
    oImage[:,:,1] = g
    oImage[:,:,2] = b
    
    
    #zaokrozevanje vrednosti
    oImage = 255.0 * oImage
    oImage[oImage>255.0]= 255.0
    oImage[oImage<0.0] = 0.0
    
    oImage = oImage.astype('uint8')

    
    return oImage

''' 
def scaleImage(iImage, iSlopeA, iIntersectionB):
    iImageType = iImage.dtype
    iImage = iImage.astype('float')
    oImage = iSlopeA*iImage + iIntersectionB
    if iImageType.kind in ('u' , 'i'):
        oImage[oImage<np.iinfo(iImageType).min]=np.iinfo(iImageType).min
        oImage[oImage>np.iinfo(iImageType).max]=np.iinfo(iImageType).max  ##omejiti moramo podatke na omejitve input tipa
    return np.array(oImage, dtype = iImageType)

def windowImage( iImage, iCenter, iWidth ):
    iImageType = iImage.dtype
    if iImageType.kind in ('u','i'):
        iMaxValue= np.iinfo(iImageType).max
        iMinValue= np.iinfo(iImageType).min
        
    else:
        iMaxValue = np.max(iImage)
        iMinValue = np.min(iImage)
    iRange = iMaxValue - iMinValue
    
    iSlopeA = iRange/float(iWidth)
    iInterceptB = -iSlopeA * (float(iCenter) - iWidth/2)
    
    
    return scaleImage(iImage, iSlopeA, iInterceptB)

def window_middle80_parameters(image):
    # YOUR CODE HERE
    sortirana=np.array(sorted(image.flatten()))
    #print(len(sortirana))
    ind_0_1=int(len(sortirana)*0.1)-1
    val_0_1=sortirana[ind_0_1]
    ind_0_9=int(len(sortirana)*0.9)-1
    val_0_9=sortirana[ind_0_9]
    
    w=val_0_9-val_0_1
    c=(val_0_1+val_0_9)/2
       
    return (c, w)
'''

import cv2
def optFlowColorVisualisation(u,v,iImage,showOrSave='show',path=''):
    '''
    Vizualizacja optičnega toka z barvami.
    Polarni kot in dolžina vektroja optičnega toka (u,v) določata h in s kanal hsv

    '''
    u=-u #obrnem optični tok, ker rišem iz slike 1 v dva, tok pa računam iz 2 v 1
    v=-v #
    #u, v v kot in dolz
    iImage=np.array(iImage)
    kot=(np.arctan2(-v,u)/np.pi*180) %360
    dolz=np.sqrt(np.square(u)+np.square(v))
    h = kot
    kot[np.isnan( kot )] = 0
    #s = np.divide( (dolz-np.amin(dolz)),(np.amax(dolz)-np.amin(dolz)))
    #s[np.isnan( s )] = 0
    #s = np.ones(iImage.shape)
    s = cv2.normalize(dolz, None, 0, 1, cv2.NORM_MINMAX)
    v = np.ones(iImage.shape) #value naj bo vedno 1
    iRGB=convertHSV2RGB(np.dstack((h,s,v)))
    if showOrSave=='show':
        plt.figure()
        plt.imshow(iRGB)
        plt.show()
    elif showOrSave=='save':
        plt.figure()
        plt.imshow(iRGB)
        plt.savefig(path+'.png')
    #    saveImage(path,np.array(iRGB),'png')
