from PIL import Image
import re
import numpy
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image as im

def convertToGray(image):
    dtype = image.dtype
    rgb = np.array(image, dtype='float')
    gray = rgb[:, :, 0]*0.299 + rgb[:, :, 1]*0.587 + rgb[:, :, 2]*0.114
    return gray.astype(dtype)

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
    for znj in range(N):
        p="{}{}.{}".format(path_with_name,znj,filetype)
        slika = Image.open(p).convert('L') #sivinska slika
        yield np.array(slika,dtype=np.float32)

def quiverOnImage(u, v, iImage, scale=3, step=5, iTitle=None):
    """
    makes quiver
    """
    ax = plt.figure()
    ax.imshow(iImage, cmap='gray', origin='lower')
    for i in range(0, u.shape[0], step):
        for j in range(0, v.shape[1], step):
            ax.arrow(j, i, v[i, j]*scale, u[i, j]*scale, color='red',
                     head_width=0.5, head_length=1) #navadni quiverplot mi ni sluzil dobro, zato sem ga spisal na roke
    if iTitle:
        ax.set_title(iTitle)
    plt.draw()