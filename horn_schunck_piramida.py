import numpy as np
import scipy.ndimage as ni
from scipy.ndimage.filters import convolve
import scipy.interpolate as si

def discreteGaussian2D(iSigma):
    U = int(2*np.ceil(3*iSigma) +1)
    V = U
    oKernel = np.zeros((U,V))
    for u in list(range(int(-(U-1)/2), int((U-1)/2)+1, 1)):
        for v in list(range(int(-(V-1)/2), int((V-1)/2)+1, 1)):
            oKernel[u+int((U-1)/2),v+int((V-1)/2)] = ((2*np.pi)**(-1)*iSigma**(-2))*np.exp(-(u**2+v**2)/2/iSigma**2)

    return oKernel/np.sum(oKernel)

#
def bicubicInterpolateGrayImage( iImage, iCoorX, iCoorY, method,fill ):
    dy, dx = iImage.shape
    return si.interpn((np.arange(dy), np.arange(dx)), 
                          iImage,
                          (iCoorY,iCoorX),
                          method=method,
                          bounds_error=False,fill_value=fill).astype('uint8')

def bicubicInterpolateColorImage( iImage, iCoorX, iCoorY, method ):
    dy, dx, dz = iImage.shape
    return si.interpn((np.arange(dy), np.arange(dx)), 
                          iImage,
                          (iCoorY,iCoorX),
                          method=method,
                          bounds_error=False).astype('uint8')


def imageGradient( iImage ):
    """Gradient slike s Sobelovim operatorjem"""
    iImage = np.array( iImage, dtype='float' )    
    iSobel = np.array( ((-1,0,1),(-2,0,2),(-1,0,1)) )    
    oGx = ni.convolve( iImage, iSobel, mode='nearest' )
    oGy = ni.convolve( iImage, np.transpose( iSobel ), mode='nearest' )
    return oGx, oGy





kernelAvg = np.array([[1/12, 1/6, 1/12],
                        [1/6,    0, 1/6],
                        [1/12, 1/6, 1/12]], dtype=np.float32)
def SORiteration(Au,Av,D,Du,Dv,U,V,alpha,w=1.9):
    #parameter SOR
    uAvg=convolve(kernelAvg,U)
    vAvg=convolve(kernelAvg,V)
    uOld=U
    vOld=V
    #posodobiTok
    U=(1-w)*uOld + w*np.divide((Au - np.multiply(D,V) + alpha*uAvg),Du)
    V=(1-w)*vOld + w*np.divide((Av - np.multiply(D,U) + alpha*vAvg),Dv)

    #napaka konvergence
    error=np.square((U-uOld))+np.square((V-vOld))
    return (error,U,V)



def HSOF(I1,I2,U,V,nx,ny,alpha,Nwarps,eps,maxiter): #na eni skali
    size=nx*ny
    I2x,I2y=imageGradient(I2)
    #iterativna aproksimacija (taylor)
    for n in range(Nwarps):

        #warp (mogoče narobe računam!)
        I2w=bicubicInterpolateGrayImage(I2,np.arange(0,nx-1),np.arange(0,ny-1),'cubic',0)
        I2wx=bicubicInterpolateGrayImage(I2x,np.arange(0,nx-1),np.arange(0,ny-1),'cubic',0)
        I2wy=bicubicInterpolateGrayImage(I2y,np.arange(0,nx-1),np.arange(0,ny-1),'cubic',0)
        
        I2wl=np.multiply(I2wx,U) + np.multiply(I2wy,V)
        dif=I1-I2w+I2wl
        Au=np.multiply(dif,I2wx)
        Av=np.multiply(dif,I2wy)
        Du=np.square(I2wx)+alpha**2
        Dv=np.square(I2wy)+alpha**2
        D=np.multiply(I2wx+I2wy)

        #SOR iteracije
        niter=0
        error=1000
        while(error>eps and niter <maxiter):

            niter+=1
            error,U,V=SORiteration(Au,Av,D,Du,Dv,U,V,alpha**2)
            error=np.sqrt(error,nx*ny)


def normalize_images(I1,I2):
    u1=np.amax(I1)
    l1=np.amin(I1)
    u2=np.amax(I2)
    l2=np.amin(I2)
    uabs=max(u1,u2)
    labs=min(l1,l2)
    den=uabs-labs
    if not(den>0):
        return I1,I2
    else:
        I1=255*(I1-labs*np.ones(I1.shape))/den
        I2=255*(I2-labs*np.ones(I2.shape))/den
        return I1, I2

#mozno je razpoznati zamake na (1/nj)**(Nscale-1) pikslih. Pri defaultnih 0.5 nj in 5 Nscale, nam to da 16 px maneverskega prostora
#NScales= -log(max_motion)/log(nj)+1
def f(I0,I1,alpha=15,eps=0.0001,nj=0.5, Nscales=5, Nwraps=5):
    '''nj med 0 in 1. 1 pomeni da ni downsamplinga'''
    sigma0=0.6
    sigma = sigma0*np.sqrt(nj**(-2)-1)
    gaussKernel = discreteGaussian2D(sigma)

    #zgladimo tako gosto kot je treba glede na nj (ta je sorazmeren stopnji piramidne decimacije)
    Igladka=convolve(gaussKernel,I0)
    #bikubična interpolacija
    bicubicInterp(Igladka)

    #default vrednost parametra pri SOR relaksaciji
    w=1.9