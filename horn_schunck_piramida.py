import numpy as np
import scipy.ndimage as ni
from scipy.ndimage.filters import convolve
import scipy.interpolate as si
import cv2

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
        
        return error,U,V


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

def HSpiramida(I1,I2,alpha=15,eps=0.0001,nj=0.5, nScales=5, nWarps=5,maxiter=1000):
        #alpha=utež pogoja gladkosti
        ny,nx=I1.shape

        #normaliziram 
        I1s=[0 for i in range(nScales)]#ustvarim pravilno velik prazen seznam
        I2s=[0 for i in range(nScales)]#ustvarim pravilno velik prazen seznam
        Us=[0 for i in range(nScales)]#ustvarim pravilno velik prazen seznam
        Vs=[0 for i in range(nScales)]#ustvarim pravilno velik prazen seznam
        I1s[0],I2s[0]=normalize_images(I1,I2)
        #zgladim
        sigma0=0.6
        sigma = sigma0*np.sqrt(nj**(-2)-1)
        gaussKernel = discreteGaussian2D(sigma)
        I1s[0]=convolve(gaussKernel,I1s[0])
        I2s[0]=convolve(gaussKernel,I2s[0])




        #inicializiram slike različnih skal
        for s in range(1,nScales):
                #downsampling

                #novi shape
                prejdy,prejdx=I1s[s-1].shape
                novdy=int(prejdy*nj+0.5)
                novdx=int(prejdx*nj+0.5)

                #iz opencv si sposodim resize za hitrost
                I1s[s]=cv2.resize(I1s[s-1], dsize=(novdy,novdx), interpolation=cv2.INTER_CUBIC)
                I2s[s]=cv2.resize(I2s[s-1], dsize=(novdy,novdx), interpolation=cv2.INTER_CUBIC)
                #po resizu še zgladim s sigma 0.6 gaussovim jedrom
                I1s[s]=convolve(gaussKernel,I1s[s])
                I2s[s]=convolve(gaussKernel,I2s[s])

        #inicializacija U in V
        U=np.zeros(I1s[nScales-1].shape)
        Us[nScales-1]=U
        V=np.zeros(I1s[nScales-1].shape)
        Vs[nScales-1]=V

        #piramidna aproksimacija optičenga toka po Horn-Schuncku:
        for s in range(nScales-1,-1,1):
                e,Utemp,Vtemp=HSOF(I1s[s],I2s[s],Us[s],Vs[s],I1s[s].shape[1],I1s[s].shape[0],alpha,nWarps,eps,maxiter)

                if s==0:  #za zadnji scale še poračunam potem pa ne več
                        break
                #else: 
                #upsample U in V
                Us[s-1]=cv2.resize(Utemp, dsize=I1s[s-1].shape, interpolation=cv2.INTER_CUBIC)
                Vs[s-1]=cv2.resize(Vtemp, dsize=I1s[s-1].shape, interpolation=cv2.INTER_CUBIC)
                Us[s-1]/=nj#skaliram optični tok s faktorjem povečave
                Vs[s-1]/=nj

        return U[0],V[0]


'''
if __name__=="__main__":
        HSpiramida()
'''



'''
#mozno je razpoznati zamake na (1/nj)**(Nscale-1) pikslih. Pri defaultnih 0.5 nj in 5 Nscale, nam to da 16 px maneverskega prostora
#NScales= -log(max_motion)/log(nj)+1
def f(I0,I1,alpha=15,eps=0.0001,nj=0.5, Nscales=5, Nwraps=5):
    #nj med 0 in 1. 1 pomeni da ni downsamplinga
    sigma0=0.6
    sigma = sigma0*np.sqrt(nj**(-2)-1)
    gaussKernel = discreteGaussian2D(sigma)

    #zgladimo tako gosto kot je treba glede na nj (ta je sorazmeren stopnji piramidne decimacije)
    Igladka=convolve(gaussKernel,I0)
    #bikubična interpolacija
    bicubicInterp(Igladka)

    #default vrednost parametra pri SOR relaksaciji
    w=1.9
'''
