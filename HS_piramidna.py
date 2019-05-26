from scipy.ndimage.filters import convolve
import numpy as np
from interpolacija import interpn
from funkcije import showImage
import cv2
import scipy.ndimage as ni
#predlagan s strani Horn- Schunck, redefiniram (uposteva Neumanove robne pogoje sistema diff enačb):
kernelAvg = np.array([[1/12, 1/6, 1/12],
                        [1/6,    0, 1/6],
                        [1/12, 1/6, 1/12]], dtype=np.float32)

def discreteGaussian2D(iSigma):
    '''
    izračun gaussovega jedra za glajenje iz sigme
    '''
    U = int(2*np.ceil(3*iSigma) +1)
    V = U
    oKernel = np.zeros((U,V))
    for u in list(range(int(-(U-1)/2), int((U-1)/2)+1, 1)):
        for v in list(range(int(-(V-1)/2), int((V-1)/2)+1, 1)):
            oKernel[u+int((U-1)/2),v+int((V-1)/2)] = ((2*np.pi)**(-1)*iSigma**(-2))*np.exp(-(u**2+v**2)/2/iSigma**2)

    return oKernel/np.sum(oKernel)



def bicubicInterpolateWarp(I2,U,V):
    '''
    popravi za u,v in bikubično interpoliraj kar manjka
    '''
    dy,dx=I2.shape
    x = np.arange(0, dx, 1)
    y = np.arange(0, dy, 1)
    #Y, X = np.meshgrid(y, x, indexing='ij')  #tocke prej
    f = I2
    xi = np.arange(0, dx, 1.0)
    yi = np.arange(0, dy, 1.0)
    Yi, Xi = np.meshgrid(yi, xi, indexing='ij')#tocke potem
    Xi -= U
    Yi -= V
    fi = interpn([Yi, Xi], [y, x], f,order=3) #3. red
    return fi

def upscaleDownscaleInterp(I,nj):
    '''
    I: image to scale, nj>1 upscale, nj<1 downscale
    '''

    dy,dx=I.shape
    x = np.arange(0, dx, 1)
    y = np.arange(0, dy, 1)
    #Y, X = np.meshgrid(y, x, indexing='ij')  #tocke prej
    f = I
    xi = np.arange(0, dx, 1/nj)
    yi = np.arange(0, dy, 1/nj)
    Yi, Xi = np.meshgrid(yi, xi, indexing='ij')#tocke potem
    fi = interpn([Yi, Xi], [y, x], f,order=3) #3. red
    return fi

def imageGradient( iImage ):
    """Gradient slike s centralnimi diferencami"""
    iImage = np.array( iImage, dtype='float' )    
    iSobel = np.array( ((0,0,0),(-1,0,1),(0,0,0)) )     #ni dejansko Sobel, je samo centralna diferenca.
    oGx = ni.convolve( iImage, iSobel, mode='constant' )
    oGy = ni.convolve( iImage, np.transpose( iSobel ), mode='constant' )
    return oGx, oGy


def HSOF(I1,I2,U,V,alpha,eps,Nmaxiter,Nwarps):
    #izračun HS na enem nivoju

    #izracun odvodov
    I2x,I2y=imageGradient(I2)

    for n in range(1,Nwarps+1):
        #popravi I2: I2(x+h), I2x(x+h), I2y(x+h)
        I2w=bicubicInterpolateWarp(I2,U,V)
        I2xw=bicubicInterpolateWarp(I2x,U,V)
        I2yw=bicubicInterpolateWarp(I2y,U,V)

        Un=U
        Vn=V
        r=0
        stopCrit=1
        while (r<Nmaxiter) and stopCrit>eps:
            #zapomnisi za izračun kriterija za ustavitev
            Uk=np.copy(U)
            Vk=np.copy(V)
            #izračunaj povprečje
            Au=convolve(U,kernelAvg,mode='nearest')
            Av=convolve(V,kernelAvg,mode='nearest')
            #gauss-seidlova metoda za iterativno numerično reševanje
            Utmp=((I1-I2w+I2xw*Un-I2yw*(V-Vn))*I2xw+alpha**2*Au)/(np.square(I2xw)+alpha**2)
            Vtmp=((I1-I2w-I2xw*(U-Un)+I2yw*Vn)*I2yw+alpha**2*Av)/(np.square(I2yw)+alpha**2)
            U=Utmp
            V=Vtmp
            #kriterij za ustavitev: se v iteracijah sploš še kaj spreminja ?
            stopCrit=1/np.size(I2)*np.sum( np.square((U-Uk))+ np.square((V-Vk)))
            r=r+1
    return U,V


def HornSchunckPiramidna(I1,I2,alpha,eps,nj,Nmaxiter,Nwarps,Nscales):
    '''
    Ocena optičnega toka po metodi Horn-Schunck z uporabo piramidne sheme

    '''


    #normalize I1, I2 between 0 and 255 (ni treba, je že taka vhodna slika)

    #convolve with gaussion of sigma=0.8 #lahko bi pred vsem skupaj še enkrat zgladil, pa to naredim že kasneje
    #kernelGauss08=discreteGaussian2D(0.8)
    #I1=convolve(I1,kernelGauss08,mode='nearest')
    #I2=convolve(I2,kernelGauss08,mode='nearest')

    I1s=[I1]
    I2s=[I2]
    Us=[np.zeros(I1.shape,dtype=np.float32)]
    Vs=[np.zeros(I1.shape,dtype=np.float32)]
    sigma0=0.6
    sigmanj=sigma0*np.sqrt(nj**(-2)-1)
    kernelGauss06nj=discreteGaussian2D(sigmanj)

    #izračunaj nivoje s pomočjo decimacije
    for s in range(1,Nscales):
        #glajenje pred decimacijo
        I1tmp=convolve(I1s[s-1],kernelGauss06nj)
        I2tmp=convolve(I2s[s-1],kernelGauss06nj)
        #downscale imgs
        I1d=upscaleDownscaleInterp(I1tmp,nj)
        I2d=upscaleDownscaleInterp(I2tmp,nj)

        I1s.append(I1d)
        I2s.append(I2d)

        #začetni približek optičnega toka je 0
        Us.append(np.zeros(I1d.shape,dtype=np.float32))
        Vs.append(np.zeros(I1d.shape,dtype=np.float32))


    for s in range(Nscales-1,-1,-1):
        U,V=HSOF(I1s[s],I2s[s],Us[s],Vs[s],alpha,eps,Nmaxiter,Nwarps)
        if s!=0:
            #upsample U,V  
            U=cv2.resize(U, dsize=(I1s[s-1].shape[1],I1s[s-1].shape[0]), interpolation=cv2.INTER_CUBIC) #pazi, dsize je kot x,y!!!
            V=cv2.resize(V, dsize=(I1s[s-1].shape[1],I1s[s-1].shape[0]), interpolation=cv2.INTER_CUBIC) #pazi, dsize je kot x,y!!!

            #primerno preskaliraj še u in v, da imata tudi na nižjemnivoju smisel
            Us[s-1]=1/nj*U
            Vs[s-1]=1/nj*V
    return U,V

if (__name__=='__main__'):
    from PIL import Image
    i1 = np.array(Image.open('C:/RV/KoncniProjekt/koncniRV/Frames/a.png').convert('L'), dtype=np.float64) #sivinska slika
    i2 = np.array(Image.open('C:/RV/KoncniProjekt/koncniRV/Frames/b.png').convert('L'), dtype=np.float64) #sivinska slikaž
    showImage(i1)
    u,v=HornSchunckPiramidna(i1,i2,7,0.001,0.5,100,10,5)
    print(u,v)
    showImage(bicubicInterpolateWarp(i1,u,v))
    from plots_and_reads import quiverOnImage
    quiverOnImage(u,v,i1,scale=1,step=10)
