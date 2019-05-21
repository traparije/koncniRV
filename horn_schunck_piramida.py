import numpy as np
import scipy.ndimage as ni
from scipy.ndimage.filters import convolve
import scipy.interpolate as si
import cv2
from funkcije import showImage
def warp_flow(img, u,v):
    flow=np.array(np.dstack((u,v)),dtype=np.float32)
    h, w = flow.shape[:2]
    flow = np.multiply(-1,flow)
    flow[:,:,0] += np.arange(w)
    flow[:,:,1] += np.arange(h)[:,np.newaxis]
    res = cv2.remap(img, flow, None, cv2.INTER_CUBIC)
    return res


def discreteGaussian2D(iSigma):
    U = int(2*np.ceil(3*iSigma) +1)
    V = U
    oKernel = np.zeros((U,V))
    for u in list(range(int(-(U-1)/2), int((U-1)/2)+1, 1)):
        for v in list(range(int(-(V-1)/2), int((V-1)/2)+1, 1)):
            oKernel[u+int((U-1)/2),v+int((V-1)/2)] = ((2*np.pi)**(-1)*iSigma**(-2))*np.exp(-(u**2+v**2)/2/iSigma**2)

    return oKernel/np.sum(oKernel)

#
def bicubicInterpolateGrayImage( iImage, iCoorX, iCoorY, method,fill ): #fail
    dy, dx = iImage.shape
    return si.interpn((np.arange(dy), np.arange(dx)), 
                          iImage,
                          (iCoorY,iCoorX),
                          method=method,
                          bounds_error=False,fill_value=fill).astype('uint8')

def bicubicInterpolateColorImage( iImage, iCoorX, iCoorY, method ): #fail
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

def SORiteration1(Au,Av,D,Du,Dv,U,V,alpha,w=1.9):
    #parameter SOR
    uAvg=convolve(U,kernelAvg)
    vAvg=convolve(V,kernelAvg)
    uOld=U
    vOld=V
    #posodobiTok
    U=(1-w)*uOld + w*np.divide((Au - np.multiply(D,V) + alpha*uAvg),Du)
    V=(1-w)*vOld + w*np.divide((Av - np.multiply(D,U) + alpha*vAvg),Dv)

    #napaka konvergence
    error=np.square((U-uOld))+np.square((V-vOld))
    return (error,U,V)



def HSOF1(I1,I2,U,V,nx,ny,alpha,Nwarps,eps,maxiter): #na eni skali
        size=nx*ny
        I2x,I2y=imageGradient(I2)
        #iterativna aproksimacija (taylor)
        print('start')
        for n in range(Nwarps):
                print('start')
                #warp (mogoče narobe računam!)
                I2w=bicubicInterpolateGrayImage(I2,np.arange(0,nx-1),np.arange(0,ny-1),'linear',0)#popravi!!! mora biti bikubična
                I2wx=bicubicInterpolateGrayImage(I2x,np.arange(0,nx-1),np.arange(0,ny-1),'linear',0)
                I2wy=bicubicInterpolateGrayImage(I2y,np.arange(0,nx-1),np.arange(0,ny-1),'linear',0)

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
                        error,U,V=SORiteration1(Au,Av,D,Du,Dv,U,V,alpha**2)
                        error=np.sqrt(error/nx*ny)
                
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
        I1s[0]=convolve(I1s[0],gaussKernel)
        I2s[0]=convolve(I2s[0],gaussKernel)
        showImage(I1s[0])

        #inicializiram slike različnih skal
        for s in range(1,nScales):
                #downsampling

                #novi shape
                prejdy,prejdx=I1s[s-1].shape
                novdy=int(prejdy*nj+0.5)
                novdx=int(prejdx*nj+0.5)
                #iz opencv si sposodim resize za hitrost
                I1s[s]=cv2.resize(I1s[s-1], dsize=(novdx,novdy), interpolation=cv2.INTER_CUBIC) #pazi, dsize je kot x,y!!!
                I2s[s]=cv2.resize(I2s[s-1], dsize=(novdx,novdy), interpolation=cv2.INTER_CUBIC)
                #  testing showImage(I1s[s]), zdaj dela
                #po resizu še zgladim s sigma 0.6 gaussovim jedrom
                I1s[s]=convolve(I1s[s],gaussKernel)
                I2s[s]=convolve(I2s[s],gaussKernel)

        #inicializacija U in V
        U=np.zeros(I1s[nScales-1].shape)
        Us[nScales-1]=U
        V=np.zeros(I1s[nScales-1].shape)
        Vs[nScales-1]=V
        #piramidna aproksimacija optičenga toka po Horn-Schuncku:
        for s in range(nScales-1,-1,-1):
                e,Utemp,Vtemp=HSOF(I1s[s],I2s[s],Us[s],Vs[s],I1s[s].shape[1],I1s[s].shape[0],alpha,nWarps,eps,maxiter)
                #print("s in  Utemp",s, Utemp)
                if s==0:  #za zadnji scale še poračunam potem pa ne več
                        break
                #else: 
                #upsample U in V
                Us[s-1]=cv2.resize(Utemp, dsize=(I1s[s-1].shape[1],I1s[s-1].shape[0]), interpolation=cv2.INTER_CUBIC)
                Vs[s-1]=cv2.resize(Vtemp, dsize=(I1s[s-1].shape[1],I1s[s-1].shape[0]), interpolation=cv2.INTER_CUBIC)
                
                Us[s-1]/=nj#skaliram optični tok s faktorjem povečave
                Vs[s-1]/=nj
        return Us[0],Vs[0]


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
    Igladka=convolve(I0,gaussKernel)
    #bikubična interpolacija
    bicubicInterp(Igladka)

    #default vrednost parametra pri SOR relaksaciji
    w=1.9
'''
def inefficientWarpNoInterp(I,U,V):

        Iout=np.array(I)
        for y in range(I.shape[0]):
                for x in range(I.shape[1]):
                        Iout[y][x]=I[max(0,int(round(y+U[y][x])))][max(0,int(round(x+V[y][x])))]
        return Iout

def HSOF(I1,I2,U,V,nx,ny,alpha,Nwarps,eps,maxiter,w=1.9): #na eni skali
        I2=np.copy(I2)
        I2x,I2y=imageGradient(I2)
        #iterativna aproksimacija (taylor)
        print("iteriram na skali",I2.shape)
        Un=np.copy(U)
        Vn=np.copy(V)
        for n in range(Nwarps):
                e=0
                print("wraping",n, "")
                #warp (mogoče narobe računam!)
                I2w=warp_flow(I2,U,V)
                I2wx=warp_flow(I2,U,V)
                I2wy=warp_flow(I2w,U,V)
                #I2w=transformImage(I2,Un,Vn)
                #I2wx=transformImage(I2x,Un,Vn)
                #I2wy=transformImage(I2y,Un,Vn)
                
                #SOR iteracije
                r=0
                error=1000
                Unr=np.copy(Un)
                Vnr=np.copy(Vn)
                while(error>eps and  r<maxiter): #SOR  (Successive Over-Relaxation) ITERACIJA
                        UnrOld=np.copy(Unr)
                        VnrOld=np.copy(Vnr)
                        r+=1
                        Aunr=convolve(Unr,kernelAvg)
                        #print("Anur",Aunr)
                        Avnr=convolve(Vnr,kernelAvg)
                        Unr=(1-w)*Unr+w*(np.multiply((I1-I2w+np.multiply(I2wx,Un)-np.multiply(I2wy,(Vnr-Vn))),I2wx)+alpha**2*Aunr)/(np.square(I2wx)+alpha**2)
                        Vnr=(1-w)*Vnr+w*(np.multiply((I1-I2w-np.multiply(I2wx,(Unr-Un))+np.multiply(I2wy,Vn)),I2wy)+alpha**2*Avnr)/(np.square(I2wy)+alpha**2)
                        #napaka konvergence
                        error=np.square((Unr-UnrOld))+np.square((Vnr-VnrOld))
                        error=np.sum(np.sqrt(error/nx*ny))
                Un=Unr
                Vn=Vnr
                e+=error
        U=Un
        V=Vn
        return e,U,V


def interpolate1Image2D( iImage, iCoorX, iCoorY ):
    """Funkcija za interpolacijo prvega reda"""
    # pretvori vhodne spremenljivke v np polje
    iImage = np.asarray( iImage )    
    iCoorX = np.asarray( iCoorX )
    iCoorY = np.asarray( iCoorY )   
    # preberi velikost slike in jedra
    dy, dx = iImage.shape
    # ustvari 2d polje koordinat iz 1d vhodnih koordinat (!!!)
    if np.size(iCoorX) != np.size(iCoorY):
        print('Stevilo X in Y koordinat je razlicno!')      
        iCoorX, iCoorY = np.meshgrid(iCoorX, iCoorY, sparse=False, indexing='xy')
    #------------------------------- za hitrost delovanja    
    return si.interpn( (np.arange(dy),np.arange(dx)), iImage, \
                      np.dstack((iCoorY,iCoorX)),\
                      method='linear', bounds_error=False)\
                      .astype( iImage.dtype )    
               
def transAffine2D( iScale=(1, 1), iTrans=(0, 0), iRot=0, iShear=(0, 0) ):
    """Funkcija za poljubno 2D afino preslikavo"""    
    iRot = iRot * np.pi / 180
    oMatScale = np.matrix( ((iScale[0],0,0),(0,iScale[1],0),(0,0,1)) )
    oMatTrans = np.matrix( ((1,0,iTrans[0]),(0,1,iTrans[1]),(0,0,1)) )
    oMatRot = np.matrix( ((np.cos(iRot),-np.sin(iRot),0),\
                          (np.sin(iRot),np.cos(iRot),0),(0,0,1)) )
    oMatShear = np.matrix( ((1,iShear[0],0),(iShear[1],1,0),(0,0,1)) )
    # ustvari izhodno matriko
    oMat2D = oMatTrans * oMatShear * oMatRot * oMatScale
    return oMat2D               
               
def transformImage( iImage, U,V):
    """Preslikaj 2D sliko z linearno preslikavo"""
    # ustvari diskretno mrezo tock
    gx, gy = np.meshgrid( range(iImage.shape[1]), \
                          range(iImage.shape[0]), \
                          indexing = 'xy' )  



    # ustvari Nx3 matriko vzorcnih tock                          
    pts = np.vstack( (gx.flatten(), gy.flatten(), np.ones( (gx.size,))) ).transpose()
    #print(pts)
    pts1=np.copy(pts)
    for i in range(len(pts)):
        pts[i][0]+=U[int(pts1[i][0])][int(pts1[i][1])]
        pts[i][1]+=V[int(pts1[i][0])][int(pts1[i][1])]
    # preslikaj vzorcne tocke
    # ustvari novo sliko z interpolacijo sivinskih vrednosti
    oImage = interpolate1Image2D( iImage, \
                                  pts[:,0].reshape( gx.shape ), \
                                  pts[:,1].reshape( gx.shape ) )
    oImage[np.isnan( oImage )] = 0
    return oImage 
