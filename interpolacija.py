from scipy.ndimage import map_coordinates
import numpy as np

#vir: kolegi iz biomedicinske tehnike so imeli to priloženo na vajah

def interpn(ti, t, f, order=1, **kwargs):
    '''
    Interpolacija ND funkcij definiranih na enakomerni mreži točk.

    Parametri
    ---------
    ti : podatkovno polje
        Seznam komponent točk v katerih je potrebno izračunati
        (interpolirati) funkcijske vrednosti.
    t : podatkovno polje
        Seznam vektorjev enakomerno razporejenih vzorčnni točk vzdolž vseh
        koordinatnih osi (razsežnosti).
    f : podatkovno polje
        Funkcijske vrednosti na ND mreži vzorčnih točkah, ki jo napenjajo
        vektorji v t. Velikost podatkovnega polja f mora
        ustrezati (t[0].size, t[1].size, ..., t[-1].size).
    order : int, neobvezno
        Red interpolacije. 0 - najbližji sosed, 1 - linearna ...
    kwargs : razno, neobvezno
        Preostali poimensko podani parametri se posredujejo funkciji
        map_coordinates.

    Vrne
    ----
    fti : podatkovno polje
        Funkcijske vrednosti v točkah f[t[0], t[1], ..., t[-1]].

    Primer
    ------
    >>> import numpy as np
    >>> from matplotlib import pyplot as pp
    >>> from mpl_toolkits.mplot3d import Axes3D
    >>>
    >>> x = np.linspace(-1, 1, 15)
    >>> y = np.linspace(-1, 1, 10)
    >>> Y, X = np.meshgrid(y, x, indexing='ij')
    >>> f = 1.0/(X**2 + Y**2 + 1)
    >>>
    >>> xi = np.linspace(0, 1, 30)
    >>> yi = np.linspace(0, 1, 30)
    >>> Yi, Xi = np.meshgrid(yi, xi, indexing='ij')
    >>> fi = interpn([Yi, Xi], [y, x], f)
    >>>
    >>> fig = pp.figure()
    >>> ax = fig.add_subplot(111, projection='3d')
    >>> ax.plot_wireframe(X, Y, f, color='r', label='vzorčne točke')
    >>> ax.plot_wireframe(Xi, Yi, fi, color='g', label='interpolirane vrednosti')
    >>> ax.legend()
    '''

    f = np.asarray(f)

    if t is not None:
        N = len(t) # space dimensionality
        if N != len(ti):
            raise IndexError('Dimensions of the coordinates must agree!')

        tind = []
        for i in range(N):
            if t[i] is not None:
                tmp = t[i].flatten()
                tind.append((ti[i] - tmp[0])*
                            ((tmp.size - 1)/(tmp[-1] - tmp[0])))
            else:
                tind.append(ti[i])
    else:
        tind = ti
    return map_coordinates(f, np.asarray(tind), order=order, **kwargs)