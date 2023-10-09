import numpy as np

tiny = 1e-99

def bnu_wav_micron(wav_um,temp):
    """Return a Planck function, avoiding overflows.
    
    Parameters
    ----------
    wave_um : ndarray of float
        Wavelengths at which to compute flux.
    temp : float
        Temperature of blackbody.
    """
    k1 = 3.9728949e19
    k2 = 14387.69
    fact1 = k1/(wav_um**3)
    fact2 = k2/(wav_um*temp)
    if isinstance(wav_um,np.ndarray):
        ofl = fact2 < 709
        bnu = np.zeros(len(fact2)) + tiny
        if np.any(ofl) == False:
            return bnu
        else:
            bnu[ofl] = fact1[ofl]/(np.exp(fact2[ofl])-1.0)
            return bnu
    elif isinstance(temp,np.ndarray):
        ofl = fact2 < 709
        bnu = np.zeros(len(fact2)) + tiny
        if np.any(ofl) == False:
            return bnu
        else:
            bnu[ofl] = fact1/(np.exp(fact2[ofl])-1.0)
            return bnu
    else:
        if fact2 > 709:
            return tiny
        else:
            return fact1/(np.exp(fact2)-1.0)


def col2temp(wav_um, flux):
    '''Convert two fluxes at different wavelengths to a blackbody temp.
    
    Parameters
    ----------
    wav_um: length 2 array or tuple
        Wavelengths of two fluxes
    flux: length 2 array or tuple
        Fluxes in same units
    '''
    
    if wav_um[0] > wav_um[1]:
        wav = [wav_um[1],wav_um[0]]
        flx = [flux[1], flux[0]]
    else:
        wav = wav_um
        flx = flux
    
    for t in np.arange(10, 2000, 10):
    
        f1 = bnu_wav_micron(wav[0], t)
        f2 = bnu_wav_micron(wav[1], t)
        
#         print(f1,f2,f1/f2,flx[0],flx[1], flx[0]/flx[1])
        
        if f1/f2 > flx[0]/flx[1]:
            return t
