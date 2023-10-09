import numpy as np
import matplotlib.pyplot as plt
import paths

from astropy.io import fits

from astropy.timeseries import LombScargle


import matplotlib as mpl
mpl.rcParams.update({'font.size': 10})
mpl.rcParams.update({'lines.markersize': 1})

fits_file08 = paths.data / "tess/hlsp_qlp_tess_ffi_s0008-0000000182582608_tess_v01_llc.fits"
fits_file35 = paths.data / "tess/hlsp_qlp_tess_ffi_s0035-0000000182582608_tess_v01_llc.fits"
fits_file34 = paths.data / "tess/hlsp_qlp_tess_ffi_s0034-0000000182582608_tess_v01_llc.fits"


def rtess(filename):

    with fits.open(filename, mode="readonly") as hdulist:
        tess_bjds = hdulist[1].data['TIME']+57000.
        sap_fluxes = hdulist[1].data['SAP_FLUX']
        kspsap_fluxes = hdulist[1].data['KSPSAP_FLUX']
        qual_flags = hdulist[1].data['QUALITY']

    return tess_bjds, sap_fluxes, kspsap_fluxes, qual_flags

fig, (ax1, ax2, ax3) = plt.subplots(3,1,figsize = (10,5))

(tess_bjds8, sap_fluxes8, kspsap_fluxes8, qual_flags8) = rtess(fits_file08)

# Locate quality flags greater than zero.
where_gt08 = np.where(qual_flags8 > 0)[0]
where_lt18 = np.where(qual_flags8 < 1)[0]

# Overplot the fluxes with quality flags greater than zero in red.
ax1.plot(tess_bjds8[where_lt18], sap_fluxes8[where_lt18], 'ko', rasterized=True)

# masked time and flux for s08
t8 = tess_bjds8[where_lt18]
f8 = sap_fluxes8[where_lt18]-1.0
frequency8, power8 = LombScargle(t8,f8).autopower(minimum_frequency=0.1,
    maximum_frequency=12,
    samples_per_peak=10)

per8 = 1./frequency8
ax3.plot(per8, power8, color='blue')


(tess_bjds34, sap_fluxes34, kspsap_fluxes34, qual_flags34) = rtess(fits_file34)

# Locate quality flags greater than zero.
where_gt034 = np.where(qual_flags34 > 0)[0]
where_lt134 = np.where(qual_flags34 < 1)[0]

ax2.plot(tess_bjds34[where_lt134], sap_fluxes34[where_lt134], 'ko', rasterized=True)

(tess_bjds35, sap_fluxes35, kspsap_fluxes35, qual_flags35) = rtess(fits_file35)

# Locate quality flags greater than zero.
where_gt035 = np.where(qual_flags35 > 0)[0]
where_lt135 = np.where(qual_flags35 < 1)[0]

ax2.plot(tess_bjds35[where_lt135], sap_fluxes35[where_lt135], 'ko', rasterized=True)



# trying all 3 data sets - conclusion - doesn't look that great....

### time_all = np.append(np.append(tess_bjds8[where_lt18], tess_bjds34[where_lt134]), tess_bjds35[where_lt135])
### flux_all = np.append(np.append(sap_fluxes8[where_lt18], sap_fluxes34[where_lt134]), sap_fluxes35[where_lt135])
###frequency_all, power_all = LombScargle(time_all, flux_all-1.0).autopower()
## ax3.plot(1./frequency_all, power_all, color='yellow')


# fit a gaussian to the 4.3 day peak (not great, but reasonable estimate for FWHM...)

from astropy.modeling import models, fitting

# select the periodogram data around the peak
tlow=3.7
thig=5.1
m = (per8>tlow)*(per8<thig)

# Fit the data using a Gaussian
g_init = models.Gaussian1D(amplitude=0.6, mean=4.3, stddev=1.)
fit_g = fitting.LevMarLSQFitter()
g = fit_g(g_init, per8[m], power8[m])

# get results of gaussian fitting
fwhm = g.stddev.value*2*np.sqrt(2*np.log(2))
period = g.mean.value
amp = g.amplitude.value

print(f'period is {period:5.2f}+-{fwhm/2:5.2f} days')

str = f'$P={period:5.2f}\pm{fwhm/2:5.2f}$ days'

ax3.plot(period+fwhm/2.,amp/2.,'ko')
ax3.plot(period-fwhm/2.,amp/2.,'ko')

ax3.text(5,0.4,str,fontsize=18)
# plotting the Gaussian fit to the 4.33d peak
# ax3.plot(per8[m], g(per8[m]), label='Gaussian')

ax3.set_ylim(0,0.6)
ax3.set_xlim(0.5,10)
#ax1.legend()
ax3.set_ylabel("Normalised power")
ax2.set_ylabel("Normalised flux")
ax1.set_ylabel("Normalised flux")
ax1.set_xlabel('Time [MJD]',fontsize=12)
ax2.set_xlabel('Time [MJD]',fontsize=12)
ax3.set_xlabel('Period [days]',fontsize=12)

ax1.text(58518,0.99,"S08",fontsize=24,color='gray', ha='left', va='bottom')
ax2.text(59229,0.94,"S34",fontsize=24,color='gray', ha='left', va='bottom')
ax2.text(59256,0.94,"S35",fontsize=24,color='gray', ha='left', va='bottom')

plt.tight_layout()

plt.savefig(paths.figures / 'all_tess_epochs.pdf',
    dpi=200)
plt.draw()
plt.show()
