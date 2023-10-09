import numpy as np
import matplotlib.pyplot as plt
from astropy.io import ascii
from astropy.time import Time
import paths

import matplotlib as mpl
mpl.rcParams.update({'font.size': 14})

twi = ascii.read(paths.data / 'obs_NEOWISE.ecsv')
twicol = ascii.read(paths.data / 'NEOWISE_coltemp.ecsv')
tat = ascii.read(paths.data / 'obs_ATLAS.ecsv')
tas = ascii.read(paths.data / 'obs_ASASSN.ecsv')

tasg = tas[tas['Filter']=='g']
tasV = tas[tas['Filter']=='V']

fig, (ax1,ax2,ax3) = plt.subplots(3,1,figsize = (10,7), sharex=True)

ax2.errorbar(twi['MJD'],twi['w1'],yerr=twi['w1err'],
    color='blue',fmt='.',
    ms=10,
    label='WISE W1')
ax2.errorbar(twi['MJD'],twi['w2'],yerr=twi['w2err'],
    color='red', fmt='.',
    ms=10,
    label='WISE W2')
# MAK added twicol w1w2temp_err must be positive - remove when bug is found!
err=twicol['w1w2temp_err']
err[(err<0)] = 0.0001
ax3.errorbar(twicol['MJD'],twicol['w1w2temp'],yerr=twicol['w1w2temp_err'],
    ms=10,
    fmt='.')

ax1.scatter(tasg['MJD'],tasg['fnorm'],
    color='green',
    alpha=0.2,
    s=10,
    edgecolors='none',
    label='ASASSN g\'')
ax1.scatter(tasV['MJD'],tasV['fnorm'],
    color='blue',
    alpha=0.2,
    s=10,
    edgecolors='none',
    label='ASASSN V')

ax1.set_ylim(0,1.1)
ax2.set_ylim(12,10.5)
ax1.set_xlim(56600,60000)
ax1.legend()
ax2.legend(loc=2, frameon=False)
ax3.set_xlabel('Epoch [MJD]')
ax1.set_ylabel('Normalised flux')
ax2.set_ylabel('Magnitude')
ax3.set_ylabel('W1-W2 temp. / K')

plt.savefig(paths.figures / 'all_photometry.png',
    bbox_inches='tight',
    dpi=200)
plt.draw()
plt.show()
