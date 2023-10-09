import numpy as np
from astropy.table import Table
import paths
from utils import col2temp
import matplotlib.pyplot as plt

# read from csv created by convert_neowise
t = Table.read(paths.data/'obs_NEOWISE.ecsv')

# get photospheric flux and subtract to get dust colour
# we assume that pre-increase was just the star (which
# may include the companion, it doesn't matter)

# WISE zero points derived with drgmk/sdf, use bandpasses
# and assume CALSPEC Vega model
w1flux = 302.878 * 10**(-0.4*t['w1'])
w2flux = 172.415 * 10**(-0.4*t['w2'])

w1flux_lo = 302.878 * 10**(-0.4*(t['w1']+t['w1err']))
w2flux_lo = 172.415 * 10**(-0.4*(t['w2']+t['w2err']))
w1flux_hi = 302.878 * 10**(-0.4*(t['w1']-t['w1err']))
w2flux_hi = 172.415 * 10**(-0.4*(t['w2']-t['w2err']))

ok = t['MJD'] < 58100
w1phot = np.median(w1flux[ok])
w2phot = np.median(w2flux[ok])
w1phot_err = np.std(w1flux[ok])
w2phot_err = np.std(w2flux[ok])
w1xs = w1flux - w1phot
w2xs = w2flux - w2phot

w1xs_lo = w1flux_lo - w1phot
w2xs_lo = w2flux_lo - w2phot
w1xs_hi = w1flux_hi - w1phot
w2xs_hi = w2flux_hi - w2phot

w1xs_err = np.sqrt( ((w1xs_hi - w1xs_lo) / 2.)**2 + w1phot_err**2 )
w2xs_err = np.sqrt( ((w2xs_hi - w2xs_lo) / 2.)**2 + w2phot_err**2 )

# now get colour temp, floats are the mean wavelengths
# for the WISE W1 and W2 bands
w1w2temp = []
w1w2temp_lo = []
w1w2temp_hi = []
for i in range(len(t['MJD'])):
    w1w2temp.append(col2temp([3.3791878170787886, 4.629290939920992],
                             [w1xs[i], w2xs[i]])
                   )
    w1w2temp_lo.append(col2temp([3.3791878170787886, 4.629290939920992],
                             [w1xs_lo[i], w2xs_hi[i]])
                   )
    w1w2temp_hi.append(col2temp([3.3791878170787886, 4.629290939920992],
                             [w1xs_hi[i], w2xs_lo[i]])
                   )

w1w2temp = np.array(w1w2temp, dtype=float)
w1w2temp_hi = np.array(w1w2temp_hi, dtype=float)
w1w2temp_lo = np.array(w1w2temp_lo, dtype=float)
print(w1w2temp_hi)
print(w1w2temp_lo)
w1w2temp_hi[-1] = 2000.
w1w2temp_err = (w1w2temp_hi-w1w2temp_lo)/2
w1w2temp[ok] = 0
w1w2temp_err[ok] = 0

tout = Table([t['MJD'], w1xs, w2xs, w1xs_err, w2xs_err, w1w2temp, w1w2temp_err],
             names=['MJD','w1excess','w2excess','w1excess_err','w2excess_err','w1w2temp','w1w2temp_err'])

tout.write(paths.data/'NEOWISE_coltemp.ecsv',format='ascii.ecsv',overwrite=True,
           formats={'w1excess':'6.4f','w2excess':'6.4f'})

for i in range(len(t)):
    print(w1w2temp[i], w1w2temp_lo[i], w1w2temp_hi[i],
          (w1w2temp_lo[i]+w1w2temp_hi[i])/2)

fig, ax = plt.subplots(2, 1)

ax[0].errorbar(tout['MJD'], tout['w1excess'], tout['w1excess_err'], fmt='.', label='W1')
ax[0].errorbar(tout['MJD'], tout['w2excess'], tout['w2excess_err'], fmt='.', label='W2')
ax[0].legend()

ax[1].errorbar(tout['MJD'], tout['w1w2temp'], tout['w1w2temp_err'], fmt='.')
ax[1].set_ylabel('Colour temp / K')

ax[1].set_xlabel('MJD')
ax[0].set_ylabel('WISE excess / Jy')

fig.savefig('_check_neowise2.pdf')
