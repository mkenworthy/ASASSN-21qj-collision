import numpy as np
import matplotlib.pyplot as plt
from astropy.io import ascii
from astropy.time import Time
import paths

import matplotlib as mpl
mpl.rcParams.update({'font.size': 12})


mpl.rcParams['font.family'] = 'sans-serif'
#plt.rc('font', family='Helvetica')
mpl.rcParams['font.sans-serif'] = ['Verdana','Helvetica']
mpl.rcParams['pdf.fonttype']=42

twi = ascii.read(paths.data / 'obs_NEOWISE.ecsv')
twicol = ascii.read(paths.data / 'NEOWISE_coltemp.ecsv')
tat = ascii.read(paths.data / 'obs_ATLAS.ecsv')
tas = ascii.read(paths.data / 'obs_ASASSN.ecsv')

tasg = tas[tas['Filter']=='g']
tasV = tas[tas['Filter']=='V']

fig, (ax1,ax2,ax3) = plt.subplots(3,1,figsize = (10,7), sharex=True)

# NEOWISE

#ax2.set_ylim(12.1,10.4)

# select pre collision points to estimate and remove magnitude
precoll_start = 56000.
precoll_end   = 58100.

m = (twi['MJD'] > precoll_start) * (twi['MJD']<precoll_end)

w1_precoll = np.mean(twi['w1'][m])
w2_precoll = np.mean(twi['w2'][m])

print(f'W1 magntiude before collision {w1_precoll:4.2f}')
print(f'W2 magntiude before collision {w2_precoll:4.2f}')

w1flux = np.power(10,(twi['w1']-w1_precoll)/-2.5)
w2flux = np.power(10,(twi['w2']-w2_precoll)/-2.5)



ax2.errorbar(twi['MJD'],w1flux,yerr=twi['w1err'],
    color='blue',
    fmt='.',
    linestyle='dashed',
    ms=10,
    label='WISE W1')

ax2.errorbar(twi['MJD'],w2flux,yerr=twi['w2err'],
    color='red', 
    fmt='.',
    linestyle='dashed',
    ms=10,
    label='WISE W2')

ax2.text(58500, 1.1,"WISE W1", color='blue')
ax2.text(58500, 2.4,"WISE W2", color='red')
ax2.axhline(y = 1.0, color = 'gray', linestyle = 'dotted')
ax3.axhline(y = 0.0, color = 'gray', linestyle = 'dotted')

# MAK added twicol w1w2temp_err must be positive - remove when bug is found!
err=twicol['w1w2temp_err']
err[(err<0)] = 0.0001

# COLOR TEMP

ax3.errorbar(twicol['MJD'], twicol['w1w2temp'], yerr=twicol['w1w2temp_err'],
    ms=10,
    fmt='.',
    color='black')

ax3.axhline(0,color='black',alpha=0.1,linestyle='dotted')
ax3.axhline(1000,color='black',alpha=0.1,linestyle='dotted')

### ASAS DATA

ax1.set_xlim(56600,60000)
ax1.set_ylim(-0.05,1.1)

ax1.scatter(tasg['MJD'],tasg['fnorm'],
    color='green',
    alpha=0.2,
    s=10,
    marker='^',
    edgecolors='none',
    label='ASASSN g\'',
    rasterized=True)

ax1.scatter(tasV['MJD'],tasV['fnorm'],
    color='green',
    alpha=0.2,
    s=10,
    marker='v',
    edgecolors='none',
    label='ASASSN V',
    rasterized=True)

ax1.axhline(0,color='black',alpha=0.1,linestyle='dotted')
ax1.axhline(1,color='black',alpha=0.1,linestyle='dotted')

for p in (ax1,ax2,ax3):
    p.axvline(57000,alpha=0.1)
    p.axvline(57500,alpha=0.1)
    p.axvline(58000,alpha=0.1)
    p.axvline(58500,alpha=0.1)
    p.axvline(59000,alpha=0.1)
    p.axvline(59500,alpha=0.1)
    p.axvline(60000,alpha=0.1)

#ax2.legend(loc=2, frameon=False)

ax2.set_ylim(0.5,3.0)

ax3.set_xlabel('Epoch [MJD]',fontsize=14)
ax1.set_ylabel('ASAS-SN flux',fontsize=14)
ax2.set_ylabel('Normalised flux',fontsize=14)
ax3.set_ylabel('W1-W2 temp. [K]',fontsize=14)


tyb = dict(color='black', weight='bold', fontsize=12)
ax1.text(0.03, 0.9, 'a', ha='right', va='top', transform=ax1.transAxes, **tyb)
ax2.text(0.03, 0.9, 'b', ha='right', va='top', transform=ax2.transAxes, **tyb)
ax3.text(0.03, 0.9, 'c', ha='right', va='top', transform=ax3.transAxes, **tyb)

plt.tight_layout()
plt.savefig(paths.figures / 'all_phot_nature_delta_flux.pdf',
    dpi=400)

plt.draw()
plt.show()
