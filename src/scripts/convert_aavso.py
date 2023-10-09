import numpy as np
import matplotlib.pyplot as plt
from astropy.io import ascii
from astropy.table import unique,vstack
import paths

###fin='aavso/aavsodata_636cdb53c895b.txt'
fin='aavso/aavsodata_63e2220f49f39.txt'
t = ascii.read(paths.data / fin)


# JD,Magnitude,Uncertainty,HQuncertainty,Band,Observer Code,Comment Code(s),Comp Star 1,Comp Star 2,Charts,Comments,Transfomed,Airmass,Validation Flag,Cmag,Kmag,HJD,Star Name,Observer Affiliation,Measurement Method,Grouping Method,ADS Reference,Digitizer,Credit
#  2459454.919896,14.844,0.088,,V,HMB,,UCAC4 254-024861,UCAC4 254-024887,,STANDARD MAG: C = 12.206  K = 12.162,0,1.791,Z,17.689,17.636,,ASASSN-21QJ,VVS,STD,,,,

#      HJD           UT Date       Camera FWHM Limit   mag   mag_err flux(mJy) flux_err Filter
# ------------- ------------------ ------ ---- ------ ------ ------- --------- -------- ------
# 2457420.65322 2016-02-02.1500246     be 1.46 17.458  13.45   0.005    15.995     0.08      V

t['MJD'] = t['JD']-2400000.5
t['Filter'] = t['Band']

# reject noisy points
#t = t[(t['flux(mJy)']<18)]
#print('rejected ASASSN points with high fluxes in both bands')



fig, (ax) = plt.subplots(1,1,figsize=(12,6))
ax.errorbar(t['MJD'],t['Magnitude'],yerr=t['Uncertainty'],fmt='.')
ax.set_ylabel('Magnitude')
ax.set_xlabel('Epoch [MJD]')
ax.set_title('data from {}'.format(fin))
fig.savefig('_check_aavso0.pdf')


# get a list of the unique bandpasses
t_by_filter = t.group_by('Filter')
print('all observed photometric bands:')
print(t_by_filter.groups.keys)

fig, (ax) = plt.subplots(1,1,figsize=(12,6))
ax.set_ylabel('Magnitude')
ax.set_xlabel('Epoch [MJD]')
ax.set_title('data from {}'.format(fin))


for key, group in zip(t_by_filter.groups.keys, t_by_filter.groups):
    ax.errorbar(group['MJD'],group['Magnitude'],yerr=group['Uncertainty'],label=key['Filter'],fmt='.')

    print('')

ax.legend()
fig.savefig('_check_aavso1.pdf')

(tB, tI, tV) = t_by_filter.groups

# check errors in both filters
fig, (ax1,ax2,ax3) = plt.subplots(1,3,figsize=(12,6))
ax1.hist(tB['Uncertainty'],bins=50)
ax2.hist(tI['Uncertainty'],bins=50)
ax3.hist(tV['Uncertainty'],bins=50)
ax1.set_ylabel('N')
ax1.set_xlabel('B Flux err ')
ax2.set_xlabel('I Flux err ')
ax3.set_xlabel('V Flux err ')
ax1.set_title('data from {}'.format(fin))
fig.savefig('_check_aavso_2err.pdf')

# reject low flux points
tB = tB[(tB['Uncertainty']<0.2)]
tI = tI[(tI['Uncertainty']<0.1)]
tV = tV[(tV['Uncertainty']<0.06)]
print('rejecting noisy points in g and V data separately in AAVSO')

V_flux_norm = 13.5
I_flux_norm = 12.8
B_flux_norm = 14.2

tB['fnorm'] = np.power(10.,(B_flux_norm-tB['Magnitude'])/2.5)
tI['fnorm'] = np.power(10.,(I_flux_norm-tI['Magnitude'])/2.5)
tV['fnorm'] = np.power(10.,(V_flux_norm-tV['Magnitude'])/2.5)

tB['fnormerr'] = tB['Uncertainty']
tI['fnormerr'] = tI['Uncertainty']
tV['fnormerr'] = tV['Uncertainty']

fig, (ax) = plt.subplots(1,1,figsize=(12,6))
ax.set_ylabel('Flux [normalised]')
ax.set_xlabel('Epoch [MJD]')
ax.set_title('Normalised AAVSO data {}'.format(fin))
ax.errorbar(tB['MJD'],tB['fnorm'],yerr=tB['fnormerr'],label='B',fmt='.',alpha=0.3)
ax.errorbar(tI['MJD'],tI['fnorm'],yerr=tI['fnormerr'],label='I',fmt='.',alpha=0.3)
ax.errorbar(tV['MJD'],tV['fnorm'],yerr=tV['fnormerr'],label='V',fmt='.',alpha=0.3)
fig.savefig('_check_asassn4.pdf')

# count up how many points are in each band
print(f'number of B points: {len(tB)}')
print(f'number of V points: {len(tI)}')
print(f'number of I points: {len(tV)}')


tn = vstack([tB,tI,tV])
tn['Survey'] = "AAVSO"

tn.write(paths.data / 'obs_AAVSO.ecsv',format='ascii.ecsv',overwrite=True)
#plt.show()
