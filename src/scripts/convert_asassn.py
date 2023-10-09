import numpy as np
import matplotlib.pyplot as plt
from astropy.io import ascii
from astropy.table import unique,vstack
import paths

fin='asassn/light_curve_410e0d3c-687e-40a3-b7cb-af0057695e0b.csv'
t = ascii.read(paths.data / fin)

#      HJD           UT Date       Camera FWHM Limit   mag   mag_err flux(mJy) flux_err Filter
# ------------- ------------------ ------ ---- ------ ------ ------- --------- -------- ------
# 2457420.65322 2016-02-02.1500246     be 1.46 17.458  13.45   0.005    15.995     0.08      V

t['MJD'] = t['HJD']-2400000.5


# reject noisy points
t = t[(t['flux(mJy)']<18)]
print('rejected ASASSN points with high fluxes in both bands')



fig, (ax) = plt.subplots(1,1,figsize=(12,6))
ax.errorbar(t['MJD'],t['flux(mJy)'],yerr=t['flux_err'],fmt='.')
ax.set_ylabel('Flux [mJy]')
ax.set_xlabel('Epoch [MJD]')
ax.set_title('data from {}'.format(fin))
fig.savefig('_check_asassn0.pdf')


# get a list of the unique bandpasses
t_by_filter = t.group_by('Filter')
print('all observed photometric bands:')
print(t_by_filter.groups.keys)

fig, (ax) = plt.subplots(1,1,figsize=(12,6))
ax.set_ylabel('Flux [mJy]')
ax.set_xlabel('Epoch [MJD]')
ax.set_title('data from {}'.format(fin))


for key, group in zip(t_by_filter.groups.keys, t_by_filter.groups):
    ax.errorbar(group['MJD'],group['flux(mJy)'],yerr=group['flux_err'],label=key['Filter'],fmt='.')

    print('')

ax.legend()
fig.savefig('_check_asassn1.pdf')

(tV, tg) = t_by_filter.groups

# check errors in both filters
fig, (ax1,ax2) = plt.subplots(1,2,figsize=(12,6))
ax1.hist(tV['flux_err'],bins=50)
ax2.hist(tg['flux_err'],bins=50)
ax1.set_ylabel('N')
ax1.set_xlabel('V Flux err ')
ax2.set_xlabel('V Flux err ')
ax1.set_title('data from {}'.format(fin))
fig.savefig('_check_asassn_2err.pdf')

# reject low flux points
tV = tV[(tV['flux_err']<0.5)]
tg = tg[(tg['flux_err']<0.5)]
print('rejecting noisy points in g and V data separately in ASASSN')



V_flux_norm = 15.71
g_flux_norm = 11.13


tV['fnorm'] = tV['flux(mJy)']/V_flux_norm
tg['fnorm'] = tg['flux(mJy)']/g_flux_norm

tV['fnormerr'] = tV['flux_err']/V_flux_norm
tg['fnormerr'] = tg['flux_err']/g_flux_norm

fig, (ax) = plt.subplots(1,1,figsize=(12,6))
ax.set_ylabel('Flux [mJy]')
ax.set_xlabel('Epoch [MJD]')
ax.set_title('data from {}'.format(fin))
ax.errorbar(tV['MJD'],tV['flux(mJy)'],yerr=tV['flux_err'],label='V',fmt='.')
ax.errorbar(tg['MJD'],tg['flux(mJy)'],yerr=tg['flux_err'],label='g',fmt='.')
fig.savefig('_check_asassn3.pdf')

fig, (ax) = plt.subplots(1,1,figsize=(12,6))
ax.set_ylabel('Flux [normalised]')
ax.set_xlabel('Epoch [MJD]')
ax.set_title('Normalised ASASSN data {}'.format(fin))
ax.errorbar(tV['MJD'],tV['fnorm'],yerr=tV['fnormerr'],label='V',fmt='.',alpha=0.3)
ax.errorbar(tg['MJD'],tg['fnorm'],yerr=tg['fnormerr'],label='g',fmt='.',alpha=0.3)
fig.savefig('_check_asassn4.pdf')


# count up how many points are in each band
print(f'number of V points: {len(tV)}')
print(f'number of g points: {len(tg)}')



tn = vstack([tV,tg])
tn['Survey'] = "ASASSN"

tn.write(paths.data / 'obs_ASASSN.ecsv',format='ascii.ecsv',overwrite=True)
