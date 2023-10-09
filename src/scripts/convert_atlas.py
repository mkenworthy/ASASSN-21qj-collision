import numpy as np
import matplotlib.pyplot as plt
from astropy.io import ascii
from astropy.table import unique,vstack
import paths

fin='atlas/job211831.txt'
t = ascii.read(paths.data / fin)

# MJD          m      dm   uJy   duJy F err chi/N     RA       Dec        x        y     maj  min   phi  apfit mag5sig Sky   Obs
# 58037.635279  13.616  0.003 12984   44 o  0 1718.56 123.84754 -38.98983  1666.50  4612.12 2.32 2.21 -22.2 -0.420 18.66 19.35 02a58037o0713o

# reject noisy points
t = t[(t['duJy']<100)]

# reject low flux points and bad filter...
t = t[(t['uJy']>500)]

fig, (ax) = plt.subplots(1,1,figsize=(12,6))
ax.errorbar(t['MJD'],t['uJy'],yerr=t['duJy'],fmt='.')
ax.set_ylabel('Flux [uJy]')
ax.set_xlabel('Epoch [MJD]')
ax.set_title('data from {}'.format(fin))
fig.savefig('_check_atlas0.pdf')

# get a list of the unique bandpasses
t_by_filter = t.group_by('F')
print('all observed photometric bands:')
print(t_by_filter.groups.keys)
#print(t_by_filter.groups[0])

fig, (ax) = plt.subplots(1,1,figsize=(12,6))
ax.set_ylabel('Flux [uJy]')
ax.set_xlabel('Epoch [MJD]')
ax.set_title('data from {}'.format(fin))

for key, group in zip(t_by_filter.groups.keys, t_by_filter.groups):
#    print(f'****** {key["F"]} *******')
#    print(group)
    plt.errorbar(group['MJD'],group['uJy'],yerr=group['duJy'],label=key['F'],fmt='.')

    print('')

plt.legend()
fig.savefig('_check_atlas1.pdf')




(tc, to) = t_by_filter.groups

# check errors in both filters
fig, (ax1,ax2) = plt.subplots(1,2,figsize=(12,6))
ax1.hist(tc['duJy'],bins=50)
ax2.hist(to['duJy'],bins=50)
ax1.set_ylabel('N')
ax1.set_xlabel('c Flux err ')
ax2.set_xlabel('o Flux err ')
ax1.set_title('data from {}'.format(fin))
fig.savefig('_check_atlas_2err.pdf')


# reject low flux points
tc = tc[(tc['duJy']<45)]
to = to[(to['duJy']<65)]
print('rejecting noisy points in g and V data separately in A')



c_flux_norm = 12000
o_flux_norm = 15000

tc['fnorm'] = tc['uJy']/c_flux_norm
to['fnorm'] = to['uJy']/o_flux_norm

tc['fnormerr'] = tc['duJy']/c_flux_norm
to['fnormerr'] = to['duJy']/o_flux_norm

fig, (ax) = plt.subplots(1,1,figsize=(12,6))
ax.set_ylabel('Flux [uJy]')
ax.set_xlabel('Epoch [MJD]')
ax.set_title('data from {}'.format(fin))
ax.errorbar(tc['MJD'],tc['uJy'],yerr=tc['duJy'],label='c',fmt='.')
ax.errorbar(to['MJD'],to['uJy'],yerr=to['duJy'],label='o',fmt='.')
fig.savefig('_check_atlas3.pdf')


fig, (ax) = plt.subplots(1,1,figsize=(12,6))
ax.set_ylabel('Flux [normalised]')
ax.set_xlabel('Epoch [MJD]')
ax.set_title('Normalised ASASSN data {}'.format(fin))
ax.errorbar(tc['MJD'],tc['fnorm'],yerr=tc['fnormerr'],label='c',fmt='.',alpha=0.3)
ax.errorbar(to['MJD'],to['fnorm'],yerr=to['fnormerr'],label='o',fmt='.',alpha=0.3)

ax.legend()
fig.savefig('_check_atlas4.pdf')

tn = vstack([tc,to])
tn['Filter'] = tn['F']
tn['Survey'] = "ATLAS"


# count up how many points are in each band
print(f'ATLAS & $c$ & {len(tc)} \\\\')
print(f'ATLAS & $o$ & {len(to)} \\\\')

tn.write(paths.data / 'obs_ATLAS.ecsv',format='ascii.ecsv',overwrite=True)
