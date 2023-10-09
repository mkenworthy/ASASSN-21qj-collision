import numpy as np
import matplotlib.pyplot as plt
import json
import utils

import paths

# SED
with open(paths.scripts / 'phoenix_sol_.json') as f:
    r = json.load(f)

# make the plot
lw = 2

fig,ax = plt.subplots(figsize=(5.5,4))

# just stellar spectrum
ax.loglog(r['star_spec']['wavelength'],r['star_spec']['fnujy'],label='star', color='C0', linewidth=lw)

wavelength = np.array(r['phot_wavelength'][0])
filter = np.array(r['phot_band'][0])
fnujy = np.array(r['phot_fnujy'][0])
e_fnujy = np.array(r['phot_e_fnujy'][0])
upperlim = np.array(r['phot_upperlim'][0])
ignore = np.array(r['phot_ignore'][0])

# photometry
ok = np.invert(np.logical_or(upperlim, ignore))
for i,f in enumerate(filter):
    if '_' in f or 'STROM' in f:
        ok[i] = False

ax.errorbar(wavelength[ok], fnujy[ok], yerr=e_fnujy[ok], fmt='o',color='firebrick', label='pre-brightening')
ax.plot(wavelength[upperlim][0], fnujy[upperlim][0], 'v', color='firebrick')

# excess flux
w1f = 9.6e-3
w1fe = 3e-4
w2f = 7.4e-3
w2fe = 3e-4
ax.errorbar([3.38, 4.63], [w1f, w2f], yerr=[w1fe, w2fe], fmt='o',color='blue', label='post-brightening')
ax.plot(wavelength[upperlim][1], fnujy[upperlim][1], 'v', color='blue')
wav = np.array(r['star_spec']['wavelength'])
bnu = utils.bnu_wav_micron(wav, 1000)
bnu /= np.max(bnu) * 2.5e2
ax.plot(wav, bnu, label='1000K BB', color='firebrick', linewidth=lw)
ax.plot(wav, np.array(r['star_spec']['fnujy'])+bnu, label='star + 1000K BB', linewidth=lw, color='C2')

# cooler bb, converting mass to area with given size
bnu = utils.bnu_wav_micron(wav, 100)
mass = 20 * 6e24 * 0.01
size = 1e-7
area = 3/4 * mass / size / 5e3
norm = (area / 1.5e11**2) / (1 / r['main_results'][0]['plx_arcsec'])**2

# 0.1um
qabs = np.ones_like(wav) * 0.01
# 1um
# qabs = np.ones_like(wav)
# generic for 1um or less
qabs[wav>10] *= (10/wav[wav>10])**1.5

ax.plot(wav, 2.35e-11 * bnu * norm * qabs, '--', label='0.1$\mu$m @ 100K', linewidth=2, color='C3')

# sanity checking
# ax.plot(wavelength[ignore], fnujy[ignore], '+', color='firebrick')
# ax.loglog(r['star_spec']['wavelength'],np.array(r['star_spec']['fnujy'])*0.8, '--')
# ax.plot([3.38, 4.63], [3.2e-3, 4.2e-3], '+')
# ld = np.trapz(np.flipud(bnu), np.flipud(1/wav))
# ls = np.trapz(np.flipud(r['star_spec']['fnujy']), np.flipud(1/wav))
# print(ls, ld, ld/ls, ld/(ls*0.8))

# annotation
ax.set_ylim(1e-5,0.5)
ax.set_xlim(0.3,1500)
ax.set_ylabel('flux density / Jy')
ax.set_xlabel('wavelength / $\mu$m')

ax.legend(frameon=True)

fig.tight_layout()
fig.savefig(paths.figures / 'sed.pdf')

# relevant parameters from SED fit
# WISE precision + photosphere precision
# ALMA flux limit
f_lim = [0.00085072, 0.00006   ]
lim_wav = [4.6, 880]
distance = 567.2
lstar = 1.
r_disk_bb = 0.089

r2t = lambda r: 278.3/np.sqrt(r) * lstar**0.25
t2r = lambda t: (278.3/t)**2 * lstar**2

# some arrays
r = np.logspace(-2, 2)
temp = r2t(r)

lims = [] #np.array([])
for i,_ in enumerate(f_lim):
    lim = 3.4e9 * f_lim[i] * distance**2 / r**2 / utils.bnu_wav_micron(lim_wav[i],temp)
    lims.append(lim)
    
lims = np.array(lims)

f = 0.04 # fractional luminosity from SED fit

# estimate emitting dust area from L = sigma_k A T^4
# and in Solar radii with A = 4pi R^2
sigma_k = 5.67e-8
Lsun = 3.83e26
Rsun = 6e8
tdust = 1000

area = f * lstar * Lsun / (sigma_k * tdust**4)
print(area / 1.5e11**2)
np.sqrt(area / 4 / np.pi) / Rsun

fig, ax = plt.subplots(figsize=(5.8,4))

ax.vlines(1, *ax.get_ylim(), linestyles=':', label='Optical duration')
ax.vlines(36, *ax.get_ylim(), linestyles='-.', label='Optical gradients')
ax.vlines(2, *ax.get_ylim(), linestyles='--', label='IR-opt delay')

ok = r > r_disk_bb
ax.loglog(r[ok], lims[0,ok]*5.4, 'red', label='WISE W2 flux')
ax.loglog(r[ok], lims[1,ok],label='ALMA upper limit')

ax.plot(r_disk_bb, f, 'or')

#ax.loglog(r, lims[0], 'grey', alpha=0.5, label='WISE limit')

secax = ax.secondary_xaxis('top', functions=(r2t, t2r))
secax.set_xlabel('temperature / K')

ax.set_ylim(2e-3,1)
ax.set_xlim(1e-2,1e2)
ax.set_xlabel('radius / au')
ax.set_ylabel('fractional luminosity')
ax.legend()

fig.savefig(paths.figures / 'fvsr.pdf')
