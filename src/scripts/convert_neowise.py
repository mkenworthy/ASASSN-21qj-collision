import numpy as np
import matplotlib.pyplot as plt
from astropy.io import ascii
from astropy.table import unique,vstack,Table
import paths


##t = ascii.read(paths.data / 'neowise/ASASSN-21qj_2013-2021.tbl')
t = ascii.read(paths.data / 'neowise/ASASSN-21qj_2013-2022.tbl')

fig, ax = plt.subplots(2,1,figsize=(8,5),sharex=True)
ax[0].errorbar(t['mjd'],t['w1mpro'],yerr=t['w1sigmpro'],fmt='.')
ax[0].invert_yaxis()
ax[1].errorbar(t['mjd'],t['w2mpro'],yerr=t['w2sigmpro'],fmt='.')
ax[1].invert_yaxis()
ax[1].set_xlabel('Epoch [MJD]')
ax[0].set_ylabel('w1mpro')
ax[1].set_ylabel('w2mpro')
fig.savefig('_check_neowise0.pdf')

# calculate weighted mean at each epoch

fig, ax = plt.subplots(3,1,figsize=(8,7),sharex=True)
ax[0].errorbar(t['mjd'],t['w1mpro'],yerr=t['w1sigmpro'],fmt='.')
ax[0].invert_yaxis()
ax[1].errorbar(t['mjd'],t['w2mpro'],yerr=t['w2sigmpro'],fmt='.')
ax[1].invert_yaxis()

ax[2].set_xlabel('Epoch [MJD]')
ax[0].set_ylabel('w1mpro')
ax[1].set_ylabel('w2mpro')
ax[2].set_ylabel('w1-w2')
ax[0].set_ylim(12,10.5)
ax[1].set_ylim(12,10.5)

t_s = 56787
t_e = 59532
t_e = 59600+300

# number of separate epochs
nmag = 18

wt = np.zeros(nmag)
w1 = np.zeros(nmag)
w2 = np.zeros(nmag)
wcol = np.zeros(nmag)
wcolsig = np.zeros(nmag)
w1sig = np.zeros(nmag)
w2sig = np.zeros(nmag)

for (j,i) in enumerate(np.linspace(t_s, t_e, nmag)):
#    ax[0].scatter(i-60,11.4)
#    ax[0].scatter(i+60,11.4)
#    ax[1].scatter(i-60,11.4)
#    ax[1].scatter(i+60,11.4)
    # select the points within an epoch
    m = (t['mjd']>(i-60))*(t['mjd']<(i+60))

    w_time = t['mjd'][m]

    w1_mag = t['w1mpro'][m]
    w1_sig = t['w1sigmpro'][m]
    w2_mag = t['w2mpro'][m]
    w2_sig = t['w2sigmpro'][m]

    w1mean = w1_mag.mean()
    w2mean = w2_mag.mean()

    ndiv = np.sqrt(np.sum(m))
    w1sig[j] = np.std(w1_mag)
    w2sig[j] = np.std(w2_mag)

    wcol[j] = w1mean-w2mean
    wcolsig[j] = np.sqrt(w1sig[j]*w1sig[j]+w2sig[j]*w2sig[j])
    w1[j] = w1mean
    w2[j] = w2mean
    wt[j] = np.average(t['mjd'][m])

ax[0].errorbar(wt,w1,yerr=w1sig,fmt='.',zorder=5)
ax[1].errorbar(wt,w2,yerr=w2sig,fmt='.',zorder=5)
ax[2].errorbar(wt,wcol,yerr=wcolsig,fmt='.',zorder=5)
fig.suptitle("NEOWISE photometry of ASASSSN-21qj")
#fig.savefig("NEOWISE_ASASSN-21dj.pdf")
fig.savefig('_check_neowise1.pdf')

tn = Table([wt, w1,w1sig,w2,w2sig,wcol,wcolsig],
            names=['MJD','w1','w1err','w2','w2err','w1w2','w1w2err'])

tn.format = '%.4f'
tn['MJD'].format = '%.5f'
tn['w1'].format = '%.3f'
tn['w1err'].format = '%.3f'
tn['w2'].format = '%.3f'
tn['w2err'].format = '%.3f'
tn['w1w2'].format = '%.3f'
tn['w1w2err'].format = '%.3f'

print(f'NEOWISE & $W1$ & {len(w1)} \\\\')
print(f'NEOWISE & $W2$ & {len(w2)} \\\\')


tn.write(paths.data / 'obs_NEOWISE.ecsv',format='ascii.ecsv',overwrite=True)
