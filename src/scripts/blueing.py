#!/usr/bin/env python
# coding: utf-8

# # Blueing
# Look at colour dependence with dip depth, this tells us about how much light is scattered off dust. See 1988SvAL...14...27G.

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import paths

#get_ipython().run_line_magic('matplotlib', 'notebook')


# ### data

# In[2]:


# data has date, mag, mag_err
data = np.loadtxt(paths.data/'aavso/aavsodata_63e2220f49f39.txt', skiprows=1, delimiter=',', usecols=(0,1,2))
band = np.loadtxt(paths.data/'aavso/aavsodata_63e2220f49f39.txt', skiprows=1, delimiter=',', usecols=(4), dtype=str)

len(data) == len(band)

data


# In[3]:


fig, ax = plt.subplots(3, figsize=(9.5,5))
for a,b in zip(ax,['B', 'V', 'I']):
    ok = band == b
    a.errorbar(data[ok,0], data[ok,1], yerr=data[ok,2], fmt='.', label=b)
    a.yaxis.set_inverted(True)
    a.legend()
    
fig.tight_layout()


# In[4]:


# merge data per night
date_int = data[:,0].round()
date_uniq = np.unique(date_int)

bvi_date = np.array([])
bvi = []
bvi_err = []

for d in date_uniq:
    
    bvi_tmp = []
    bvi_err_tmp = []
    for b in ['B','V','I']:
    
        use = (date_int == d) & (band == b)
        if np.sum(use) == 0:
            break
        bvi_tmp.append( np.mean(data[use,1]) )
        bvi_err_tmp.append( np.max(data[use,2]) )
        
    if len(bvi_tmp) != 3:
        continue
    bvi_date = np.append(bvi_date, d)
    bvi.append(bvi_tmp)
    bvi_err.append(bvi_err_tmp)

bvi = np.array(bvi)
bvi_err = np.array(bvi_err)


# In[5]:


fig, ax = plt.subplots(3, figsize=(9.5,5))
for i,b in enumerate(['B','V','I']):
    ax[i].errorbar(bvi_date, bvi[:,i], yerr=bvi_err[:,i], fmt='.', label=b)
    ax[i].yaxis.set_inverted(True)
    ax[i].legend()
    
fig.tight_layout()


# In[6]:


fig, ax = plt.subplots(1, 2, figsize=(9.5,5))

ax[0].scatter(bvi[:,0]-bvi[:,1], bvi[:,1], c=bvi_date, s=2)
ax[1].scatter(bvi[:,1]-bvi[:,2], bvi[:,1], c=bvi_date, s=2)

ax[0].set_xlabel('B-V')
ax[1].set_xlabel('V-I')
ax[0].set_ylabel('V')

ax[0].yaxis.set_inverted(True)
ax[1].yaxis.set_inverted(True)
fig.tight_layout()


# In[7]:


fig, ax = plt.subplots(1, 2, figsize=(9.5,5))

ax[0].errorbar(bvi[:,0]-bvi[:,1], bvi[:,1], xerr=bvi_err[:,0]+bvi_err[:,1], yerr=bvi_err[:,1], fmt='.')
ax[1].errorbar(bvi[:,1]-bvi[:,2], bvi[:,1], xerr=bvi_err[:,1]+bvi_err[:,2], yerr=bvi_err[:,1], fmt='.')

ax[0].set_xlabel('B-V')
ax[1].set_xlabel('V-I')
ax[0].set_ylabel('V')

ax[0].yaxis.set_inverted(True)
ax[1].yaxis.set_inverted(True)
fig.tight_layout()


# ### theoretical model

# In[8]:


# band wavelengths, these are means for BJ, VJ, and IC
lam_b = 438e-9
lam_v = 550e-9
lam_i = 802e-9

# zero points in Jy
zp_b = 3965.367
zp_v = 3588.567
zp_i = 2419.640


# In[9]:


# our star, finding flux for 13.6 mag in V, B-V=0.7 and V-I=0.7
f0_v = zp_v * 10**(-0.4*13.6)
f0_b = f0_v * 10**(-0.4*0.6) * zp_b/zp_v
f0_i = f0_v / 10**(-0.4*0.65) * zp_i/zp_v

# check
-2.5*np.log10(f0_b/zp_b) + 2.5*np.log10(f0_v/zp_v) # B-V
-2.5*np.log10(f0_v/zp_v) + 2.5*np.log10(f0_i/zp_i) # V-I


# In[10]:


def fnu_tau(tau, s, f0, c1):
    '''Reddening and scattering contributions to flux, a la Grinin'''
#     return f0 * (np.exp(-c1*tau/lam) + c2*s/lam**4) # Rayleigh scattering
    return f0 * (np.exp(-c1*tau) + s) # grey scattering

def mag_tau(tau, s, f0, zp, c1):
    return -2.5 * np.log10(fnu_tau(tau, s, f0, c1) / zp)


# In[11]:


# paramter space we will explore
tau = np.linspace(0, 5.5, endpoint=True)
s = 0.075


# In[12]:


# evolution with optical depth
magb = mag_tau(tau, s, f0_b, zp_b, 1.9)
magv = mag_tau(tau, s, f0_v, zp_v, 1)
magi = mag_tau(tau, s, f0_i, zp_i, 0.38)


# In[13]:


fig, ax = plt.subplots()
ax.plot(tau, magb, label='B')
ax.plot(tau, magv, label='V')
ax.plot(tau, magi, label='I')

ax.set_xlabel('$\\tau_V$')
ax.set_ylabel('V / mag')
ax.yaxis.set_inverted(True)
ax.legend()
fig.tight_layout()


# In[14]:


fig, ax = plt.subplots()
ax.plot(magb-magv, magv)

ax.set_xlabel('B-V / mag')
ax.set_ylabel('V / mag')
ax.yaxis.set_inverted(True)
fig.tight_layout()


# ### together

# In[15]:


v_star = 13.5
vv = np.linspace(v_star, np.max(bvi[:,1]))
bv_star = 0.6
vi_star = 0.65
ab_v = 0.75
ai_v = 0.58

def v_col(v, v0, col0, ax_v):
    return (v-v0) * ax_v + col0


# In[16]:


fig, ax = plt.subplots(1, 2, figsize=(12,5.5), sharey=True)

ax[0].errorbar(bvi[:,0]-bvi[:,1], bvi[:,1], xerr=bvi_err[:,0]+bvi_err[:,1], yerr=bvi_err[:,1], fmt='.')
ax[1].errorbar(bvi[:,1]-bvi[:,2], bvi[:,1], xerr=bvi_err[:,1]+bvi_err[:,2], yerr=bvi_err[:,1], fmt='.')

ax[0].plot(v_col(vv, v_star, bv_star, ab_v), vv, '--', zorder=99, label=f'$A_B/A_V={1/ab_v:.1f}$')
ax[1].plot(v_col(vv, v_star, vi_star, ai_v), vv, '--', zorder=99, label=f'$A_I/A_V={ai_v:.1f}$')

ax[0].plot(magb-magv, magv, zorder=99, label=f's = {s}')
ax[1].plot(magv-magi, magv, zorder=99, label=f's = {s}')

ax[0].set_xlabel('$B-V$',fontsize=16)
ax[1].set_xlabel('$V-I$',fontsize=16)
ax[0].set_ylabel('$V$',fontsize=16)

ax[0].yaxis.set_inverted(True)
ax[1].yaxis.set_inverted(True)
ax[0].legend(frameon=False)
ax[1].legend(frameon=False)


tyb = dict(color='black', weight='bold', fontsize=16)
ax[0].text(0.1, 0.9, 'a', ha='right', va='top', transform=ax[0].transAxes, **tyb)
ax[1].text(0.1, 0.9, 'b', ha='right', va='top', transform=ax[1].transAxes, **tyb)



fig.tight_layout()
fig.savefig(paths.figures/'blueing.pdf')


# In[ ]:




