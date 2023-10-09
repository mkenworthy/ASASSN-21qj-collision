# from SJL Jupyter notebook plot_Hill_Bondi_radii_Figure_XXX sent to MAK on 2023-08-15
import numpy as np
import scipy.constants as const

import matplotlib as mpl
import matplotlib.pyplot as plt
import pylab
import matplotlib.cm as cm
from matplotlib import gridspec
from matplotlib.colors import LogNorm


#add Tex fonts if needed
import sys
import os
#if sys.platform== 'darwin':
#    os.environ["PATH"] += os.pathsep + '/Library/TeX/texbin'

#### Properties of the star
Rs=1.009*6.957E8 #Radius [m]
Ts=5900. #Black body emission temperature [K]
Ms=1.988500E30 #mass [kg]

#### Inferred properties of emmitting object
Remit=7*Rs  #size of emitter
Temit=1000. #Temperature of emitter

#### Constants for plotting
REarth = 6.371e6   # m
MEarth = 5.9724e24  # kg

#calculate the hills sphere for bodies at different semi-major axes and masses
a=np.asarray([1,3,5,10])*const.au
M=np.linspace(0.5,50,100)*MEarth

#initialize Hills sphere array
RH=np.ones((np.size(a),np.size(M)))

#Loop over all semi-major axis and calculate the Hills radius
for i in np.arange(np.size(a)):
    RH[i,:]=a[i]*(M/Ms/3.)**(1./3)

#Calculate the Bondi radius for different gases
gas_label=['H2', 'H$_2$O', 'SiO', 'SiO$_2$']
gamma=np.asarray([1.4, 1+1./3, 1.4, 1.4]) #ratio of specific heat capactities
ma=np.asarray([2,18,28,44])*1E-3/const.N_A

#calculate the sound speed
cs=np.sqrt(gamma*const.k*Temit/ma)

#initialize the array
RB=np.ones((np.size(gamma),np.size(M)))

#loop over all gasses and calculate the Bondi radius
for i in np.arange(np.size(gamma)):
    RB[i,:]=2*const.G*M/(cs[i]**2)

#Calculate the cooling of different post-impact bodies

#plot a range of different power law exponents etc.
beta_plot=np.asarray([-3,-2,-1])
Rp_plot=np.asarray([1,10,100])*REarth
Mp_plot=np.asarray([10,50,100])*MEarth

#parameters for calculating the cooling
Nt=1000
tplot=np.linspace(0.01,1100,Nt)*24*60*60
rplot=np.linspace(0.01,Remit,10000)
lheat=2.256E6
fvap=1.0

#initialize arrays to store the output
rcool_pl=np.zeros((np.size(beta_plot),np.size(Rp_plot),np.size(Mp_plot),Nt))
sigma0_pl=np.zeros((np.size(beta_plot),np.size(Rp_plot),np.size(Mp_plot)))
rcool_pl0=np.zeros((np.size(beta_plot),np.size(Rp_plot),np.size(Mp_plot),Nt))
sigma0_pl0=np.zeros((np.size(beta_plot),np.size(Rp_plot),np.size(Mp_plot)))

#loop
for i in np.arange(np.size(beta_plot)):
    beta=beta_plot[i]
    for j in np.arange(np.size(Rp_plot)):
        Rp=Rp_plot[j]
        for k in np.arange(np.size(Mp_plot)):
            Mp=Mp_plot[k]

            #Power law with constant-surface-density core
            if (beta==-2.)|(beta==-2):
                sigma03=Mp/(np.pi*(Rp**2+2*Rp**-beta*(np.log(Remit)-np.log(Rp))))
            else:
                sigma03=Mp/(np.pi*(Rp**2+2*Rp**-beta*(Remit**(beta+2)-\
                                                        Rp**(beta+2))/(beta+2)))

            sigma0_pl[i,j,k]=sigma03 #surface density in central region

            ind=np.where(rplot<Rp)[0]
            sigma3=sigma03*Rp**-beta*rplot**beta
            sigma3[ind]=sigma03 #When inside Rp set to sigma0

            #time for cooling as a function of radius
            tcool3=fvap*sigma3*lheat/(2*const.sigma*(Temit**4)) 

            #radius at which the gas is completely cooled
            rcool3=((tplot*2*const.sigma*(Temit**4))/\
                (fvap*sigma03*Rp**-beta*lheat))**(1./beta)
            ind=np.where(tplot>(fvap*sigma03*lheat)/(2*const.sigma*(Temit**4)))[0]
            if np.size(ind)>0:
                rcool3[ind]=0.
            ind=np.where(rcool3>Remit)[0] #limit the radius to the emitting radius
            rcool3[ind]=Remit

            rcool_pl[i,j,k,:]=rcool3

            #do the same for the initial powerlaw with inner core and sigma=0 at Remit
            if (beta==-2.)|(beta==-2):
                sigma04=Mp/(np.pi*(Remit**2+\
                                     (1./(Rp**beta-Remit**beta))*((2*(np.log(Remit)-\
                                            np.log(Rp)))-\
                                            Rp**beta*(Remit**2-Rp**2))))
            else:
                sigma04=Mp/(np.pi*(Remit**2+\
                                     (1./(Rp**beta-Remit**beta))*((2*(Remit**(beta+2)-\
                                            Rp**(beta+2))/(beta+2))-\
                                            Rp**beta*(Remit**2-Rp**2))))

            sigma0_pl0[i,j,k]=sigma04

            alpha=sigma04/(Rp**beta-Remit**beta)
            gamma_temp=sigma04-alpha*Rp**beta
            ind=np.where(rplot<Rp)[0]
            sigma4=alpha*rplot**beta+gamma_temp
            sigma4[ind]=sigma04

            tcool4=fvap*sigma4*lheat/(2*const.sigma*(Temit**4))

            rcool4=(((tplot*2*const.sigma*(Temit**4))/\
                    (fvap*lheat) - gamma_temp)/alpha)**(1./beta)
            ind=np.where(tplot>(fvap*sigma04*lheat)/(2*const.sigma*(Temit**4)))[0]
            if np.size(ind)>0:
                rcool4[ind]=0.
            ind=np.where(rcool4>Remit)[0]
            rcool4[ind]=Remit

            rcool_pl0[i,j,k,:]=rcool4
print('done')

#plot the combined figure

#intialize the figure
fig = plt.figure(figsize=(7.5, 6.0))
gs = gridspec.GridSpec(2,2)

gs00 = gridspec.GridSpecFromSubplotSpec(2, 1,
                                        height_ratios=[0.8,0.2],
                                        subplot_spec=gs[0:2,0])

ax=[plt.subplot((gs00[0,0])),plt.subplot(gs[0,1]),plt.subplot(gs[1,1])]

#set font size
plt.rc('text', usetex=True)
plt.rc('font', family='sans-serif')
plt.rcParams.update({'font.size': 8, 'legend.fontsize':8,'xtick.labelsize':8, 'axes.titlesize':8, 'axes.labelsize':8,'ytick.labelsize':8})

mpl.rcParams['text.latex.preamble'] = r'\usepackage{siunitx} \sisetup{detect-all} \usepackage{times} \usepackage{helvet} \usepackage{sansmath} \sansmath \usepackage{amsmath}'

ind_beta=1
ind_Rp=1
ind_Mp=1


#Plot the reference line for the radius of the emitting object
ax[0].plot(M/MEarth, np.ones(np.size(M))*Remit, 'k--')

#plot lines for the Hills sphere
col=cm.plasma(np.linspace(0,0.9,np.size(a)))
for i in np.arange(np.size(a)):
    ax[0].plot(M/MEarth, RH[i,:], color=col[i], label=str(np.asarray(np.round(a[i]/const.au),dtype=int)),linewidth=1.5)

#set limit, scale, and legend
xlimits=np.asarray([4E7,2.5E11])
ax[0].set_ylim(xlimits)
ax[0].set_yscale('log')
leg=ax[0].legend(title=r"Hill radius " "\n " "semi-major axis [au]",loc='lower right',bbox_to_anchor=(0.65,0.01), frameon=False)
leg._legend_box.align = "left"

ax[0].set_ylabel('Radius [m]')
ax[0].set_xlabel('Post-impact body mass [$M_{Earth}$]')

#Make twin axis for showing results in
ax2 = ax[0].twinx()
col=cm.viridis(np.linspace(0,0.95,np.size(gamma)))

for i in np.arange(np.size(gamma)):
    ax2.plot(M/MEarth, RB[i,:]/REarth, ':', color=col[i], label=gas_label[i],linewidth=1.5)
ax2.set_ylim(xlimits/REarth)
ax2.set_yscale('log')
leg2=ax2.legend(title=r"Bondi radius" "\n" "vapor species",loc='lower right',bbox_to_anchor=(0.98,0.01), frameon=False)
leg2._legend_box.align = "left"

ax2.set_ylabel('Radius [$R_{Earth}$]')


#Plot the cooling times
col=cm.inferno([0.0,0.5,0.9])
for i in np.flipud(np.arange(np.size(beta_plot))):
    ax[1].plot(tplot/(24*60*60), (rcool_pl[i,ind_Rp,ind_Mp,:]/Remit)**2,'-', color=col[i], label=str(beta_plot[i]),linewidth=1.5)
for i in np.arange(np.size(beta_plot)):
    ax[1].plot(tplot/(24*60*60), (rcool_pl0[i,ind_Rp,ind_Mp,:]/Remit)**2,'--', color=col[i],linewidth=1.5)

for k in np.flipud(np.arange(np.size(Mp_plot))):
    ax[2].plot(tplot/(24*60*60), (rcool_pl[ind_beta,ind_Rp,k,:]/Remit)**2,'-', color=col[k], label=str(int(Mp_plot[k]/MEarth)),linewidth=1.5)
for k in np.arange(np.size(Mp_plot)):
    ax[2].plot(tplot/(24*60*60), (rcool_pl0[ind_beta,ind_Rp,k,:]/Remit)**2,'--', color=col[k],linewidth=1.5)

ax[1].legend(title="Power law exponent", frameon=False,handlelength=1.4)
ax[2].legend(title="Mass of body [$M_{Earth}$]", frameon=False,handlelength=1.4)

for i in np.arange(1,3):
    ax[i].set_ylabel('Relative flux')
plt.setp( ax[1].get_xticklabels(), visible=False)
ax[2].set_xlabel('Time [days]')

ax[0].text(0.05, 0.95, r"A", horizontalalignment='left',verticalalignment='top', fontsize=8,transform=ax[0].transAxes)
ax[1].text(0.05, 0.05, r"B", horizontalalignment='left',verticalalignment='bottom', fontsize=8,transform=ax[1].transAxes)
ax[2].text(0.05, 0.05, r"C", horizontalalignment='left',verticalalignment='bottom', fontsize=8,transform=ax[2].transAxes)

fig.tight_layout()
pout = 'Size_cooling_combined_figure.pdf'
print(f'Writing plot to {pout}')
plt.savefig(pout)
#plt.show()
