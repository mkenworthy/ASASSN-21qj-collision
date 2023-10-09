import numpy as np
from gyrointerp import gyro_age_posterior
import paths
from asassn21qj import *

# units: days
Prot, Prot_err = 4.43, 0.33

# units: kelvin
Teff, Teff_err = 5900, 74

print(f'Running gyrointerp with P={Prot}+-{Prot_err} days and T={Teff}+-{Teff_err} K')
# uniformly spaced grid between 0 and 2600 megayears
age_grid = np.linspace(0, 2600, 500)

# calculate the age posterior - takes ~30 seconds
age_posterior = gyro_age_posterior(
    Prot, Teff, Prot_err=Prot_err, Teff_err=Teff_err, age_grid=age_grid
)

import pickle

fout = 'gyro_age_posterior.pkl'
print(f'writing out age_posterior to {fout}')
with open(paths.data / 'gyro_age_posterior.pkl', 'wb') as gyro:
  pickle.dump([age_grid, age_posterior], gyro)