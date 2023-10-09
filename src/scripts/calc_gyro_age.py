import pickle
import paths

# calculate dictionary of summary statistics
from gyrointerp import get_summary_statistics

fin = 'gyro_age_posterior.pkl'

with open(paths.data / fin, "rb") as f:
    (age_grid, age_posterior) = pickle.load(f)

result = get_summary_statistics(age_grid, age_posterior)

print(f"Age = {result['median']} +{result['+1sigma']} -{result['-1sigma']} Myr.")

fstr = f"${result['median']:3.0f}\pm{result['+1sigma']:2.0f}$ Myr"

print(fstr)

with open(paths.output / 'gyro_age.txt', 'w') as f:
    f.write(fstr)
