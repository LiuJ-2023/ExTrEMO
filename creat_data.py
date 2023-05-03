from Problems.dtlz import DTLZ1, DTLZ2, DTLZ3, DTLZ4
from scipy.stats import norm, qmc
import pickle
import numpy as np
from scipy.stats import norm, qmc
import numpy as np
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

# Evaluate Source Data
def generate_init_samples_ori(x, f_obj):
    F = []
    F = f_obj(x)
    return F

dim = 10
################# Source data for DTLZ1 #################
sampler = qmc.LatinHypercube(d=dim)
gsize = 100
tsize = 20
PT = DTLZ1(n_var=dim)
x_t = sampler.random(n=tsize)*(PT.standard_bounds[1] - PT.standard_bounds[0]) + PT.standard_bounds[0]
F_t = generate_init_samples_ori(x_t, PT)
dataSX = [x_t]
dataSY = [F_t]
for j in range(200):
    delta1 = 1
    delta2 = 0.1*np.random.rand() - 0.05
    PH = DTLZ1(n_var = dim, delta1 = delta1, delta2 = delta2)
    x_s = sampler.random(n=gsize)*(PH.standard_bounds[1] - PH.standard_bounds[0]) + PH.standard_bounds[0]
    F_s = generate_init_samples_ori(x_s, PH)
    dataSX.append(x_s)
    dataSY.append(F_s)

for j in range(200):
    delta1 = 1
    delta2 = 0.1*np.random.rand() + 0.2
    PH = DTLZ1(n_var = dim, delta1 = delta1, delta2 = delta2)
    x_s = sampler.random(n=gsize)*(PH.standard_bounds[1] - PH.standard_bounds[0]) + PH.standard_bounds[0]
    F_s = generate_init_samples_ori(x_s, PH)
    dataSX.append(x_s)
    dataSY.append(F_s)

for j in range(200):
    delta2 = 0.1*np.random.rand() + 0.4
    delta3 = 0
    PH = DTLZ1(n_var = dim, delta2 = delta2, delta3 = delta3)
    x_s = sampler.random(n=gsize)*(PH.standard_bounds[1] - PH.standard_bounds[0]) + PH.standard_bounds[0]
    F_s = generate_init_samples_ori(x_s, PH)
    dataSX.append(x_s)
    dataSY.append(F_s)

for j in range(200):
    delta1 = -1
    delta2 = 0.1*np.random.rand() + 0.2
    PH = DTLZ1(n_var = dim, delta1 = delta1, delta2 = delta2)
    x_s = sampler.random(n=gsize)*(PH.standard_bounds[1] - PH.standard_bounds[0]) + PH.standard_bounds[0]
    F_s = generate_init_samples_ori(x_s, PH)
    dataSX.append(x_s)
    dataSY.append(F_s)

for j in range(200):
    delta1 = -1
    delta2 = 0.1*np.random.rand() - 0.05
    PH = DTLZ1(n_var = dim, delta1 = delta1, delta2 = delta2)
    x_s = sampler.random(n=gsize)*(PH.standard_bounds[1] - PH.standard_bounds[0]) + PH.standard_bounds[0]
    F_s = generate_init_samples_ori(x_s, PH)
    dataSX.append(x_s)
    dataSY.append(F_s)

dataS = [dataSX,dataSY]

pickle.dump(dataS, open("Data\\source_data_DTLZ1.p", "wb"))

################# Source data for DTLZ2 #################
sampler = qmc.LatinHypercube(d=dim)
gsize = 100
tsize = 20
PT = DTLZ2(n_var=dim)
x_t = sampler.random(n=tsize)*(PT.standard_bounds[1] - PT.standard_bounds[0]) + PT.standard_bounds[0]
F_t = generate_init_samples_ori(x_t, PT)
dataSX = [x_t]
dataSY = [F_t]
for j in range(200):
    delta1 = 1
    delta2 = 0.1*np.random.rand() - 0.05
    PH = DTLZ2(n_var = dim, delta1 = delta1, delta2 = delta2)
    x_s = sampler.random(n=gsize)*(PH.standard_bounds[1] - PH.standard_bounds[0]) + PH.standard_bounds[0]
    F_s = generate_init_samples_ori(x_s, PH)
    dataSX.append(x_s)
    dataSY.append(F_s)

for j in range(200):
    delta1 = 1
    delta2 = 0.1*np.random.rand() + 0.3
    PH = DTLZ2(n_var = dim, delta1 = delta1, delta2 = delta2)
    x_s = sampler.random(n=gsize)*(PH.standard_bounds[1] - PH.standard_bounds[0]) + PH.standard_bounds[0]
    F_s = generate_init_samples_ori(x_s, PH)
    dataSX.append(x_s)
    dataSY.append(F_s)

for j in range(200):
    delta2 = 0.1*np.random.rand() + 0.4
    delta3 = 0
    PH = DTLZ2(n_var = dim, delta2 = delta2, delta3 = delta3)
    x_s = sampler.random(n=gsize)*(PH.standard_bounds[1] - PH.standard_bounds[0]) + PH.standard_bounds[0]
    F_s = generate_init_samples_ori(x_s, PH)
    dataSX.append(x_s)
    dataSY.append(F_s)

for j in range(200):
    delta1 = -1
    delta2 = 0.1*np.random.rand() + 0.3
    PH = DTLZ2(n_var = dim, delta1 = delta1, delta2 = delta2)
    x_s = sampler.random(n=gsize)*(PH.standard_bounds[1] - PH.standard_bounds[0]) + PH.standard_bounds[0]
    F_s = generate_init_samples_ori(x_s, PH)
    dataSX.append(x_s)
    dataSY.append(F_s)

for j in range(200):
    delta1 = -1
    delta2 = 0.1*np.random.rand() - 0.05
    PH = DTLZ2(n_var = dim, delta1 = delta1, delta2 = delta2)
    x_s = sampler.random(n=gsize)*(PH.standard_bounds[1] - PH.standard_bounds[0]) + PH.standard_bounds[0]
    F_s = generate_init_samples_ori(x_s, PH)
    dataSX.append(x_s)
    dataSY.append(F_s)

dataS = [dataSX,dataSY]

pickle.dump(dataS, open("Data\\source_data_DTLZ2.p", "wb"))

################# Source data for DTLZ3 #################
sampler = qmc.LatinHypercube(d=dim)
gsize = 100
tsize = 20
PT = DTLZ3(n_var=dim)
x_t = sampler.random(n=tsize)*(PT.standard_bounds[1] - PT.standard_bounds[0]) + PT.standard_bounds[0]
F_t = generate_init_samples_ori(x_t, PT)
dataSX = [x_t]
dataSY = [F_t]
for j in range(200):
    delta1 = 1
    delta2 = 0.1*np.random.rand() - 0.05
    PH = DTLZ3(n_var = dim, delta1 = delta1, delta2 = delta2)
    x_s = sampler.random(n=gsize)*(PH.standard_bounds[1] - PH.standard_bounds[0]) + PH.standard_bounds[0]
    F_s = generate_init_samples_ori(x_s, PH)
    dataSX.append(x_s)
    dataSY.append(F_s)

for j in range(200):
    delta1 = 1
    delta2 = 0.1*np.random.rand() + 0.2
    PH = DTLZ3(n_var = dim, delta1 = delta1, delta2 = delta2)
    x_s = sampler.random(n=gsize)*(PH.standard_bounds[1] - PH.standard_bounds[0]) + PH.standard_bounds[0]
    F_s = generate_init_samples_ori(x_s, PH)
    dataSX.append(x_s)
    dataSY.append(F_s)

for j in range(200):
    delta2 = 0.1*np.random.rand() + 0.4
    delta3 = 0
    PH = DTLZ3(n_var = dim, delta2 = delta2, delta3 = delta3)
    x_s = sampler.random(n=gsize)*(PH.standard_bounds[1] - PH.standard_bounds[0]) + PH.standard_bounds[0]
    F_s = generate_init_samples_ori(x_s, PH)
    dataSX.append(x_s)
    dataSY.append(F_s)

for j in range(200):
    delta1 = -1
    delta2 = 0.1*np.random.rand() + 0.2
    PH = DTLZ3(n_var = dim, delta1 = delta1, delta2 = delta2)
    x_s = sampler.random(n=gsize)*(PH.standard_bounds[1] - PH.standard_bounds[0]) + PH.standard_bounds[0]
    F_s = generate_init_samples_ori(x_s, PH)
    dataSX.append(x_s)
    dataSY.append(F_s)

for j in range(200):
    delta1 = -1
    delta2 = 0.1*np.random.rand() - 0.05
    PH = DTLZ3(n_var = dim, delta1 = delta1, delta2 = delta2)
    x_s = sampler.random(n=gsize)*(PH.standard_bounds[1] - PH.standard_bounds[0]) + PH.standard_bounds[0]
    F_s = generate_init_samples_ori(x_s, PH)
    dataSX.append(x_s)
    dataSY.append(F_s)

dataS = [dataSX,dataSY]

pickle.dump(dataS, open("Data\\source_data_DTLZ3.p", "wb"))

################# Source data for DTLZ4 #################
sampler = qmc.LatinHypercube(d=dim)
gsize = 100
tsize = 20
PT = DTLZ4(n_var=dim)
x_t = sampler.random(n=tsize)*(PT.standard_bounds[1] - PT.standard_bounds[0]) + PT.standard_bounds[0]
F_t = generate_init_samples_ori(x_t, PT)
dataSX = [x_t]
dataSY = [F_t]
for j in range(200):
    delta1 = 1
    delta2 = 0.1*np.random.rand() - 0.05
    PH = DTLZ4(n_var = dim, delta1 = delta1, delta2 = delta2)
    x_s = sampler.random(n=gsize)*(PH.standard_bounds[1] - PH.standard_bounds[0]) + PH.standard_bounds[0]
    F_s = generate_init_samples_ori(x_s, PH)
    dataSX.append(x_s)
    dataSY.append(F_s)

for j in range(200):
    delta1 = 1
    delta2 = 0.1*np.random.rand() + 0.3
    PH = DTLZ4(n_var = dim, delta1 = delta1, delta2 = delta2)
    x_s = sampler.random(n=gsize)*(PH.standard_bounds[1] - PH.standard_bounds[0]) + PH.standard_bounds[0]
    F_s = generate_init_samples_ori(x_s, PH)
    dataSX.append(x_s)
    dataSY.append(F_s)

for j in range(200):
    delta2 = 0.1*np.random.rand() + 0.4
    delta3 = 0
    PH = DTLZ4(n_var = dim, delta2 = delta2, delta3 = delta3)
    x_s = sampler.random(n=gsize)*(PH.standard_bounds[1] - PH.standard_bounds[0]) + PH.standard_bounds[0]
    F_s = generate_init_samples_ori(x_s, PH)
    dataSX.append(x_s)
    dataSY.append(F_s)

for j in range(200):
    delta1 = -1
    delta2 = 0.1*np.random.rand() + 0.3
    PH = DTLZ4(n_var = dim, delta1 = delta1, delta2 = delta2)
    x_s = sampler.random(n=gsize)*(PH.standard_bounds[1] - PH.standard_bounds[0]) + PH.standard_bounds[0]
    F_s = generate_init_samples_ori(x_s, PH)
    dataSX.append(x_s)
    dataSY.append(F_s)

for j in range(200):
    delta1 = -1
    delta2 = 0.1*np.random.rand() - 0.05
    PH = DTLZ4(n_var = dim, delta1 = delta1, delta2 = delta2)
    x_s = sampler.random(n=gsize)*(PH.standard_bounds[1] - PH.standard_bounds[0]) + PH.standard_bounds[0]
    F_s = generate_init_samples_ori(x_s, PH)
    dataSX.append(x_s)
    dataSY.append(F_s)

dataS = [dataSX,dataSY]

pickle.dump(dataS, open("Data\\source_data_DTLZ4.p", "wb"))