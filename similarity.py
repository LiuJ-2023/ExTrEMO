from Problems import dtlz
from scipy.stats import qmc
from Utils import decomposition
import numpy as np
import matplotlib.pyplot as plt

# P1 = dtlz.DTLZ1(n_var=10)
# pho_all = []
# for i in range(1000):
#     delta1 = 1
#     delta2 = 0.1*np.random.rand() + 0.2
#     delta3 = 1
#     P2 = dtlz.DTLZ1(n_var=10,delta1=delta1, delta2=delta2, delta3=delta3)
#     # P2 = dtlz.DTLZ5()
    
#     sampler = qmc.LatinHypercube(d=10)
#     x_t = sampler.random(n=1000)*(P1.standard_bounds[1] - P1.standard_bounds[0]) + P1.standard_bounds[0]
#     y1 = P1(x_t)
#     y2 = P2(x_t)

#     lambda_ = np.array([0.2500249925022493, 0.49995001499550135, 0.2500249925022493])

#     norm_y1 = decomposition.normalize(P1.obj_num, y1)[0]
#     yT_1 = decomposition.scalarize_tchebycheff(norm_y1, lambda_, augmented=True) 
#     yT_1 = (yT_1 - yT_1.mean(0)) / yT_1.std(0)

#     norm_y2 = decomposition.normalize(P2.obj_num, y2)[0]
#     yT_2 = decomposition.scalarize_tchebycheff(norm_y2, lambda_, augmented=True)
#     yT_2 = (yT_2 - yT_2.mean(0)) / yT_2.std(0) 

#     pho_all.append(np.mean((yT_1 - yT_1.mean(0))*(yT_2 - yT_2.mean(0))/(yT_1.std(0)*yT_2.std(0))))

# pho_mean = np.mean(np.array(pho_all))
# print(str(pho_mean))

P1 = dtlz.DTLZ1(n_var=10)
P2 = dtlz.DTLZ1(n_var=10,delta1=1,delta2=0)
P3 = dtlz.DTLZ1(n_var=10,delta1=-1,delta2=0)
P4 = dtlz.DTLZ1(n_var=10,delta1=1,delta2=0.4)
sampler = qmc.LatinHypercube(d=10)
x_t1 = sampler.random(n=10000)*(P1.standard_bounds[1] - P1.standard_bounds[0]) + P1.standard_bounds[0]
y1 = P1(x_t1)
y2 = P2(x_t1)
y3 = P3(x_t1)
y4 = P4(x_t1)

lambda_ = np.array([0.2500249925022493, 0.49995001499550135, 0.2500249925022493])
norm_y1 = decomposition.normalize(P1.obj_num, y1)[0]
# norm_y1 = y1
yT_1 = decomposition.scalarize_tchebycheff(norm_y1, lambda_, augmented=True) 
yT_1 = (yT_1 - yT_1.mean(0)) / yT_1.std(0)

norm_y2 = decomposition.normalize(P2.obj_num, y2)[0]
# norm_y2 = y2
yT_2 = decomposition.scalarize_tchebycheff(norm_y2, lambda_, augmented=True)
yT_2 = (yT_2 - yT_2.mean(0)) / yT_2.std(0) 

norm_y3 = decomposition.normalize(P3.obj_num, y3)[0]
# norm_y3 = y3
yT_3 = decomposition.scalarize_tchebycheff(norm_y3, lambda_, augmented=True) 
yT_3 = (yT_3 - yT_3.mean(0)) / yT_3.std(0)

norm_y4 = decomposition.normalize(P4.obj_num, y4)[0]
# norm_y4 = y4
yT_4 = decomposition.scalarize_tchebycheff(norm_y4, lambda_, augmented=True)
yT_4 = (yT_4 - yT_4.mean(0)) / yT_4.std(0) 

pho_1 = np.mean((yT_1 - yT_1.mean(0))*(yT_2 - yT_2.mean(0)))/(np.std(yT_1)*np.std(yT_2))
pho_2 = np.mean((yT_1 - yT_1.mean(0))*(yT_3 - yT_3.mean(0)))/(np.std(yT_1)*np.std(yT_3))
pho_3 = np.mean((yT_1 - yT_1.mean(0))*(yT_4 - yT_4.mean(0)))/(np.std(yT_1)*np.std(yT_4))

print([pho_1,pho_2,pho_3])

# xx = ['0.05','0.2','0.4']
# yy = [pho_1,pho_2,pho_3]
xx = ['-1','1']
yy = [pho_2,pho_1]
plt.bar(xx,yy,color='darkgrey',width=0.5)
plt.plot((1.9+0.9)*np.array(range(500)) - 0.9,np.zeros(500), color='darkgrey')

# plt.legend(prop = {'size':12})
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
plt.xlim([-0.9,1.9])
plt.xlabel('Parameter Settings of $\delta_1$',fontsize = 16)
plt.ylabel('Pearson Correlation Coefficients',fontsize = 16)
plt.title('DTLZ1b-($\delta_1,0,1$)',fontsize = 16)
plt.show()