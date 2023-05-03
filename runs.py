from ExTrEMO import ExTrEMO
from ParEGO import ParEGO
from Problems import dtlz
import pickle
import numpy as np
import os
from Utils.cal_hv import cal_hv
import matplotlib.pyplot as plt

os.environ['KMP_DUPLICATE_LIB_OK']='True'
test_problems = [dtlz.DTLZ1(n_var=10),dtlz.DTLZ2(n_var=10),dtlz.DTLZ3(n_var=10),dtlz.DTLZ4(n_var=10)]
strpros = ["DTLZ1","DTLZ2","DTLZ3","DTLZ4"]

for pb in [1]:
    test_problem = test_problems[pb]
    strpro = strpros[pb]

    MOEA_ExTrEMO = []
    x_ExTrEMO = []
    y_ExTrEMO = []

    MOEA_ParEGO = []
    x_ParEGO = []
    y_ParEGO = []
    for i in range(20):
        # Load source datasets
        dataS = pickle.load(open("Data\\source_data_" + strpro + ".p", "rb"))

        # Run ExTrEMO
        xT_ExTrEMO,yT_ExTrEMO,Sel_Task_ES = ExTrEMO(test_problem, 
            dataS[0][10:11]+dataS[0][210:211]+dataS[0][410:411]+dataS[0][610:611]+dataS[0][810:811], 
            dataS[1][10:11]+dataS[1][210:211]+dataS[1][410:411]+dataS[1][610:611]+dataS[1][810:811], 
            dataS[0][0], 
            dataS[1][0], 
            max_iters = 30, 
            likelihood = 'gaussian', 
            kernel = 'RBF',
            opt_train='l-bfgs',
            opt_acqf='ga',
            subset_sel = None,
            )
        hv_ExTrEMO = cal_hv(yT_ExTrEMO, test_problem.norm_for_hv)
        MOEA_ExTrEMO.append(hv_ExTrEMO)
        x_ExTrEMO.append(xT_ExTrEMO)
        y_ExTrEMO.append(yT_ExTrEMO)
        print(hv_ExTrEMO)
        print(str(i))


        # Run ParEGO
        xT_ParEGO,yT_ParEGO = ParEGO(test_problem, 
            dataS[0][0], 
            dataS[1][0], 
            max_iters = 30, 
            likelihood = 'gaussian', 
            kernel = 'RBF',
            opt_train='l-bfgs',
            opt_acqf='ga',
            )
        hv_ParEGO = cal_hv(yT_ParEGO, test_problem.norm_for_hv)
        MOEA_ParEGO.append(hv_ParEGO)
        x_ParEGO.append(xT_ParEGO)
        y_ParEGO.append(yT_ParEGO)
        print(hv_ParEGO)
        print(str(i))

    mean_hv_ExTrEMO = np.mean( np.array(MOEA_ExTrEMO), axis=0 )
    std_hv_ExTrEMO = np.std( np.array(MOEA_ExTrEMO), axis=0 )
    mean_hv_parego = np.mean( np.array(MOEA_ParEGO), axis=0 )
    std_hv_parego = np.std( np.array(MOEA_ParEGO), axis=0 )

    # dataExTrEMO = [mean_hv_ExTrEMO,std_hv_ExTrEMO,MOEA_ExTrEMO,x_ExTrEMO,y_ExTrEMO]
    # pickle.dump(dataExTrEMO, open("Data//ExTrEMO_" + strpro + "___5sources_20samples_20runs.p", "wb"))
    # dataParEGO = [mean_hv_parego,std_hv_parego,MOEA_ParEGO,x_ParEGO,y_ParEGO]
    # pickle.dump(dataParEGO, open("Data//ParEGO_" + strpro + "___.p", "wb"))


plt.plot(mean_hv_ExTrEMO,label = 'ExTrEMO')
plt.plot(mean_hv_parego,label = 'ParEGO')
plt.legend()
plt.show()