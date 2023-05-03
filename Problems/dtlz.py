import numpy as np

class DTLZ1():
    def __init__(self, n_var = 10, delta1 = 1, delta2 = 0, delta3 = 1):
        self.dim = n_var
        self.obj_num = 3
        self.standard_bounds = np.array([ np.zeros(n_var), np.ones(n_var) ])
        self.norm_for_hv = np.array([[0,0,0],[1000,1000,1000]])
        self.delta1 = delta1
        self.delta2 = delta2
        self.delta3 = delta3
    
    def __call__(self, x):
        if x.ndim == 1:
            x = np.array([x])
        elif x.ndim >= 3:
            Exception("The dimension of the input array should be small than 3.")

        M = 3
        g = 100*self.delta3*(8 + np.sum( (x[:,M-1:] - 0.5 - self.delta2 )**2 - np.cos(2*np.pi*(x[:,M-1:] - 0.5 - self.delta2)), axis=1 )) + (1-self.delta3)*(-20*np.exp(-0.2*np.sqrt( np.mean( ((x[:,M-1:] -0.5 - self.delta2)*50)**2, axis=1 ) )) - np.exp( np.mean( np.cos(2*np.pi*(x[:,M-1:] - 0.5 - self.delta2)*50) , axis=1 ) ) + 20 + np.e)

        f1 = 0.5*self.delta1*self.delta3*x[:,0]*x[:,1]*(1+g) + (1-self.delta3)*(1+g)*np.cos(x[:,0]*np.pi/2)*np.cos(x[:,1]*np.pi/2)
        f2 = 0.5*self.delta1*self.delta3*x[:,0]*(1-x[:,1])*(1+g) + (1-self.delta3)*(1+g)*np.cos(x[:,0]*np.pi/2)*np.sin(x[:,1]*np.pi/2)
        f3 = 0.5*self.delta1*self.delta3*(1-x[:,0])*(1+g) + (1-self.delta3)*(1+g)*np.sin(x[:,0]*np.pi/2)
        f = np.array([f1,f2,f3]).T
        return f

class DTLZ2():
    def __init__(self, n_var = 10, delta1 = 1, delta2 = 0, delta3 = 1):
        self.dim = n_var
        self.obj_num = 3
        self.standard_bounds = np.array([ np.zeros(n_var), np.ones(n_var) ])
        self.norm_for_hv = np.array([[0,0,0],[100,100,100]])
        self.delta1 = delta1
        self.delta2 = delta2
        self.delta3 = delta3

    def __call__(self, x):
        if x.ndim == 1:
            x = np.array([x])
        elif x.ndim >= 3:
            Exception("The dimension of the input array should be small than 3.")

        M = 3
        g = 100*np.sum( (x[:,M-1:] - 0.5 - self.delta2 )**2, axis=1 ) + (1-self.delta3)*(-20*np.exp(-0.2*np.sqrt( np.mean( ((x[:,M-1:] - 0.5 - self.delta2)*50)**2, axis=1 ) )) - np.exp( np.mean( np.cos(2*np.pi*(x[:,M-1:] - 0.5 - self.delta2)*50) , axis=1 ) ) + 20 + np.e)

        f1 = (1+g)*self.delta1*self.delta3*np.cos(x[:,0]*np.pi/2)*np.cos(x[:,1]*np.pi/2) + 0.5*(1 - self.delta3)*x[:,0]*x[:,1]*(1+g)
        f2 = (1+g)*self.delta1*self.delta3*np.cos(x[:,0]*np.pi/2)*np.sin(x[:,1]*np.pi/2) + 0.5*(1 - self.delta3)*x[:,0]*(1-x[:,1])*(1+g)
        f3 = (1+g)*self.delta1*self.delta3*np.sin(x[:,0]*np.pi/2) + 0.5*(1 - self.delta3)*(1-x[:,0])*(1+g)
        f = np.array([f1,f2,f3]).T
        return f

class DTLZ3():
    def __init__(self, n_var = 10, delta1 = 1, delta2 = 0, delta3 = 1):
        self.dim = n_var
        self.obj_num = 3
        self.standard_bounds = np.array([ np.zeros(n_var), np.ones(n_var) ])
        self.norm_for_hv = np.array([[0,0,0],[1000,1000,1000]])
        self.delta1 = delta1
        self.delta2 = delta2
        self.delta3 = delta3
    
    def __call__(self, x):
        if x.ndim == 1:
            x = np.array([x])
        elif x.ndim >= 3:
            Exception("The dimension of the input array should be small than 3.")

        M = 3
        g = 100*self.delta3*(8 + np.sum( (x[:,M-1:] - 0.5 - self.delta2 )**2 - np.cos(2*np.pi*(x[:,M-1:] - 0.5 - self.delta2)), axis=1 )) + (1-self.delta3)*(-20*np.exp(-0.2*np.sqrt( np.mean( ((x[:,M-1:] -0.5 - self.delta2)*50)**2, axis=1 ) )) - np.exp( np.mean( np.cos(2*np.pi*(x[:,M-1:] - 0.5 - self.delta2)*50) , axis=1 ) ) + 20 + np.e)

        f1 = (1+g)*self.delta1*self.delta3*np.cos(x[:,0]*np.pi/2)*np.cos(x[:,1]*np.pi/2) + 0.5*(1 - self.delta3)*x[:,0]*x[:,1]*(1+g)
        f2 = (1+g)*self.delta1*self.delta3*np.cos(x[:,0]*np.pi/2)*np.sin(x[:,1]*np.pi/2) + 0.5*(1 - self.delta3)*x[:,0]*(1-x[:,1])*(1+g)
        f3 = (1+g)*self.delta1*self.delta3*np.sin(x[:,0]*np.pi/2) + 0.5*(1 - self.delta3)*(1-x[:,0])*(1+g)
        f = np.array([f1,f2,f3]).T
        return f

class DTLZ4():
    def __init__(self, n_var = 10, delta1 = 1, delta2 = 0, delta3 = 1):
        self.dim = n_var
        self.obj_num = 3
        self.standard_bounds = np.array([ np.zeros(n_var), np.ones(n_var) ])
        self.norm_for_hv = np.array([[0,0,0],[100,100,100]])
        self.delta1 = delta1
        self.delta2 = delta2
        self.delta3 = delta3
    
    def __call__(self, x):
        if x.ndim == 1:
            x = np.array([x])
        elif x.ndim >= 3:
            Exception("The dimension of the input array should be small than 3.")
            
        M = 3
        g = 100*np.sum( (x[:,M-1:] - 0.5 - self.delta2 )**2, axis=1 ) + (1-self.delta3)*(-20*np.exp(-0.2*np.sqrt( np.mean( ((x[:,M-1:] - 0.5 - self.delta2)*50)**2, axis=1 ) )) - np.exp( np.mean( np.cos(2*np.pi*(x[:,M-1:] - 0.5 - self.delta2)*50) , axis=1 ) ) + 20 + np.e)

        f1 = (1+g)*self.delta1*self.delta3*np.cos((x[:,0]**5)*np.pi/2)*np.cos((x[:,1]**5)*np.pi/2) + 0.5*(1 - self.delta3)*x[:,0]*x[:,1]*(1+g)
        f2 = (1+g)*self.delta1*self.delta3*np.cos((x[:,0]**5)*np.pi/2)*np.sin((x[:,1]**5)*np.pi/2) + 0.5*(1 - self.delta3)*x[:,0]*(1-x[:,1])*(1+g)
        f3 = (1+g)*self.delta1*self.delta3*np.sin((x[:,0]**5)*np.pi/2) + 0.5*(1 - self.delta3)*(1-x[:,0])*(1+g)
        f = np.array([f1,f2,f3]).T
        return f