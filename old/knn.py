import numpy as np
import numpy as np
import scipy.stats as ss

######################################################################
### class KNN
######################################################################

class KNN(object):
    
    def __init__(self):
        self.Xmeans = None
        self.Xstds = None
        self.X = None  # data will be stored here
        self.T = None  # class labels will be stored here

    def _standardizeX(self,X):
        # if self.Xmeans is not None:
        #     result = (X - self.Xmeans) / self.XstdsFixed
        #     result[:,self.Xconstant] = 0.0
        # else:
        #     result = X
        # return result
        return X
    
    def train(self,X,T):
        if self.Xmeans is None:
            self.Xmeans = X.mean(axis=0)
            self.Xstds = X.std(axis=0)
            # self.Xconstant = (self.Xstds == 0).reshape((1,-1))
            self.Xconstant = self.Xstds == 0
            self.XstdsFixed = self.Xstds
            self.XstdsFixed[self.Xconstant] = 1
        self.X = self._standardizeX(X)
        self.T = T

    def use(self,Xnew,k = 1):
        self.k = k
        # Calc squared distance from all samples in Xnew with all stored in self.X
        distsSquared = np.sum( (self._standardizeX(Xnew)[:,np.newaxis,:] - self.X)**2, axis=-1 )
        indices = np.argsort(distsSquared,axis=1)[:,:k]
        classes = ss.mode(T[indices,:][:,:,0], axis=-1)[0]
        return classes
