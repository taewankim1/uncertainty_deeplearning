import numpy as np
from scipy.spatial.distance import cdist
def kernel_se(_X1,_X2,_hyp={'gain':1,'len':1,'s2w':1e-8},_use_s2w=False):
    hyp_gain = float(_hyp['gain'])**2
    hyp_len  = 1/float(_hyp['len'])
    if len(_X1.shape)<=1: _X1=_X1.reshape((-1,1))
    if len(_X2.shape)<=1: _X2=_X2.reshape((-1,1))
    pairwise_dists = cdist(_X1,_X2,'euclidean')
    K = hyp_gain*np.exp(-pairwise_dists ** 2/(hyp_len**2))
    if _use_s2w:
        K = K + _hyp['s2w']*np.eye(_X1.shape[0])
    return K

class gpr(object):
    def __init__(self,_xTr,_yTr,_hyp):
        self.xTr = _xTr
        self.yTr = _yTr
        self.hyp = _hyp
        # self.nzrX = nzr(self.xTr)
        # self.nzrY = nzr(self.yTr)
        self.K_TrTr = kernel_se(self.xTr,self.xTr,_hyp=self.hyp,_use_s2w=True)
        self.alpha = np.matmul(np.linalg.inv(self.K_TrTr),self.yTr)
    def inference(self,_xTe):
        self.xTe = _xTe
        self.K_TeTr = kernel_se(self.xTe,self.xTr,_hyp=self.hyp)
        self.K_TeTe = kernel_se(self.xTe,self.xTe,_hyp=self.hyp)
        self.yTe = np.matmul(self.K_TeTr,self.alpha)
        _varTe = self.K_TeTe - np.matmul(np.matmul(self.K_TeTr,
                        np.linalg.inv(self.K_TrTr)),self.K_TeTr.T)
        _varTe = np.diag(_varTe).reshape((-1,1))
        self.sigmaTe = np.squeeze(np.sqrt(_varTe))
        return self.yTe, self.sigmaTe
        
if __name__ == "__main__":
    print ("Kernel function defined.")