import numpy as np

class mlcEvaluator1:
    def __init__(self, gt=None, pred=None, use_unknown=True):
        self.use_unknown = use_unknown
        self.confusion_matrix = None
        
        if gt is not None:
            self.setTrueLabels(gt)
        if pred is not None:
            self.setPredictedLabels(pred)
            
    def setTrueLabels(self, gt):
        self.q = gt.shape[1]
        if self.use_unknown:
            self.gt = self.addUnknown(gt.T)
            self.q += 1
        else:
            self.gt = gt.T
            
    def setPredictedLabels(self, pred):
        if self.gt is None:
            raise Exception('GT labels not povided')

        if self.use_unknown:
            self.pred = self.addUnknown(pred.T)
        else:
            self.pred = pred.T
            
        if self.gt.shape != self.pred.shape:
            self.pred = None
            raise Exception('GT labels and predictions do not match')
    
    def addUnknown(self, m):
        h, w = m.shape
        mu = np.zeros((h+1,w))
        mu[:h,:] = m 
        mu[h,:] = (np.sum(mu, axis=0) == 0).astype(float)
        return mu
    
    def getContribution(self, g, p):
        if np.array_equal(g, p):
            return np.diag(g)
        
        gc = np.sum(g)
        pc = np.sum(p)
        
        d=np.logical_and(g, p).astype(g.dtype)
        
        gd = g-d
        pd = p-d
        
        gdc = np.sum(gd)
        pdc = np.sum(pd)
        
        if gdc == 0:
            gd += d
            C = (np.outer(gd, pd) + gc*np.diag(d))/pc
        elif pdc == 0:
            pd += d
            C = np.outer(gd, pd)/pc + np.diag(d)
        else:
            C = np.outer(gd, pd)/pdc + np.diag(d)
            
        return C

    def computeConfusionMatrix(self, gt=None, pred=None):
        if gt is not None:
            self.setTrueLabels(gt)
        if pred is not None:
            self.setPredictedLabels(pred)
            
        self.confusion_matrix = np.zeros((self.q,self.q))
        nsamples = self.gt.shape[1]
        for k in range(nsamples):
            g = self.gt[:,k]
            p = self.pred[:,k]
            self.confusion_matrix += self.getContribution(g, p)

        return self.confusion_matrix
    
    def getConfusionMatrix(self):
        return self.confusion_matrix

    def getPrecisionByClass(self):
        if self.confusion_matrix is None:
            return None
        p = self.confusion_matrix.diagonal()/self.confusion_matrix.sum(axis=0)
        return p

    def getRecallByClass(self):
        if self.confusion_matrix is None:
            return None
        r = self.confusion_matrix.diagonal()/self.confusion_matrix.sum(axis=1)
        return r
    
    def getOneVsRest(self):
        vvr = np.zeros((self.q, 2, 2))
        vvr[:,1,1] = self.confusion_matrix.diagonal()
        vvr[:,0,0] = vvr[:,1,1].sum() - vvr[:,1,1] 
        vvr[:,0,1] = self.confusion_matrix.sum(axis=0) - vvr[:,1,1]
        vvr[:,1,0] = self.confusion_matrix.sum(axis=1) - vvr[:,1,1]
        return vvr

            
        
