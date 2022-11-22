import numpy as np

class mlcEvaluator2:
    def __init__(self, gt=None, pred=None):
        self.confusion_matrix = None
        
        if gt is not None:
            self.setTrueLabels(gt)
        if pred is not None:
            self.setPredictedLabels(pred)
            
    def setTrueLabels(self, gt):
        h = gt.shape[1]
        self.q = h+1     # add NTL row, NLP column
        self.ntlr = h    # NTL row and NPL column index
        self.gt = gt.T
            
    def setPredictedLabels(self, pred):
        if self.gt is None:
            raise Exception('GT labels not povided')
        self.pred = pred.T
        if self.gt.shape != self.pred.shape:
            self.pred = None
            raise Exception('GT labels and predictions do not match')
        
    def getContribution(self, g, p):
        C = np.zeros((self.q, self.q))
        
        if np.sum(g+p) == 0:                    # Empty g & p
            C[self.ntlr, self.ntlr] = 1         # Update NTL, NLP element
            return C
        
        d=np.logical_and(g, p).astype(g.dtype)
        # g1 = p1 = d   (T1 & P1 are the same)
        g2 = g-d     # T2
        p2 = p-d     # P2
        
        C[:self.ntlr,:self.ntlr] = np.diag(d)   # First step
        
        g2c = np.sum(g2)  # Cardinality of T2
        p2c = np.sum(p2)  # Cardinality of P2
        
        if g2c > 0 and p2c == 0:              # CATEGORY 1
            C[:self.ntlr,self.ntlr] += g2
        elif g2c == 0 and p2c > 0:            # CATEGORY 2
            if np.sum(g) == 0:                # Empty T
                C[self.ntlr,:self.ntlr] += p2 # Increment NTL row
            else:
                C[:self.ntlr,:self.ntlr] += np.outer(g, p2)
        elif g2c>0 and p2c > 0:               # CATEGORY 3
            C[:self.ntlr,:self.ntlr] += np.outer(g2, p2)

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

    def getInstanceContribution(self, i):
        g = self.gt[:,i]
        p = self.pred[:,i]
        return self.getContribution(g, p)

    def getOneVsRest(self):
        vvr = np.zeros((self.q, 2, 2))
        vvr[:,1,1] = self.confusion_matrix.diagonal()
        vvr[:,0,0] = vvr[:,1,1].sum() - vvr[:,1,1] 
        vvr[:,0,1] = self.confusion_matrix.sum(axis=0) - vvr[:,1,1]
        vvr[:,1,0] = self.confusion_matrix.sum(axis=1) - vvr[:,1,1]
        return vvr
       
