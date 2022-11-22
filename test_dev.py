import numpy as np
from mlcevaluator1 import mlcEvaluator1
from mlcevaluator2 import mlcEvaluator2
from sklearn.metrics import multilabel_confusion_matrix

gt=np.asarray([[1,1,0], [1,1,1], [0,0,0],
               [1,0,0], [1,1,0], [0,0,0],
               [1,0,0], [1,1,0], [1,1,0]])
              
pred=np.asarray([[1,1,0],[1,0,1],[0,0,0],
                 [1,1,1], [1,1,1], [0,1,1],
                 [0,1,1], [1,0,1], [0,0,1]])


# Evaluator1, using category unknown
evalD=mlcEvaluator1(gt, pred, use_unknown=True)
MD=evalD.computeConfusionMatrix()

# Evaluator1, without category unknown
evalD1=mlcEvaluator1(gt, pred, use_unknown=False)
MD1=evalD1.computeConfusionMatrix()

# Evaluator 2
evalO=mlcEvaluator2(gt, pred)
MO = evalO.computeConfusionMatrix()

# DEV
# Compute contribution of a particular data instance


# Contribution of the instance 0
# g=[1,1,0]; p=[1,1,0]
# CATEGORY: (i) in [1], (I) in [2] 
evalD1.getInstanceContribution(0)
evalD.getInstanceContribution(0)
evalO.getInstanceContribution(0)

# Contribution of the instance 1
# g=[1,1,1]; p=[1,0,1]
# CATEGORY: (iii) in [1], (I) in [2] 
evalD1.getInstanceContribution(1)
evalD.getInstanceContribution(1)
evalO.getInstanceContribution(1)

# Contribution of the instance 2
# g=[0,0,0]; p=[0,0,0]
# CATEGORY: (i) in [1], (I) in [2] 
evalD1.getInstanceContribution(2)
evalD.getInstanceContribution(2)
evalO.getInstanceContribution(2)

# Contribution of the instance 5
# g=[0,0,0]; p=[0,1,1]
# CATEGORY: (ii) in [1], (II) in [2] 
evalD1.getInstanceContribution(5)
evalD.getInstanceContribution(5)
evalO.getInstanceContribution(5)

# Contribution of the instance 7
# g=[1,1,0]; p=[1,0,1]
# CATEGORY: (iv) in [1], (III) in [2] 
evalD1.getInstanceContribution(7)
evalD.getInstanceContribution(7)
evalO.getInstanceContribution(7)

