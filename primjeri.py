import numpy as np
from mlcevaluator import mlcEvaluator
from mlcevaluatorMTR import mlcEvaluatorMTR
from sklearn.metrics import multilabel_confusion_matrix

gt=np.asarray([[1,1,0], [1,1,1], [0,0,0],
               [1,0,0], [1,1,0], [0,0,0],
               [1,0,0], [1,1,0], [1,1,0]])
              
pred=np.asarray([[1,1,0],[1,0,1],[0,0,0],
                 [1,1,1], [1,1,1], [0,1,1],
                 [0,1,1], [1,0,1], [0,0,1]])


evalD=mlcEvaluator(gt, pred, use_unknown=True)
MD=evalD.computeConfusionMatrix()

evalO=mlcEvaluatorMTR(gt, pred)
MO = evalO.computeConfusionMatrix()

MS = multilabel_confusion_matrix(gt, pred)


# Testiranje - razvoj
g=gt[:,0]
p=pred[:,0]
evalO.getContribution(g, p)

MOS[:,1,1] = MO.diagonal()
MOS[:,0,0] = MOS[:,1,1].sum() - MOS[:,1,1] 
MOS[:,0,1] = MO.sum(axis=0) - MOS[:,1,1]
MOS[:,1,0] = MO.sum(axis=1) - MOS[:,1,1]

'''
>>> MD
array([[4.        , 0.83333333, 2.16666667, 0.        ],
       [0.5       , 1.66666667, 2.83333333, 0.        ],
       [0.        , 0.        , 1.        , 0.        ],
       [0.        , 0.5       , 0.5       , 1.        ]])
>>> 
>>> MO
array([[5., 2., 4., 0.],
       [0., 2., 3., 1.],
       [0., 0., 1., 0.],
       [0., 1., 1., 1.]])
>>> 
>>> MS
array([[[2, 0],
        [2, 5]],

       [[1, 3],
        [3, 2]],

       [[2, 6],
        [0, 1]]])
>>>
>>> MSS
array([[5, 9],
       [5, 8]])
>>> 
'''

