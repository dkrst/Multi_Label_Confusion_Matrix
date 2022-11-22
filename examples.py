import numpy as np
from mlcevaluator1 import mlcEvaluator1
from mlcevaluator2 import mlcEvaluator2
from sklearn.metrics import multilabel_confusion_matrix

# Example GT and Prediction matrices
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

MS = multilabel_confusion_matrix(gt, pred)

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

