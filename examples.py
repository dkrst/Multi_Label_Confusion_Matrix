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

# MS = multilabel_confusion_matrix(gt, pred)

from  utils import print_MLCM
from utils import show_heatmap
from utils import normalize_MLCM

print_MLCM(MD)

labels=['label1', 'label2', 'label3', 'label4']
show_heatmap(MD, labels)

ND = normalize_MLCM(MD)
ND1 = normalize_MLCM(MD1)
NO= normalize_MLCM(MO)

# Load GT and prediction from file
gt=np.load('LOCAL/gt_t09.npy')
gt.shape

pred=np.load('LOCAL/pred_t09.npy')
pred.shape

# Evaluator1, using category unknown
evalD=mlcEvaluator1(gt, pred, use_unknown=True)
MD=evalD.computeConfusionMatrix()

# Evaluator1, without category unknown
evalD1=mlcEvaluator1(gt, pred, use_unknown=False)
MD1=evalD1.computeConfusionMatrix()

# Evaluator 2
evalO=mlcEvaluator2(gt, pred)
MO = evalO.computeConfusionMatrix()

ND = normalize_MLCM(MD)
ND1 = normalize_MLCM(MD1)
NO= normalize_MLCM(MO)

print_MLCM(MD)

ND = normalize_MLCM(MD)
ND1 = normalize_MLCM(MD1)
NO= normalize_MLCM(MO)

show_heatmap(ND, labels=None)
show_heatmap(ND1, labels=None)
show_heatmap(NO, labels=None)

# Spectral_r, seismic, Set1_r, rainbow, turbo
''' 
LABELS:
Labels:
0. Action
1. Adventure
2. Animation
3. Biography
4. Comedy
5. Crime
6. Documentary
7. Drama
8. Family
9. Fantasy
10. History
11. Horror
12. Music
13. Mystery
14. Romance
15. Sci-Fi
16. Thriller
17. War
'''

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

