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

'''MLCM heatmaps
'''

import matplotlib.pyplot as plt
'''
def show_heatmap(MLCM, labels=None, caption=None):
    fig, ax = plt.subplots()
    im = ax.imshow(MLCM, cmap='turbo')
    n = MLCM.shape[0]
    if caption is not None:
        ax.set_title(caption)
    if labels is not None:
        ax.set_xticks(np.arange(len(labels)), labels=labels)
        ax.set_yticks(np.arange(len(labels)), labels=labels)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")
    else:
        ax.set_xticks([],[])
        ax.set_yticks([],[])
    for i in range(n):
        for j in range(n):
            t = ax.text(j, i, '{:.2f}'.format(MLCM[i,j]), 
                        ha='center', va='center', color='w', weight='bold')
            
    plt.show()
'''
import seaborn as sns
'''
def show_heatmap(*args, **kwargs):
    labels = kwargs.get('labels', None)
    captions = kwargs.get('captions', None)
    colorbar = kwargs.get('colorbar', True)
    nplots = len(args)
    fig, ax = plt.subplots(ncols=nplots, sharey=True)
    #print(len(ax))
    #MLCM = args[0]
    for k in range(nplots):
        sns.heatmap(args[k], ax=ax[k], annot=True, fmt='.2f',
                    annot_kws={"weight": "bold"}, square=True, cbar=False)
        if labels is not None:
            ax[k].set_xticks(np.arange(len(labels))+.5, labels=labels)
            ax[k].set_yticks(np.arange(len(labels))+.5, labels=labels)
            plt.setp(ax[k].get_xticklabels(), rotation=45, ha="right",
                     rotation_mode="anchor")
        else:
            ax[k].set_xticks([],[])
            ax[k].set_yticks([],[])
        if captions is not None:
            ax[k].set_title(captions[k])
            
    plt.tight_layout()                
    plt.show()

labels=['label1', 'label2', 'label3', 'label4']
show_heatmap(MD, MO)

'''
def show_heatmap(MLCM, labels=None, caption=None, colorbar=True):
    fig, ax = plt.subplots()
    sns.heatmap(MLCM, annot=True, fmt='.0f', annot_kws={"weight": "bold"})
    if labels is not None:
        ax.set_xticks(np.arange(len(labels)), labels=labels)
        ax.set_yticks(np.arange(len(labels)), labels=labels)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")
    else:
        ax.set_xticks([],[])
        ax.set_yticks([],[])
    if caption is not None:
        ax.set_title(caption)
        
    plt.show()

labels=['label1', 'label2', 'label3', 'label4']
show_heatmap(MD, labels)

# Spectral_r, seismic, Set1_r, rainbow, turbo

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

