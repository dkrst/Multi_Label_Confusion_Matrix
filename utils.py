import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle
import seaborn as sns

# Print MLCM elements
def print_MLCM(MLCM, norm=False):
    M = MLCM.copy()
    n = M.shape[0]
    if norm:
        rs = np.sum(M, axis=1)
        for k in range(n):
            M[k,:] = 100*M[k,:]/rs[k]
            
    print('Classes & $C_{0}$ & $C_{1}$ & $C_{2}$ & $C_{3}$ & $C_{4}$ & $C_{5}$ & $C_{6}$ & $C_{7}$ & $C_{8}$ & $C_{9}$ & $C_{10}$ & $C_{11}$ & $C_{12}$ & $C_{13}$ & $C_{14}$ & $C_{15}$ & $C_{16}$ & $C_{17}$ & NPL \\\\ ')
    for i in range(n):
        print('$C_{%d}$' %i, end='')
        for j in range(n):
            print(' & %d' %np.round(M[i,j]), end='')
            # print('%7.2f' %MLCM[i,j], end='  ')
        print(' \\\\')


def show_heatmap(MLCM, labels=None, caption=None, cmap='YlOrBr', cbar=False,
                 logscale=False):
    fig, ax = plt.subplots(figsize=(9,6))
    
    fmt = '.2f'
    vmax=1
    annotations = MLCM.round(decimals=2).astype(str)
    n = MLCM.shape[0]
    for i in range(n):
        for j in range(n):
            annotations[i,j] = annotations[i,j].replace('0.', '.')
    if logscale:
        norm=LogNorm(vmax=vmax, clip=True)
    else:
        norm=None
        #norm=LogNorm(vmax=vmax, clip=True)
        
    #sns.heatmap(MLCM, fmt=fmt, annot=annot, annot_kws={"weight": "bold"}, cmap=cmap,
    #            cbar=cbar, norm=norm)
    map = sns.heatmap(MLCM, vmax=vmax, fmt='s', annot=annotations,
                      annot_kws={"weight": "bold"}, cmap=cmap,
                      cbar=cbar, norm=norm)
    for k in range(n):
        map.add_patch(Rectangle((k,k), 1, 1, fill=False,
                                edgecolor='black', lw=1))

    if labels is not None:
        ax.set_xticks(np.arange(len(labels))+.5, labels=labels)
        ax.set_yticks(np.arange(len(labels))+.5, labels=labels, rotation=0)
        #plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
        #         rotation_mode="anchor")
    else:
        ax.set_xticks([],[])
        ax.set_yticks([],[])
    if caption is not None:
        ax.set_title(caption)

    plt.tight_layout()
    plt.show()

def normalize_MLCM(MLCM, axis=1):
    M = MLCM.copy()
    n = M.shape[0]
    s = np.sum(M, axis=axis)
    for k in range(n):
        if axis==0:
            M[:,k] = 100*M[:,k]/s[k]
        else:
            M[k,:] = 100*M[k,:]/s[k]
    return M

def convPrecisionRecall(gt, pred):
    if gt.shape != pred.shape:
        raise Exception('GT labels and predictions do not match')

    k, q = gt.shape

    tp = np.zeros(q)
    fp = np.zeros(q)
    tn = np.zeros(q)
    fn = np.zeros(q)

    for i in range(k):
        for l in range(q):
            if gt[i][l]:
                if pred[i][l]:
                    tp[l] += 1
                else:
                    fn[l] += 1
            else:
                if pred[i][l]:
                    fp[l] += 1
                else:
                    tn[l] += 1

    precision = np.zeros(q)
    recall = np.zeros(q)
    for i in range(q):
        precision[i] = tp[i] / (tp[i]+fp[i])
        recall[i] = tp[i] / (tp[i] + fn[i])

    return precision, recall


def addUnknown(x):
    h, w = x.shape
    y = np.zeros((h, w+1))
    y[:,:w] = x
    y[:,w] = (np.sum(x, axis=1)==0).astype(float)
    return y
