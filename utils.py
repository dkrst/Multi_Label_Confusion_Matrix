import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Print MLCM elements
def print_MLCM(MLCM, norm=True):
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


def show_heatmap(MLCM, labels=None, annot=True, caption=None, cbar=True):
    fig, ax = plt.subplots()
    sns.heatmap(MLCM, fmt='.2f', annot=annot, annot_kws={"weight": "bold"}, cbar=cbar)
    if labels is not None:
        ax.set_xticks(np.arange(len(labels))+.5, labels=labels)
        ax.set_yticks(np.arange(len(labels))+.5, labels=labels)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")
    else:
        ax.set_xticks([],[])
        ax.set_yticks([],[])
    if caption is not None:
        ax.set_title(caption)
        
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

        
