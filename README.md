# Multi_Label_Confusion_Matrix

Implementations of Multi Label Confusion Matrix (MLCM) for Multi Label Classifier performance evaluation

## Files
- [mlcevaluator1.py](mlcevaluator1.py): Implementation of MLCM as proposed by Krstinic et al. [[1]](#1)
- [mlcevaluator2.py](mlcevaluator2.py): Implementation of MLCM as proposed by Heydarian et al. [[2]](#2)
- [examples.py](examples.py): Examples of using MLCM evaluators
- [test_dev.py](test_dev.py): Dev & testing code

## Usage
Set Ground Truth (GT) and prediction matrices. Each row represnts vector coresponding to one GT or prediction data instance.
```python
import numpy as np
gt=np.asarray([[1,1,0], [1,1,1], [0,0,0],
               [1,0,0], [1,1,0], [0,0,0],
               [1,0,0], [1,1,0], [1,1,0]])
pred=np.asarray([[1,1,0],[1,0,1],[0,0,0],
                 [1,1,1], [1,1,1], [0,1,1],
                 [0,1,1], [1,0,1], [0,0,1]])
```

Initialize evaluators:
```python
from mlcevaluator1 import mlcEvaluator1
from mlcevaluator2 import mlcEvaluator2
evalD=mlcEvaluator1(gt, pred, use_unknown=True)
evalD1=mlcEvaluator1(gt, pred, use_unknown=False)
evalO=mlcEvaluator2(gt, pred)
```

Compute MLCM for both implementations
```python
MD=evalD.computeConfusionMatrix()
MD1=evalD1.computeConfusionMatrix()
MO = evalO.computeConfusionMatrix()
```

Compare results:
```python
print(MD1)
print(MD)
print(MO)
```

Compare contribution of a particular instance to the MLCM:
```python
# Contribution of the instance 7
# g=[1,1,0]; p=[1,0,1]
# CATEGORY: (iv) in [1], (III) in [2] 
evalD1.getInstanceContribution(7)
evalD.getInstanceContribution(7)
evalO.getInstanceContribution(7)
```

## References
<a id="1">[1]</a>
Damir Krstinić, Maja braović, Ljiljana Šerić and Dunja Božić-Štulić, *"Multi-Label Classifier Performance Evaluation with Confusion Matrix"*, in Proc.of Int.Conf. on Soft Computing, Artificial Intelligence and Machine Learning [(SAIM2020), vol. 10](https://airccse.org/csit/V10N08.html), pp. 1-14., 2020. [doi: 10.5121/csit.2020.100801](https://aircconline.com/csit/abstract/v10n8/csit100801.html).

<a id="2">[2]</a>
Mohammadreza Heydarian, Thomas E. Doyle and Reza Samavi, *"MLCM: Multi-Label Confusion Matrix"*, [IEEE Access, vol 10.](https://ieeexplore.ieee.org/xpl/tocresult.jsp?isnumber=9668973&punumber=6287639), pp. 19083-19095, 2022, [doi: 10.1109/ACCESS.2022.3151048](https://ieeexplore.ieee.org/abstract/document/9711932).
