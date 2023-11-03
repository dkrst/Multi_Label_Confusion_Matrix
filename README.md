# Multi Label Confusion Tensor

Implementations of Multi Label Confusion Tensor for Multi Label Classifier performance evaluation

## Files
- [Evaluate_Multi-Label-movies.ipynb](Evaluate_Multi-Label-movies.ipynb): Evaluate Movie Posters classifier using all three Confusion-based evaluators
- [Evaluate_Multi-Label-syntetic.ipynb](Evaluate_Multi-Label-syntetic.ipynb): Evaluate syntetic example using all three Confusion-based evaluators 
- [Micro-Macro_averaging.ipynb](Micro-Macro_averaging.ipynb): Computing Label-based evaluation metrics from MLCT 
- [mlctensor.py](mlctensor.py): Implementation of Multi Label Confusion Tensor (MLCT)
- [mlcevaluator1.py](mlcevaluator1.py): Implementation of Multi Label Confusion Matrix (MLCM) as proposed by Krstinic et al. [[1]](#1)
- [mlcevaluator2.py](mlcevaluator2.py): Implementation of MLCM as proposed by Heydarian et al. [[2]](#2)
- [utils.py](utils.py): Utils


## References
<a id="1">[1]</a>
Damir Krstinić, Maja braović, Ljiljana Šerić and Dunja Božić-Štulić, *"Multi-Label Classifier Performance Evaluation with Confusion Matrix"*, in Proc.of Int.Conf. on Soft Computing, Artificial Intelligence and Machine Learning [(SAIM2020), vol. 10](https://airccse.org/csit/V10N08.html), pp. 1-14., 2020. [doi: 10.5121/csit.2020.100801](https://aircconline.com/csit/abstract/v10n8/csit100801.html).

<a id="2">[2]</a>
Mohammadreza Heydarian, Thomas E. Doyle and Reza Samavi, *"MLCM: Multi-Label Confusion Matrix"*, [IEEE Access, vol 10.](https://ieeexplore.ieee.org/xpl/tocresult.jsp?isnumber=9668973&punumber=6287639), pp. 19083-19095, 2022, [doi: 10.1109/ACCESS.2022.3151048](https://ieeexplore.ieee.org/abstract/document/9711932).
