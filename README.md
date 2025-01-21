# DeRES: a Deep Redundancy Elimination System for Large Graph GNN Acceleration

DeRES is an acceleration component for GNN redundancy removal system, which can significantly improve the processing speed and computational overhead of large-scale graph redundancy removal.

DeRES mainly consists of two modules: Pruning module and Index-Multiplication module. The Pruning module is responsible for reducing the number of branches of the common neighbor search algorithm(CNS); the Index-Multiplication module is responsible for reducing the computational overhead when finding the intersection of neighbor sets between nodes. With their combined effect, DeRES can greatly reduce the computational overhead of redundant search, so as to achieve the purpose of accelerating the overall redundancy removal execution process.



##Requirements
```
    pip install pytorch=1.11.0=py3.8_cuda11.3_cudnn8_0
    pip install pyg=2.0.4=py38_torch_1.11.0_cu113
```

If the environment is configured incorrectly, try:

```
conda env create -f environment.yml
```

## Overall performance

Due to significant differences in dataset scales, time overheads vary widely. To illustrate DeRES's optimization effects, we use the speedup ratio to evaluate its performance in accelerating the redundancy removal system, testing the time for redundant-free TGAT during 1000 epochs of end-to-end inference. "Total" refers to the complete redundancy-free GNN process (redundancy removal + GNN inference), while "inference only" excludes redundancy elimination.

To run the Overall performance of DeRES:
```
python ./models/end-to-end.py
```



## Prunning performance

We evaluate the speedup of the redundancy removal system on different datasets when only the Pruning module is used.

To run the performance of Prunning module in DeRES:
```
python ./models/prunning.py
```



## Index-Multiplication  performance

We evaluate the speedup of the redundancy removal system on different datasets when only the Index-Multiplication module is used.

To run the performance of Index-Multiplication module in DeRES:
```
python ./models/index_multi.py
```

## DeRES module performance

We evaluate the speedup of the redundancy removal system on different datasets when DeRES is integrated.

To run the speedup performance of DeRES on Redundancy Elimination:
```
python ./models/DeRES.py
```
