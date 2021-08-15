# Optimization Project
The following is a [framework](https://github.com/ncuxomun/Optimization-project/blob/master/pte_project_framework.py) that uses well controls as inputs into a proxy model to compute "optimal" set that maximizes Net Present Value (NPV). The framework consists of several parts that are combined in a loop, which in turn terminates when improvement can no longer be observed. The termination criteria is defined by user expressed as a tolerance value, i.e. difference between what is predicted by proxy controls & simulator production response. 

On a high level, the procedure can be explained in the following steps:

1. We first pick a reservoir to optimize. In this case, we have a set of Gaussian-distributed 2D fields in which three producing and one injecting wells are located at four corners. Control changes are set to vary every six months for each well, that is 27 changes by 4 wells produce 108-sized concatenated vector of control variables. The producer wells are set to vary flowing bottom-hole pressure (psi) which do not fall below bubble-point pressure, whilst the injector's control to vary is injection rate (stb/day).

2. We generate 300 examples given 300 different control settings produced using Latin Hypercube Sampling algorithm to compute NPV value for each case. 

3. Following the completion of data collection, we then train our proxy. The proxy is built using PyTorch Lightning PyTorch-based wrapper. The model is trained based on 200 examples with the rest, 100 examples, set for validation. The following figures show the predictive capability of the model by comparing with True values, expressed via Root Mean Squared Error (RMSE) and R2 score. Please note both true and predicted values are normalized betweet 0 and 1. 

RMSE

| Training set      | Validation set     |
|------------|-------------|
| <img src="https://github.com/ncuxomun/Optimization-project/blob/master/train_npv.png" > | <img src="https://github.com/ncuxomun/Optimization-project/blob/master/val_npv.png"> |
 
R2
  
| Training set      | Validation set     |
|------------|-------------|
| <img src="https://github.com/ncuxomun/Optimization-project/blob/master/train_x_x.png" > | <img src="https://github.com/ncuxomun/Optimization-project/blob/master/val_x_x.png"> |

4. Once the training is completed, the newly optimized controls are then fed into a reservoir simulator which then returns a corresponding NPV value. This process is necessary because "the optimized" controls fall out of the training dataset that most likely be overestimated by the proxy.

5. The process is then continued by adding new samples into the data bank. The proxy's training is then conducted with the new data bank - to improve the model's predictive ability to prevent overestimation. The computation continues until there is no longer (reasonably large) difference between the proxy and reservoir simulator NPV values.

All components used in the framework are located the [lib](https://github.com/ncuxomun/Optimization-project/tree/master/lib) folder.
