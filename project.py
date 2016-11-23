import itertools

def make_params():
    iterations_params = [100, 1000]
    step_params = [0.01, 0.1, 0.5, 1.0, 2.0]
    regParam_params = [0.00001, 0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
    miniBatchFraction_params = [1.0]
    initialWeights_params = [None]
    regType_params = ["l2", None]
    intercept_params = [True]
    validateData_params = [True]
    convergenceTol_params = [0.001, 0.0001]
    listOParams = [iterations_params,step_params,regParam_params,miniBatchFraction_params,initialWeights_params,regType_params,intercept_params,validateData_params,convergenceTol_params]
    
    params = list(itertools.product(*listOParams))
    return params


print(len(make_params()))