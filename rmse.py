import numpy as np

def rmse(predictions, targets):
    r"""
    RMSE (Root Mean Square Error) = 
    $\sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y_i})^2}$,
    where $y_i$ are the true values, $\hat{y_i}$ are the predicted values,
    and $n$ is the total number of samples.
    """
    pred = np.array(predictions)
    tar = np.array(targets)
    rmse = np.sqrt(np.mean((tar-pred)**2))
    return rmse
