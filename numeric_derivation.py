def derive(f, x, h=0.0001):
    r"""
    Calculate the derivative of function f at point x using the central difference method.

    Parameters:
    f (function): The function for which the derivative is to be calculated.
    x (float): The point at which the derivative is calculated.
    h (float): A small value for computing the difference. Default is 0.0001.

    Returns:
    float: The derivative of function f at point x.
    """
    return (f(x + h) - f(x - h)) / (2 * h) 
