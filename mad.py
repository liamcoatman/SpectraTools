def mad(arr):

    """ 
    Median Absolute Deviation: a "Robust" version of standard deviation.
    https://en.wikipedia.org/wiki/Median_absolute_deviation 
    """
    
    arr = np.ma.array(arr).compressed() 
    med = np.median(arr)
    return np.median(np.abs(arr - med))