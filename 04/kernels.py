import numpy as np

def gausKernel( X1 , X2 , gamma=0.1 ):
    return gamma * np.exp( -np.sum( ( X1 - X2.T ) ** 2 ) )


class RBF:

    def __init__( self , gamma=0.01 ):
        self.gamma = gamma

    def __call__( self , X1 , X2 ):
        return np.exp( - self.gamma * np.sum( ( X1 - X2.T ) ** 2 ) )
