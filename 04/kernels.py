import numpy as np

class RBF:

    def __init__( self , gamma=0.01 ):
        self.gamma = gamma

    def __call__( self , X1 , X2 ):
        if ( X1.ndim == 1 and X2.ndim == 1 ):
            tmp_ = np.sum( ( X1 - X2.T ) ** 2 )
        elif ( ( X1.ndim == 1 and  X2.ndim != 1 ) or ( X1.ndim != 1 and  X2.ndim == 1 ) ):
            tmp_ = np.linalg.norm( X1 - X2 , axis=1 ) ** 2
        else:
            tmp_ = np.reshape( np.sum( X1**2 , axis=1 ), ( len( X1 ), 1) ) + np.sum( X2**2 , axis=1 )  -2 * ( X1 @ X2.T )

        K = np.exp( - tmp_ / ( 2*self.gamma**2 ) )

        return K


class Linear:

    def __init__( self ):
        pass

    def __call__( self , X1 , X2 ):

        K = tmp_ = X1 @ X2

        return K
