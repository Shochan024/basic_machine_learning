#!-*-coding:utf-8-*-
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
np.set_printoptions( precision=10 , floatmode='fixed' )

class linearSVM:

    def __init__( self , C=1.0 ):
        self.C = C
        self.b = None
        self.w = None
        self.P = None
        self.xi = None
        self.sv = None

    def fit( self , X , Y , save_path=None , e=1e-9 , max_epochs=1000 ):
        self.w = np.zeros( ( 1 , X.shape[1] ) )
        self.b = 0
        self.P = np.zeros_like( Y , dtype=np.float )
        self.xi = np.zeros_like( Y , dtype=np.float )

        # plot用
        ims = []
        fig = plt.figure()
        x1_plus = ( np.min( X[:,0] ) + np.max( X[:,0] ) ) * 0.25
        x2_plus = ( np.min( X[:,1] ) + np.max( X[:,1] ) ) * 0.25
        x1_plot = np.arange( np.min( X[:,0] ) - x1_plus , np.max( X[:,0] ) + x1_plus , 0.2 )
        x2_plot = np.arange( np.min( X[:,1] ) - x2_plus , np.max( X[:,1] ) + x2_plus , 0.2 )
        plt.xlim( np.min( X[:,0] ) - x1_plus , np.max( X[:,0] ) + x1_plus )
        plt.ylim( np.min( X[:,1] ) - x2_plus , np.max( X[:,1] ) + x2_plus )
        plt.scatter( X[Y == -1][:, 0], X[Y == -1][:, 1], color='lightskyblue' )
        plt.scatter( X[Y == 1][:, 0], X[Y == 1][:, 1], color='sandybrown' )

        # loop
        for _ in range( max_epochs ):
            P = self.P

            violations = self.__kkt( X , Y , e )

            print( "epoch : {} , violation samples : {} , w : {} , b : {}".format( _ , len( violations ) , self.w , self.b ) )
            # 停止条件
            if len( violations ) <= 0:
                break

            E = self.__E( X , Y )

            i , j = self._lambda_choice( violations , X , Y , E )

            xi_square = X[ i ] @ X[ i ].T
            xj_square = X[ j ] @ X[ j ].T
            xij = X[ i ] @ X[ j ].T
            _Pi = ( Y[ i ] * ( E[ j ] - E[ i ] ) ) / ( xi_square - 2*xij + xj_square ) #ここが0ということはない
            Pi_delta = self.__clip( i , j , Y , self.P[ i ] + _Pi )
            P[ i ] = Pi_delta

            Pj_delta = Y[ i ] * Y[ j ] * ( self.P[ i ] - P[ i ] )
            P[ j ] += Pj_delta


            xi_ = 1 - Y * self.__decision_function( X )

            self.sv = np.where( P > 0 )
            self.P = P
            self.xi = np.where( xi_ >= 0 , xi_ , 0 )




            # Wを更新 経過観察用にloopの中に入れておく
            for n in range( X.shape[ 1 ] ):
                self.w[:,n] = np.sum( P * Y  * X[:,n] )


            # bを更新
            b_ = Y[ self.sv ] -  self.w @ X[ self.sv ].T
            self.b = np.sum( b_ ) / len( b_ )

            w_ = self.w.reshape( -1 )
            y1 = ( -x1_plot * w_[0] - self.b ) / w_[1]
            y2 = ( 1 -x1_plot * w_[0] - self.b ) / w_[1]
            y3 = ( -1 -x1_plot * w_[0] - self.b ) / w_[1]
            image1 = plt.plot( x1_plot , y1 , color='lightskyblue' )
            image2 = plt.plot( x1_plot , y2 , color='yellow' )
            image3 = plt.plot( x1_plot , y3 , color='yellow' )
            area = plt.fill_between( x1_plot , y2 , y3 , facecolor='y' , color="blue" , alpha=0.3 )

            ims.append( image1 + image2 + image3 + [area] )

        ani = animation.ArtistAnimation( fig , ims , interval=200 )
        ani.save( save_path , writer='imagemagick' )


    def __kkt( self , X , Y , e ):
        comp_condition = Y * self.__decision_function( X , Y )

        violation = [ n for n in range( self.P.shape[0] ) if ( ( self.P[ n ] == 0 and comp_condition[ n ] < 1 - self.xi[ n ] ) or ( self.P[ n ] > 0 and comp_condition[ n ] != 1.0 ) ) ]

        return violation


    def _lambda_choice( self , violations , X , Y , E ):

        i = np.random.choice( violations , 1 , replace=True )[0] #一つ目のλの添字を選択

        j = np.argmax( np.abs( E - E[ i ] ) )


        return i , j

    def __decision_function( self , X , Y=None ):
        return ( self.w @ X.T + self.b ).T.reshape( -1 )

    def __E( self , X , Y ):
        y_hat = self.__decision_function( X , Y ).T.reshape( -1 )
        return y_hat - Y

    def __clip( self , i , j , Y , P_del ):
        if Y[i] == Y[j]:
            L = max( 0 , self.P[i] + self.P[j] - self.C )
            H = max( self.C , self.P[i] + self.P[j] )
        else:
            L = max( 0 , self.P[j] - self.P[i] )
            H = min( self.C , self.C - self.P[j] + self.P[i] )

        _Pdel = P_del

        if P_del > H:
            P_del = H
        elif L > P_del:
            P_del = L


        return P_del


class kernelSVM:

    def __init__( self , kernel , C=float("inf") ):
        self.P = None
        self.X_sv = None
        self.Y_sv = None
        self.b = None
        self.sv = None
        self.C = C
        self.kernel = kernel
        self.fig = plt.figure()
        self.ims = []

    def fit( self , X , Y , e=1e-2  , max_epochs=1e+3 ):
        P = np.zeros_like( Y , dtype=np.float )
        grad_D = np.ones_like( Y , dtype=np.float )

        # Plot用
        x1_plus = ( np.min( X[:,0] ) + np.max( X[:,0] ) ) * 0.25
        x2_plus = ( np.min( X[:,1] ) + np.max( X[:,1] ) ) * 0.25
        x = np.arange( np.min( X[:,0] ) - x1_plus , np.max( X[:,0] ) + x1_plus , 0.2 )
        y = np.arange( np.min( X[:,1] ) - x2_plus , np.max( X[:,1] ) + x2_plus , 0.2 )
        xx , yy = np.meshgrid( x , y )
        pred_x = np.c_[ xx.ravel() , yy.ravel() ]
        plt.xlim( np.min( X[:,0] ) - x1_plus , np.max( X[:,0] ) + x1_plus )
        plt.ylim( np.min( X[:,1] ) - x2_plus , np.max( X[:,1] ) + x2_plus )

        for i in range( int( max_epochs ) ):
            condition = Y * grad_D
            i , j = self._lambda_choice( condition , P , Y , e )
            if condition[ i ] <= condition[ j ] + e:
                break
            else:
                A = self.C - P[ i ] if Y[ i ] == 1 else P[ i ]
                B = P[ j ] if Y[ j ] == 1 else self.C - P[ j ]
                gamma = min( A , B ,
                 ( Y[ i ] * grad_D[ i ] - Y[ j ] * grad_D[ j ] ) / ( self.kernel( X[ i ] , X[ i ] ) - 2*self.kernel( X[ i ] , X[ j ] ) + self.kernel( X[ j ] , X[ j ] ) ) )

                # ラグランジュ乗数を更新
                P[ i ] += gamma * Y[ i ]
                P[ j ] -= gamma * Y[ j ]

                # 勾配を更新
                grad_D -= gamma * Y * ( self.kernel( X , X[ i ] ) - self.kernel( X , X[ j ] ) )

            self.b = ( condition[ i ] + condition[ j ] ) / 2
            self.sv = np.where( P > e )
            self.P = P[ self.sv ]
            self.X_sv = X[ self.sv ]
            self.Y_sv = Y[ self.sv ]
            print( len( self.P ) )

            z = self.decision_function( pred_x ).reshape( xx.shape )

            im = plt.contourf( xx , yy , z , alpha=0.8 )
            plt.scatter( X[Y == -1][:, 0], X[Y == -1][:, 1], color='lightskyblue' )
            plt.scatter( X[Y == 1][:, 0], X[Y == 1][:, 1], color='sandybrown' )
            add_anim = im.collections
            self.ims.append( add_anim )


    def decision_function( self , X ):
        return self.kernel( X , self.X_sv ) @ ( self.P * self.Y_sv ) + self.b


    def observe( self , save_path ):
        ani = animation.ArtistAnimation( self.fig , self.ims , interval=200 )
        if save_path == "":
            print( "save_pathに保存先のパスを指定してください" )
            sys.exit()
        else:
            ani.save( save_path , writer='imagemagick' )

    def _lambda_choice( self , condition , P , Y , e ):
        I_up = [ condition[ i ] if ( ( P[ i ] < self.C - e and Y[ i ] == 1 ) or ( P[ i ] > e and Y[ i ] == -1 ) ) else -float( "inf" ) for i in range( len( Y ) ) ]
        I_low = [ condition[ i ] if ( ( P[ i ] < self.C - e and Y[ i ] == -1 ) or ( P[ i ] > e and Y[ i ] == 1 ) ) else float( "inf" ) for i in range( len( Y ) ) ]
        i = np.argmax( I_up )
        j = np.argmin( I_low )


        return i , j
