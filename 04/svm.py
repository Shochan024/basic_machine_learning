#!-*-coding:utf-8-*-
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
np.set_printoptions( precision=10 , floatmode='fixed' )

class LinearSVM:
    """
    共通メソッド
    - ルール
      プログラム上でギリシャ文字が扱えないことと、lambdaは予約語であるのでλをPとおいて記述する
    """
    def __init__( self , observe_mode=False , C=0 ):
        self.w = None
        self.b = None
        self.P = None
        self.C = C
        self.y_hat = None
        self.support_vectors_ = None
        self.fig = plt.figure()
        self.observe_mode = observe_mode
        self.ims = []

    def fit( self , X , Y , max_epochs=1e+4 ):
        self.w = np.zeros( X.shape[1] ) # 特徴行列の列数が変数の数
        self.b = 0
        self.P = np.zeros( X.shape[0] )
        self.X , self.Y = X , Y
        for count in range( int( max_epochs ) ):
            self.support_vectors_ = np.where( ( np.dot( X , self.w ) + self.b - 1 ) / np.linalg.norm( self.w ) )
            self.y_hat = self.predict( X , Y )
            condition = self._kkt( X , Y )
            if len( condition[0] ) <=0:
                break
            i , j = self._lambda_choice( Y , condition ) #2つのλの添字を取得。λ自体ではなく添字を取得した方があとで再利用性が高いから
            self._optimize( i , j , X , Y )
            print( "{}回目の学習 # w:{} b:{}".format( count , self.w , self.b ) )
            print( "KKT条件に違反するλ # λ:{}".format( condition[0] ) )
            print( "\n" )
            if self.observe_mode:
                plt.scatter( X[Y == -1][:, 0], X[Y == -1][:, 1], color='lightskyblue' )
                plt.scatter( X[Y == 1][:, 0], X[Y == 1][:, 1], color='sandybrown' )


    def predict( self , X , Y=None ):
        return Y * np.dot( X , self.w ) + self.b

    def observe( self , save_path = "" ):
        ani = animation.ArtistAnimation( self.fig , self.ims , interval=200 )
        if save_path == "":
            print( "save_pathに保存先のパスを指定してください" )
            sys.exit()
        else:
            ani.save( save_path , writer='imagemagick' )


    """
    専用メソッド
    """

    def _observe( self , X , Y ):
        if self.observe_mode:
            x1_plus = ( np.min( X[:,0] ) + np.max( X[:,0] ) ) * 0.25
            x2_plus = ( np.min( X[:,1] ) + np.max( X[:,1] ) ) * 0.25
            if X.ndim == 2:
                x = np.arange( np.min( X[:,0] ) , np.max( X[:,0] ) , 0.1 )
                y = self._div_line( x )
                plt.xlim( np.min( X[:,0] ) - x1_plus , np.max( X[:,0] ) + x1_plus )
                plt.ylim( np.min( X[:,1] ) - x2_plus , np.max( X[:,1] ) + x2_plus )
                plt.scatter( X[Y == -1][:, 0], X[Y == -1][:, 1], color='lightskyblue' )
                plt.scatter( X[Y == 1][:, 0], X[Y == 1][:, 1], color='sandybrown' )
                plt.ylabel("x2")
                plt.xlabel("x1")
                plt.title( "SVM" )
                plt.legend()
                self.ims.append( plt.plot( x , y , color="lightskyblue" ) )
            elif X.ndim == 3:
                pass

    def _optimize( self , i , j , X , Y ):
        # 重みの更新
        P , E = self.P , self._E( Y )
        x1_square = np.dot( X[ i ] , X[ i ].T )
        x1x2 = np.dot( X[i] , X[j].T )
        x2_square = np.dot( X[ j ] , X[ j ].T )
        P_i_delta = self._clip( i , j , Y , Y[ i ] * ( E[ j ] - E[ i ] ) / ( x1_square + x1x2 + x2_square ) )
        P_j_delta = Y[ i ] * Y[ j ] * ( self.P[ i ] - P_i_delta )
        P[ i ] += P_i_delta
        P[ j ] += P_j_delta
        w_ , b_ = 0 , 0
        for n in range( Y.shape[ 0 ] ):
            w_ += P[ n ] * Y[ n ] * X[ n ]
            tmp_ = 0
            for m in range( Y.shape[ 0 ] ): #mはサポートベクトルの数に変えなければならない
                tmp_ += P[ m ] * Y[ m ] * np.dot( X[ m ] , X[ n ].T )
            b_ += Y[ n ] - tmp_

        self.w = w_
        self.b = b_ / Y.shape[ 0 ]
        self.P = P
        self._observe( X , Y )

    def _kkt( self , X , Y ):
        # Karush-Kuhn-Tucker条件に違反するラグランジュ係数Pを選択する
        P , Y = self.P.reshape( -1 ) , Y.reshape( -1 )
        condition = self.P * ( Y * ( self.y_hat ) - 1 ) #np.dot()は内積、*はアダマール積

        return np.where( condition == 0 )

    def _lambda_choice( self , Y , condition ):
        # KKT条件に違反する二つのλをランダムに選択
        P = self.P
        i = np.random.choice( condition[0] , 1 , replace=True )[0] #一つ目のλの添字を選択
        E = self._E( Y )
        E_delta = E - E[ i ] #iの誤差との差が最大になるようなp<λ>jを選択
        j = np.argmax( E_delta )

        return i , j

    def _E( self , Y ):
        # 予測結果と教師データの差分を求める
        return self.y_hat - Y

    def _div_line( self , x ):
        return ( -x * self.w[0] - self.b ) / self.w[1]

    def _clip( self , i , j , Y , P_del ):
        P = self.P
        if Y[i] == Y[j]:
            L = max( 0 , P[i] + P[j] - self.C )
            H = max( self.C , P[i] + P[j] )
        else:
            L = max( 0 , P[j] - P[i] )
            H = min( self.C , self.C - P[j] + P[i] )

        if P_del > H:
            P_del = H
        elif L > P_del:
            P_del = L

        return P_del


class kernelSVM:

    def __init__( self , kernel , C=1.0 ):
        self.P = None
        self.X_sv = None
        self.Y_sv = None
        self.b = None
        self.sv = None
        self.C = C
        self.kernel = kernel
        self.fig = plt.figure()
        self.ims = []

    def fit( self , X , Y , e=1e-6 , max_epochs=1e+3 ):
        P = np.zeros_like( Y , dtype=np.float )
        grad_D = np.ones_like( Y , dtype=np.float )

        x1_plus = ( np.min( X[:,0] ) + np.max( X[:,0] ) ) * 0.25
        x2_plus = ( np.min( X[:,1] ) + np.max( X[:,1] ) ) * 0.25
        x = np.arange( np.min( X[:,0] ) - x1_plus , np.max( X[:,0] ) + x1_plus , 0.2 )
        y = np.arange( np.min( X[:,1] ) - x2_plus , np.max( X[:,1] ) + x2_plus , 0.2 )
        xx , yy = np.meshgrid( x , y )
        pred_x = np.c_[ xx.ravel() , yy.ravel() ]
        plt.xlim( np.min( X[:,0] ) - x1_plus , np.max( X[:,0] ) + x1_plus )
        plt.ylim( np.min( X[:,1] ) - x2_plus , np.max( X[:,1] ) + x2_plus )
        plt.scatter( X[Y == -1][:, 0], X[Y == -1][:, 1], color='lightskyblue' )
        plt.scatter( X[Y == 1][:, 0], X[Y == 1][:, 1], color='sandybrown' )

        for i in range( int( max_epochs ) ):
            print( i )
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
