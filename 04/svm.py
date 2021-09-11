#!-*-coding:utf-8-*-
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class SVM:
    """
    共通メソッド
    - ルール
      プログラム上でギリシャ文字が扱えないことと、lambdaは予約語であるのでλをPとおいて記述する
    """
    def __init__( self , observe_mode=False ):
        self.w = None
        self.b = None
        self.P = None
        self.y_hat = None
        self.fig = plt.figure()
        self.observe_mode = observe_mode
        self.ims = []

    def fit( self , X , Y , max_epochs=1e+4 ):
        self.w = np.zeros( X.shape[1] ) # 特徴行列の列数が変数の数
        self.b = 0
        self.P = np.zeros( X.shape[0] )
        for count in range( int( max_epochs ) ):
            self.y_hat = Y * np.dot( X , self.w ) + self.b
            condition = self.__kkt( X , Y )
            if len( condition[0] ) <=0:
                break
            i , j = self.__lambda_choice( Y , condition ) #2つのλの添字を取得。λ自体ではなく添字を取得した方があとで再利用性が高いから
            self.__optimize( i , j , X , Y )
            if self.observe_mode:
                print( "{}回目の学習 # w:{} b:{}".format( count , self.w , self.b ) )
                print( "KKT条件に違反するλ # λ:{}".format( condition[0] ) )
                print( "\n" )


    def predict( self ):
        pass

    def observe( self , save_path = "" ):
        ani = animation.ArtistAnimation( self.fig , self.ims , interval=200)
        if save_path == "":
            print( "save_pathに保存先のパスを指定してください" )
            sys.exit()
        else:
            ani.save( save_path , writer='imagemagick' )


    """
    専用メソッド
    """

    def __kkt( self , X , Y ):
        # Karush-Kuhn-Tucker条件に違反するラグランジュ係数Pを選択する
        P , Y = self.P.reshape( -1 ) , Y.reshape( -1 )
        condition = self.P * ( Y * ( self.y_hat ) - 1 ) #np.dot()は内積、*はアダマール積

        return np.where( condition == 0 )

    def __lambda_choice( self , Y , condition ):
        # KKT条件に違反する二つのλをランダムに選択
        P = self.P
        i = np.random.choice( condition[0] , 1 , replace=True )[0] #一つ目のλの添字を選択
        E = self.__E( Y )
        E_delta = E - E[ i ] #iの誤差との差が最大になるようなp<λ>jを選択
        j = np.argmax( E_delta )

        return i , j

    def __E( self , Y ):
        # 予測結果と教師データの差分を求める
        return self.y_hat - Y

    def __optimize( self , i , j , X , Y ):
        # 重みの更新
        P , E = self.P , self.__E( Y )
        x1_square = np.dot( X[ i ] , X[ i ].T )
        x1x2 = np.dot( X[i] , X[j].T )
        x2_square = np.dot( X[ j ] , X[ j ].T )
        P[ i ] += ( Y[ i ] * ( E[ j ] - E[ i ] ) ) / ( x1_square + x1x2 + x2_square )
        P[ j ] += Y[ i ] * Y[ j ] * ( self.P[ i ] - P[ i ] )
        w_ , b_ = 0 , 0
        for n in range( Y.shape[ 0 ] ):
            w_ += P[ n ] * Y[ n ] * X[ n ]
            tmp_ = 0
            for m in range( Y.shape[ 0 ] ): #mはサポートベクトルの数に変えなければならない
                tmp_ +=P[ m ] * Y[ m ]* np.dot( X[ m ] , X[ n ].T )
            b_ += Y[ n ] - tmp_

        self.w = w_
        self.b = b_ / Y.shape[ 0 ]
        self.P = P
        self.__observe( X , Y )

    def __div_line( self , x ):
        return ( -x*self.w[0] - self.b ) / self.w[1]

    def __observe( self , X , Y ):
        if self.observe_mode:
            if X.ndim == 2:
                plt.cla()
                x = np.arange( np.min( X[:,0] ) , np.max( X[:,0] ) , 0.1 )
                y = self.__div_line( x )
                plt.xlim( np.min( X[:,0] ) * 1.5 , np.max( X[:,0] ) * 1.5 )
                plt.ylim( np.min( X[:,1]  * 1.5) , np.max( X[:,1] ) * 1.5 )
                plt.scatter( X[Y == -1][:, 0], X[Y == -1][:, 1], color='lightskyblue', label='-1' )
                plt.scatter( X[Y == 1][:, 0], X[Y == 1][:, 1], color='sandybrown', label='1' )
                plt.ylabel("x2")
                plt.xlabel("x1")
                plt.title( "SVM" )
                plt.legend()
                self.ims.append( plt.plot( x , y ) )
            elif X.ndim == 3:
                pass
