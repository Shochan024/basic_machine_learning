#!-*-coding:utf-8-*-
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
np.set_printoptions( precision=10 , floatmode='fixed' )

class linearSVM:

    def __init__( self , C ):
        self.C = C
        self.w = None
        self.b = 0
        self.E = None
        self.sv = []
        self.P = None
        self.Y_hat = None
        self.tol = 1e-3
        self.fig = plt.figure()
        self.ims = []
        self.x1_plus = None
        self.x2_plus = None

    def fit( self , X , Y ):
        """
        ~ 学習メソッド ~

        X <numpy> : 特徴量 < 行数 : サンプル数 , 列数 : 特徴の数 >
        Y <numpy> : 教師データ
        """
        self.P = np.zeros_like( Y ) #ラグランジュ乗数の数はトレーニングサンプル数と同じ数
        self.Y_hat = self._decision_function( X , Y )
        self.E = self._E( Y , self.Y_hat ) # error cacheを計算しておく
        self.w = np.zeros( ( 1 , X.shape[ 1 ] ) )

        """
        plot用
        """
        x1_plus = ( np.min( X[:,0] ) + np.max( X[:,0] ) ) * 0.25
        x2_plus = ( np.min( X[:,1] ) + np.max( X[:,1] ) ) * 0.25
        self.x1_plot = np.arange( np.min( X[:,0] ) - x1_plus , np.max( X[:,0] ) + x1_plus , 0.2 )
        self.x2_plot = np.arange( np.min( X[:,1] ) - x2_plus , np.max( X[:,1] ) + x2_plus , 0.2 )
        plt.xlim( np.min( X[:,0] ) - x1_plus , np.max( X[:,0] ) + x1_plus )
        plt.ylim( np.min( X[:,1] ) - x2_plus , np.max( X[:,1] ) + x2_plus )
        plt.scatter( X[Y == -1][:, 0], X[Y == -1][:, 1], color='lightskyblue' )
        plt.scatter( X[Y == 1][:, 0], X[Y == 1][:, 1], color='sandybrown' )

        numChanged = 0
        examineAll = True

        # 更新量が1以上または全てをサーチするステータスがOnならloopを継続
        while numChanged > 0 or examineAll:
            numChanged = 0
            if examineAll:
                for i in range( len( self.P ) ):
                    numChanged += self._examineExample( i , X , Y )
            else:
                for i in np.where( ( ( self.P > 0 ) & ( self.P < self.C ) ) )[ 0 ]:
                    numChanged += self._examineExample( i , X , Y )

            if examineAll:
                # もし全てのサンプルをサーチするステータスがOnになっていたら、それをOffにする
                examineAll = False
            elif numChanged == 0:
                # 一部のサンプルだけをみても更新量がゼロなら、全てのサンプルをサーチする
                examineAll = True


        ani = animation.ArtistAnimation( self.fig , self.ims , interval=200 )
        ani.save( "graphs/softmargin.gif" , writer='imagemagick' )



    def predict( self ):
        pass

    def _takeStep( self , i1 , i2 , X , Y , eps=1e-3 ):
        """
        ~ 更新メソッド ~

        i1 <int>  : 1つめのラグランジュ乗数の添字
        i2 <int>  : 2つめのラグランジュ乗数の添字
        X <numpy> : 特徴量 < 行数 : サンプル数 , 列数 : 特徴の数 >
        Y <numpy> : 教師データ
        eps <float> : εのこと。誤差を許す範囲。論文ではε=10^-3とされている
        """
        E = self.E
        changedStatus = False
        if i1 == i2:
            changedStatus = True

        alph1 , alph2 = self.P[ i1 ] , self.P[ i2 ]
        y1 , y2 = Y[ i1 ] , Y[ i2 ]
        E1 , E2 = E[ i1 ] , E[ i2 ]
        s = y1 * y2

        k11 = X[ i1 ] @ X[ i1 ].T
        k12 = X[ i1 ] @ X[ i2 ].T
        k22 = X[ i2 ] @ X[ i2 ].T

        eta = k11 + k22 - 2 * k12

        if Y[ i1 ] == Y[ i2 ]:
            L = max( 0 , self.P[ i2 ] + self.P[ i1 ] - self.C )
            H = min( self.C , self.P[ i2 ] + self.P[ i1 ] )
        else:
            L = max( 0 , self.P[ i2 ] - self.P[ i1 ] )
            H = min( self.C , self.C - ( self.P[ i2 ] - self.P[ i1 ] ) )

        if eta > 0:
            #etaが正の値なら、ラグランジュ乗数を普通に更新
            a2_new = alph2 + y2 * ( E1 - E2 ) / eta
            a2 = self._clip( Y , i1 , i2 , a2_new )
        else:
            #etaが負の場合、ヘッセ行列が正定値ではないので少し特別な処理をする
            Lobj , Hobj = self._objective_function( X , Y , i1 , i2 )
            if Lobj < Hobj - eps: #ε = 10^-3までの誤差を許す
                a2 = L
            elif Lobj > Hobj + eps:
                a2 = H
            else:
                a2 = alph2

        if np.abs( a2 - alph2 ) < eps * ( a2 + alph2 + eps ):
            # 更新量が非常に小さい場合はFalseを返す
            changedStatus = False
        else:
            # 更新量が小さくない場合は、二つ目のラグランジュ乗数を更新する
            a1 = alph1 + s * ( alph2 - a2 )

            # Update threshold to reflect change in Lagrange multipliers
            b1 = E1 + y1 * ( a1 - alph1 ) * k11 + y2 * ( a2 - alph2 ) * k12 + self.b
            b2 = E2 + y1 * ( a1 - alph1 ) * k12 + y2 * ( a2 - alph2 ) * k22 + self.b

            # Store a1 in the alpha array
            self.P[ i1 ] = a1

            # Store a2 in the alpha array
            self.P[ i2 ] = a2

            self.sv = np.where( ( ( self.P > 0 ) & ( self.P < self.C ) ) )[ 0 ]
            if b1 == b2 and L != H:
                self.b = b1 / len( self.sv )

            # Update weight vector to reflect change in a1 & a2, if SVM is linear
            self.w += y1 * ( a1 - alph1 ) * X[ i1 ] + y2 * ( a2 - alph2 ) * X[ i2 ]

            # Update error cache using new Lagrange multipliers
            self.Y_hat = self._decision_function( X , Y )
            self.E = self._E( Y , self.Y_hat )

            # Plot
            self._plot( self.x1_plot , self.x2_plot , X , Y )


        # 更新された場合はTrueを返す
        return changedStatus


    def _examineExample( self , i2 , X , Y ):
        """
        ~takeStepを各サンプルに実行するためのメソッド~

        i2 <int>  : 二つ目のラグランジュ乗数の添字
        Y <numpy> : 教師データ
        """

        E , Y_hat = self.E , self.Y_hat
        changedStep = False
        subscripts = []
        y2 , alph2 , ui2 , E2 = Y[ i2 ] , self.P[ i2 ] , Y_hat[ i2 ] , E[ i2 ]

        r2 = E2 * y2

        if ( ( ( r2 < - self.tol ) and ( alph2 < self.C ) ) or ( r2 > self.tol and alph2 > 0 ) ):

            # 非ゼロ＆非Cのラグランジュ乗数の数＞1 →サポートベクトルがあればその中を優先的に処理する
            if len( self.sv ) > 1:
                i1 = np.argmax( np.abs( E - E[ i2 ] ) )

                if self._takeStep( i1 , i2 , X , Y ):
                    subscripts += [ i1 ]
                    changedStep = True

            # ゼロではない、Cではないすべてのアルファを、ランダムな位置からループする → 上の条件分岐でゼロだった場合は、全てのαのうちから0よりおおきくCより小さいものをサーチする
            if len( subscripts ) == 0:
                subscripts += [ n for n in np.where( ( ( self.P > 0 ) & ( self.P < self.C ) ) )[ 0 ] if self._takeStep( n , i2 , X , Y ) ]

            # 二つ目のラグランジュ乗数であるi1を全てのトレーニングサンプルからサーチする
            if len( subscripts ) == 0:
                subscripts += [ n for n in range( len( self.P ) ) if self._takeStep( n , i2 , X , Y ) ]


        # 更新された数をリターンする
        return len( subscripts )


    def _clip( self , Y , i1 , i2 , P_new ):
        """
        ~更新後のラグランジュ乗数をクリップするメソッド~

        Y <numpy> : 教師データ
        i1 <int>  : 一つ目のラグランジュ乗数の添字
        i2 <int>  : 二つ目のラグランジュ乗数の添字
        P_new <float> : 更新されたラグランジュ乗数
        """

        if Y[ i1 ] == Y[ i2 ]:
            L = max( 0 , self.P[ i2 ] + self.P[ i1 ] - self.C )
            H = min( self.C , self.P[ i2 ] + self.P[ i1 ] )
        else:
            L = max( 0 , self.P[ i2 ] - self.P[ i1 ] )
            H = min( self.C , self.C - ( self.P[ i2 ] - self.P[ i1 ] ) )

        if P_new <= L:
            P_new = L
        elif P_new >= H:
            P_new = H

        return P_new

    def _objective_function( self , X , Y , i1 , i2 ):
        """
        ~ 更新メソッド ~

        X <numpy> : 特徴量
        Y <numpy> : 教師
        i1 <int> : 1つめのラグランジュ乗数の添字
        i2 <int> : 2つめのラグランジュ乗数の添字
        """

        # LとHの時点での目的関数の値を返す
        E , P , C , b = self.E , self.P , self.C , self.b
        y1 , y2 = Y[ i1 ] , Y[ i2 ]
        E1 , E2 = E[ i1 ] , E[ i2 ]
        s = y1 * y2
        k11 = ( X[ i1 ] @ X[ i1 ].T )
        k12 = ( X[ i1 ] @ X[ i2 ].T )
        k22 = ( X[ i2 ] @ X[ i2 ].T )

        f1 = y1 * ( E1 + b ) - P[ i1 ] * k11 - s * P[ i2 ] * k12
        f2 = y2 * ( E2 + b ) - s * P[ i1 ] * k12 - P[ i2 ] * k22

        # LとHを取得
        if Y[ i1 ] == Y[ i2 ]:
            L = max( 0 , P[ i2 ] + P[ i1 ] - C )
            H = min( C , P[ i2 ] + P[ i1 ] )
        else:
            L = max( 0 , P[ i2 ] - P[ i1 ] )
            H = min( C , C - ( P[ i2 ] - P[ i1 ] ) )


        L1 = P[ i1 ] + s * ( P[ i2 ] - L )
        H1 = P[ i1 ] + s * ( P[ i2 ] - H )

        # 目的関数の値を計算する
        Obj_L = L1*f1 + L*f2 + 1/2 * L1**2 * k11 + 1/2 * L**2 * k22 + s*L*L1*k12
        Obj_H = H1*f1 + H*f2 + 1/2 * H1**2 * k11 + 1/2 * H**2 * k22 + s*H*H1*k12

        return Obj_L , Obj_H



    def _decision_function( self , X , Y ):
        """
        ~ 決定境界を出力するメソッド ~

        X <numpy> : 特徴量 < 行数 : サンプル数 , 列数 : 特徴の数 >
        Y <numpy> : 教師データ
        """
        # 決定境界を出力する
        X_sv , Y_sv , P_sv , b , sv = X , Y , self.P , self.b , self.sv
        if len( sv ) > 0:
            X_sv , Y_sv , P_sv = X[ sv ] , Y[ sv ] , P_sv[ sv ]


        return ( X @ X_sv.T ) @ ( P_sv * Y_sv ) + b


    def _E( self , Y , Y_hat ):
        """
        ~ 誤差を出力するメソッド ~
        ※損失関数ではない点に注意

        Y <numpy> : 教師データ
        Y_hat <numpy> : 推定結果
        """
        return Y_hat - Y

    def _plot( self , x1_plot , x2_plot , X , Y ):
        w_ = self.w.reshape( -1 )
        y1 = ( -x1_plot * w_[0] - self.b ) / w_[1]
        y2 = ( 1 -x1_plot * w_[0] - self.b ) / w_[1]
        y3 = ( -1 -x1_plot * w_[0] - self.b ) / w_[1]
        image1 = plt.plot( x1_plot , y1 , color='lightskyblue' )
        image2 = plt.plot( x1_plot , y2 , color='yellow' )
        image3 = plt.plot( x1_plot , y3 , color='yellow' )
        area = plt.fill_between( x1_plot , y2 , y3 , facecolor='y' , color="blue" , alpha=0.3 )
        self.ims.append( image1 + image2 + image3 + [ area ] )



class kernelSVM:

    def __init__( self , C=float("inf") ):
        self.C = C
        self.sv = None
        self.yg = None
        self.alpha = None
        self.b = None


    def fit( self , X , Y , max_epoch=1e6 , tol=1e-3 ):
        alpha = np.zeros_like( Y , dtype=np.float )
        grad = np.ones_like( Y , dtype=np.float )

        for _ in range( int( max_epoch ) ):
            yg = grad * Y # アダマール積
            i , j = self.__mvp( Y , alpha , yg , tol )
            # 停止条件
            if yg[ i ] <= yg[ j ] + tol:
                print( "収束しました" )
                break

            A = self.C - alpha[ i ] if Y[ i ] == 1 else alpha[ i ]
            B = alpha[ j ] if Y[ j ] == 1 else self.C - alpha[ j ]

            # 更新幅のλを計算 lambdaは予約語であるため、deltaという変数名を使用する
            eta = X[ i ] @ X[ i ] - 2*X[ i ] @ X[ j ] + X[ j ] @ X[ j ]
            delta = min( A , B , ( yg[ i ] - yg[ j ] ) / eta )

            # ラグランジュ乗数を更新
            alpha[ i ] += delta * Y[ i ]
            alpha[ j ] -= delta * Y [ j ]

            # 勾配を計算
            grad -= delta*( ( X @ X[ i ] ) - ( X @ X[ j ] ) )

            # 分離直線の切片を計算
            self.b = ( yg[ i ] + yg[ j ] ) / 2

            # サポートベクトルを取得
            self.sv = np.where( alpha > tol )
            self.alpha = alpha[ self.sv ]
            self.X_sv = X[ self.sv ]
            self.Y_sv = Y[ self.sv ]




    def __mvp( self , Y , alpha , yg , tol ):
        Iup = [ yg[ i ] if ( ( Y[ i ] == 1 and alpha[ i ] > tol ) or ( Y[ i ] == -1 and alpha[ i ] < self.C - tol ) ) else float("inf") for i in range( len( Y ) ) ]
        Idown = [ yg[ i ] if ( ( Y[ i ] == -1 and alpha[ i ] > tol ) or ( Y[ i ] == 1 and alpha[ i ] < self.C - tol ) ) else float("-inf") for i in range( len( Y ) ) ]

        i = np.argmax( Idown )
        j = np.argmin( Iup )

        return i , j
