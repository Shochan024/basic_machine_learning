from svm import linearSVM , kernelSVM
from kernels import RBF , Linear
from sklearn import datasets
import sys
import numpy as np
import matplotlib.pyplot as plt

"""
データセットを作成
"""
# データセットを作成

"""
#線形分離不可能なデータセット
iris = datasets.load_iris()
X = iris.data[:,:2]
Y = (iris.target != 0)  * 2 - 1
"""


#カーネル法を用いないと分類できないもの
"""
moon = datasets.make_moons( n_samples=300 , noise=0.2 , random_state=0 )
X = moon[0]
Y = moon[1] * 2 - 1
"""

#線形分離可能なデータセット
N = 200
x1_1 = np.ones( N ) + 10 * np.random.random( N )
x1_2 = np.ones( N ) + 10 * np.random.random( N )
x2_1 = -np.ones( N ) - 10 * np.random.random( N )
x2_2 = -np.ones( N ) - 10 * np.random.random( N )

x1 = np.c_[ x1_1 , x1_2 ]
x2 = np.c_[ x2_1 , x2_2 ]

# データセットを行列の形に変換
X = np.array( np.r_[ x1 , x2 ] )
Y = np.array( np.r_[ np.ones( N ) , -np.ones( N ) ] )


"""
学習
"""

"""
#svm = LinearSVM( observe_mode=True , C=10 )
svm = kernelSVM( kernel=RBF( gamma=0.5 ) , C=1.0 )
svm.fit( X , Y , max_epochs=200 )
"""

svm = kernelSVM()
svm.fit( X , Y )

#svm = linearSVM( C=float("inf") ) #softmarginはC=73
#svm.fit( X , Y )
#svm.observe( save_path="./graphs/linearSVC.gif" )
