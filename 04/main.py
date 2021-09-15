from svm import LinearSVM
from sklearn import datasets
import sys
import numpy as np
import matplotlib.pyplot as plt

"""
データセットを作成
"""
# データセットを作成


#線形分離不可能なデータセット
iris = datasets.load_iris()
X = iris.data[:,:2]
Y = (iris.target != 0) * 2 - 1



"""
#線形分離可能なデータセット
N = 100
x1_1 = np.ones( N ) + 10 * np.random.random( N )
x1_2 = np.ones( N ) + 10 * np.random.random( N )
x2_1 = -np.ones( N ) -10 * np.random.random( N )
x2_2 = -np.ones( N ) -10 * np.random.random( N )

x1 = np.c_[ x1_1 , x1_2 ]
x2 = np.c_[ x2_1 , x2_2 ]

# データセットを行列の形に変換
X = np.array( np.r_[ x1 , x2 ] )
Y = np.array( np.r_[ np.ones( N ) , -np.ones( N ) ] )
"""


"""
学習
"""
svm = LinearSVM( observe_mode=True , C=30 )
svm.fit( X , Y , max_epochs=300 )
svm.observe( save_path="./graphs/hardmargin.gif" )
