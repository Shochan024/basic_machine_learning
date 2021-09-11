from svm import SVM
import sys
import numpy as np
import matplotlib.pyplot as plt

"""
データセットを作成
"""
# データセットを作成
N = 300
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
学習
"""
svm = SVM( observe_mode=True )
svm.fit( X , Y , max_epochs=10 )
svm.observe( save_path="./graphs/hardmargin.gif" )
