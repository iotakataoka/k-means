import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

def kmeans(x,k, terations):
    # 選んだindexのデータを取り出しcentroidsとする
    idx = [1, 7, 8]
    centroids = x[idx, :]
    print("0 回目 ")
    print(centroids)

    # centroidsと各データ分のユークリッド距離を計算
    distances = cdist(x, centroids ,'euclidean')

    # 上で計算した距離からcentroidsに一番近い近いほうデータのindexを入れた配列を作る。
    points = np.array([np.argmin(i) for i in distances])

    # centroids更新処理
    for i in range(terations):
        centroids = []
        for idx in range(k):
            #kと同じデータの平均を取る
            temp_cent = x[points==idx].mean(axis=0)
            centroids.append(temp_cent)

        # Centroidsを更新
        centroids = np.vstack(centroids)
        print(i+1, " 回目 ")
        print(centroids)

        #再度距離を計算
        distances = cdist(x, centroids ,'euclidean')
        points = np.array([np.argmin(i) for i in distances])

    return points

def main():
    # テストデータ
    X = np.array([[1, 1], [1, 2], [1, 3], [1, 5], [2, 3], [2, 4], [3, 1], [4, 1], [4, 4], [5, 3], [6, 3], [6, 5]])

    # 実行
    label = kmeans(X, 3, 10)

    #結果を可視化
    u_labels = np.unique(label)
    for i in u_labels:
        plt.scatter(X[label == i , 0] , X[label == i , 1] , label = i)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()