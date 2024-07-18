import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import hilbert
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from zjb.main.data.series import TimeSeries


def compute_phase_coherence(timeSeries: TimeSeries):
    """
    计算相位一致性矩阵
    Parameters
    ----------
    timeSeries: TimeSeries
                时间序列

    Returns
    ----------
    phase_coherence: np.ndarray
                    相位一致性矩阵
    """
    bold_signal = timeSeries.data.T

    # 应用希尔伯特变换
    Hilbert_transformed = hilbert(bold_signal)
    # 计算瞬时相位
    instantaneous_phase = np.angle(Hilbert_transformed)
    # 初始化相位一致性矩阵
    phase_coherence = np.zeros(
        (bold_signal.shape[1], bold_signal.shape[0], bold_signal.shape[0])
    )
    # 对于所有时间点计算相位一致性
    for t in range(bold_signal.shape[1]):
        for i in range(bold_signal.shape[0]):
            for j in range(bold_signal.shape[0]):
                phase_difference = np.abs(
                    instantaneous_phase[i, t] - instantaneous_phase[j, t]
                )
                phase_coherence[t, i, j] = np.cos(phase_difference)
    return phase_coherence


def pms(timeSeries: TimeSeries, k_value=3, show_clusters=True):
    """
    计算 probabilistic metastable substates

    Parameters
    ----------
    timeSeries: TimeSeries
                时间序列
    k_value: int
            聚类的数量
    show_clusters: Bool
                   是否展示聚类结果

    Returns
    ----------
    centers: np.ndarray
             聚类中心对应的向量
    reduced_data: np.ndarray
                  时间序列降维后的数据
    clusters: list
              数据对应的状态的标签
    probabilities: list
                   每个状态的概率
    """
    # 计算相位一致性矩阵
    phase_coherence = compute_phase_coherence(timeSeries)

    # 特征值分解
    eigenvalues, eigenvectors = np.linalg.eigh(phase_coherence)

    # 选择主要特征向量（选择前k个特征向量）
    k = 1
    leading_eigenvectors = eigenvectors[:, k]

    # 使用PCA降维到3维空间
    pca = PCA(n_components=3)
    reduced_data = pca.fit_transform(leading_eigenvectors)

    # 聚类分析
    kmeans = KMeans(n_clusters=k_value)
    clusters = kmeans.fit_predict(leading_eigenvectors)
    if show_clusters == True:
        # 绘制可视化聚类结果
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")
        scatter = ax.scatter(
            reduced_data[:, 0],
            reduced_data[:, 1],
            reduced_data[:, 2],
            c=clusters,
            cmap="viridis",
        )

        # 计算并绘制聚类中心
        centers = kmeans.cluster_centers_
        ax.scatter(
            centers[:, 0],
            centers[:, 1],
            centers[:, 2],
            c="black",
            s=30,
            alpha=0.75,
            marker="+",
        )

        legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
        ax.add_artist(legend1)
        ax.set_title("PMS Clustering in 3D PCA Space")
        ax.set_xlabel("PCA Component 1")
        ax.set_ylabel("PCA Component 2")
        ax.set_zlabel("PCA Component 3")
        ax.set_title("PMS Clustering in 3D PCA Space")
        plt.show()

        ax.add_artist(legend1)
        ax.set_title("PMS Clustering in 3D PCA Space")
        plt.show()

    probabilities = []
    for i in range(k_value):
        probabilities.append(np.sum(clusters == i) / len(clusters))

    return centers, reduced_data, clusters, probabilities
