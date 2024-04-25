import numpy as np
import matplotlib.pyplot as plt

# 数据
X = np.array([[0.666, 0.091], 
              [0.243, 0.267],
              [0.244, 0.056], 
              [0.342, 0.098],
              [0.638, 0.16],
              [0.656, 0.197],
              [0.359, 0.369],
              [0.592, 0.041],
              [0.718, 0.102],
              [0.697, 0.46],
              [0.774, 0.376],
              [0.633, 0.263], 
              [0.607, 0.317],
              [0.555, 0.214],
              [0.402, 0.236],
              [0.481, 0.149],
              [0.436, 0.21],
              [0.557, 0.216]])

y = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])

# 线性判别分析
def lda(X, y):
    # 计算类别均值向量
    class_means = []
    for class_val in np.unique(y):
        class_means.append(np.mean(X[y == class_val], axis=0))
    class_means = np.array(class_means)

    # 计算内部散布矩阵
    S_W = np.zeros((X.shape[1], X.shape[1]))
    for class_val, mean_vec in zip(np.unique(y), class_means):
        class_scatter = np.cov(X[y == class_val].T, bias=True)
        S_W += (X[y == class_val].shape[0]) * class_scatter
    
    # 计算类间散布矩阵
    overall_mean = np.mean(X, axis=0)
    S_B = np.zeros((X.shape[1], X.shape[1]))
    for class_val, mean_vec in zip(np.unique(y), class_means):
        n = X[y == class_val].shape[0]
        mean_diff = (mean_vec - overall_mean).reshape(-1, 1)
        S_B += n * (mean_diff @ mean_diff.T)

    # 计算LDA权重向量
    coef = np.linalg.solve(S_W, S_B.T)
    coef = coef[:, 0]  # 取第一个特征向量
    
    # 计算阈值
    threshold = np.dot(overall_mean, coef)
    for class_val, mean_vec in zip(np.unique(y), class_means):
        threshold -= (np.dot(mean_vec, coef) * X[y == class_val].shape[0])
    threshold /= X.shape[0]

    return coef, class_means, threshold

# 执行LDA
coef, class_means, threshold = lda(X, y)

# 打印LDA模型的权重向量和均值向量
print("LDA Coefficients (Weights):\n", coef)
print("\nLDA Mean vectors:\n", class_means)

# 打印分类阈值
print("\nClassification Threshold: ", threshold)

# 计算分数
y_pred = np.dot(X, coef.T) > threshold
score = np.sum(y_pred == y) / X.shape[0]
print("\nLinear Discriminant Analysis Score:", score)

# 设置画布
fig, ax = plt.subplots(figsize=(5, 4))

# 绘制散点图
scatter1 = ax.scatter(X[y==0, 0], X[y==0, 1], c='r', edgecolor='k', s=50, label='Class 0')
scatter2 = ax.scatter(X[y==1, 0], X[y==1, 1], c='b', edgecolor='k', s=50, label='Class 1')

# 绘制判别直线
x1 = np.arange(0, 1, 0.01)
x2 = -(coef[0]*x1 + threshold)/coef[1]
ax.plot(x1, x2, 'k', linewidth=2)

# 设置坐标轴
ax.set_xlabel('Feature 1', fontsize=14)
ax.set_ylabel('Feature 2', fontsize=14)
ax.tick_params(axis='both', labelsize=12)

# 设置图例
ax.legend(handles=[scatter1, scatter2], fontsize=12, loc='upper left', framealpha=1)

# 设置边框和网格
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(linestyle='--', alpha=0.5)

# 调整画布
plt.tight_layout()
plt.show()
