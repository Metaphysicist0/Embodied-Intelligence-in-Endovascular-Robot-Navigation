from libsvm.svm import *
from libsvm.svmutil import *
import matplotlib as mat
import matplotlib.pyplot as plt
import numpy as np

# 读取数据
labels, features = svm_read_problem("watermelon_3a.txt")

# 定义不同的γ参数
gamma_values = [0.05, 0.1, 0.5, 1, 5, 10]

# 创建figure和子图
fig, axes = plt.subplots(2, 3, figsize=(12, 8))

# 训练模型并绘制分类图
for i, gamma in enumerate(gamma_values):
    ax = axes.flat[i]
    # 训练模型
    model = svm_train(labels, features, f'-s 0 -t 2 -c 3000 -g {gamma}')
    
    # 获取支持向量的索引
    sv_indices = model.get_sv_indices()
    print(f"γ = {gamma}, 支持向量个数: {len(sv_indices)}")
    
    # 预测结果
    p_label, p_acc, p_val = svm_predict(labels, features, model)
    
    # 获取数据中的第二列和第三列
    x = np.asarray([[mapi[1], mapi[2]] for mapi in features])
    
    # 生成网格点并预测
    N, M = 100, 100  # 设置绘图网格的大小
    x_min, x_max = x.min(axis=0) - 0.1, x.max(axis=0) + 0.1  # 获取x轴和y轴的最小值和最大值
    t1 = np.linspace(x_min[0], x_max[0], N)
    t2 = np.linspace(x_min[1], x_max[1], M)
    grid_x, grid_y = np.meshgrid(t1,t2)
    grid = np.stack([grid_x.flat, grid_y.flat], axis=1)
    y_np = np.zeros((N*M,))
    y_predict, _, _ = svm_predict(y_np, grid, model)
    
    # 绘图
    cm_light = mat.colors.ListedColormap(['#00eaff', '#ff4c4c'])
    ax.pcolormesh(grid_x, grid_y, np.array(y_predict).reshape(grid_x.shape), cmap=cm_light)
    ax.scatter(x[:,0], x[:,1], s=20, c=labels, marker='*')
    ax.set_title(f"γ = {gamma}")
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")

# 调整子图间距并显示
plt.tight_layout()
plt.show()
