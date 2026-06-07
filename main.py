# ============================================================
# 模块1：导入库 + 参数设置
# ============================================================
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os, urllib.request

# 支持中文显示
plt.rcParams['font.sans-serif'] = ['WenQuanYiMicroHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

np.random.seed(42)

# -- 场景参数 --
AREA_SIZE = 100          # 区域大小 100m x 100m
N_ANCHORS = 8            # 锚节点数量
T = 100                  # 仿真步数（秒）
DT = 1.0                 # 采样间隔
NOISE_STD = 2.0          # 基础噪声标准差（米）

# ============================================================
# 模块2：生成仿真场景
# ============================================================

# 2.1 随机生成锚节点位置
anchors = np.random.uniform(0, AREA_SIZE, (N_ANCHORS, 2))

# 2.2 生成目标真实轨迹（S形曲线）
t = np.arange(T)
vx = 0.5 + 0.3 * np.sin(2 * np.pi * t / T)     # 速度变化
vy = 0.4 * np.cos(2 * np.pi * t / T) + 0.2
true_x = 10 + np.cumsum(vx) * DT
true_y = 50 + np.cumsum(vy) * DT
true_traj = np.column_stack([true_x, true_y])

# 2.3 生成带噪声的观测数据
# 每个锚节点有不同的噪声水平
noise_levels = NOISE_STD * np.random.uniform(0.5, 1.5, N_ANCHORS)

# 存储每时刻每锚节点的带噪声距离
noisy_distances = np.zeros((T, N_ANCHORS))
for k in range(T):
    for i in range(N_ANCHORS):
        true_dist = np.linalg.norm(true_traj[k] - anchors[i])
        noise = np.random.randn() * noise_levels[i]
        # 偶尔加野值（5% 概率）
        if np.random.rand() < 0.05:
            noise += np.random.choice([-1, 1]) * true_dist * 0.5
        noisy_distances[k, i] = true_dist + noise

# 从距离转观测位置（用最小二乘法每时刻算一个观测位置）
def distance_to_position(anchors, dists):
    """用最小二乘法从距离估计目标位置"""
    A = np.zeros((N_ANCHORS - 1, 2))
    b = np.zeros(N_ANCHORS - 1)
    for i in range(N_ANCHORS - 1):
        A[i] = 2 * (anchors[i] - anchors[-1])
        b[i] = dists[-1]**2 - dists[i]**2 + \
               anchors[i,0]**2 + anchors[i,1]**2 - \
               anchors[-1,0]**2 - anchors[-1,1]**2
    return np.linalg.lstsq(A, b, rcond=None)[0]

obs_positions = np.zeros((T, 2))
obs_covariances = np.zeros((T, 2, 2))
for k in range(T):
    obs_positions[k] = distance_to_position(anchors, noisy_distances[k])
    # 简单估计观测协方差（对角线）
    avg_noise = np.mean(noise_levels)
    obs_covariances[k] = np.eye(2) * avg_noise**2


# ============================================================
# 模块3：三种融合算法
# ============================================================

# 3.1 综合平均法 —— 加权最小二乘：噪声小的锚节点权重大
def weighted_average_fusion(anchors, noisy_distances, noise_levels):
    """加权最小二乘法：每时刻独立融合，噪声小的锚节点贡献大"""
    est = np.zeros((T, 2))
    for k in range(T):
        # 权重 = 1/噪声方差（噪声越小，权重越大）
        weights = 1.0 / (noise_levels ** 2)
        W = np.diag(weights[:-1])
        
        # 构建加权最小二乘线性系统 A^T W A x = A^T W b
        A = np.zeros((N_ANCHORS - 1, 2))
        b = np.zeros(N_ANCHORS - 1)
        ref = anchors[-1]
        for i in range(N_ANCHORS - 1):
            A[i] = 2 * (anchors[i] - ref)
            b[i] = noisy_distances[k, -1]**2 - noisy_distances[k, i]**2 \
                   + anchors[i,0]**2 + anchors[i,1]**2 \
                   - ref[0]**2 - ref[1]**2
        
        # 加权最小二乘解: x = (A^T W A)^{-1} A^T W b
        ATWA = A.T @ W @ A
        ATWb = A.T @ W @ b
        est[k] = np.linalg.solve(ATWA, ATWb)
    return est

est_avg = weighted_average_fusion(anchors, noisy_distances, noise_levels)

# 3.2 贝叶斯估计 —— 先验×似然 → 后验
def bayesian_fusion(obs_positions, obs_covariances, process_noise=1.0):
    """贝叶斯递推估计，输出位置和不确定性"""
    est = np.zeros_like(obs_positions)
    covs = np.zeros((T, 2, 2))
    # 初始先验
    mu = obs_positions[0]
    Sigma = obs_covariances[0].copy()
    
    est[0] = mu
    covs[0] = Sigma
    
    for k in range(1, T):
        # 先验 = 上一时刻后验 + 过程噪声
        Sigma_prior = Sigma + np.eye(2) * process_noise
        
        # 观测噪声
        R = obs_covariances[k]
        
        # 贝叶斯更新（高斯-高斯共轭）
        K = Sigma_prior @ np.linalg.inv(Sigma_prior + R)
        mu = mu + K @ (obs_positions[k] - mu)
        Sigma = (np.eye(2) - K) @ Sigma_prior
        
        est[k] = mu
        covs[k] = Sigma
    return est, covs

est_bayes, covs_bayes = bayesian_fusion(obs_positions, obs_covariances)

# 3.3 卡尔曼滤波 —— 预测+更新
def kalman_filter(obs_positions, dt=DT, q=0.1, r=4.0):
    """卡尔曼滤波：4维状态 [x, y, vx, vy]"""
    # 状态转移矩阵（匀速模型）
    F = np.array([[1, 0, dt, 0],
                  [0, 1, 0, dt],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])
    # 观测矩阵（只观测位置）
    H = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0]])
    # 过程噪声
    Q = q * np.eye(4)
    # 观测噪声
    R = r * np.eye(2)
    
    est = np.zeros((T, 2))
    # 初始状态
    x = np.array([obs_positions[0, 0], obs_positions[0, 1], 0.0, 0.0])
    P = np.eye(4) * 10
    
    for k in range(T):
        # 预测
        x_pred = F @ x
        P_pred = F @ P @ F.T + Q
        
        # 更新
        z = obs_positions[k]
        y = z - H @ x_pred              # 新息
        S = H @ P_pred @ H.T + R        # 新息协方差
        K = P_pred @ H.T @ np.linalg.inv(S)  # 卡尔曼增益
        
        x = x_pred + K @ y
        P = (np.eye(4) - K @ H) @ P_pred
        
        est[k] = x[:2]
    return est

est_kf = kalman_filter(obs_positions)


# ============================================================
# 模块4：性能评估
# ============================================================

def compute_rmse(true_traj, est_traj):
    """计算均方根误差"""
    diff = true_traj - est_traj
    return np.sqrt(np.mean(diff**2, axis=0))

rmse_avg = compute_rmse(true_traj, est_avg)
rmse_bayes = compute_rmse(true_traj, est_bayes)
rmse_kf = compute_rmse(true_traj, est_kf)

print("=" * 50)
print("RMSE 对比表")
print("=" * 50)
print(f"{'方法':<12} {'X方向(m)':<12} {'Y方向(m)':<12} {'综合(m)':<12}")
print("-" * 50)
print(f"{'综合平均法':<12} {rmse_avg[0]:<12.3f} {rmse_avg[1]:<12.3f} "
      f"{np.sqrt((rmse_avg**2).sum()):<12.3f}")
print(f"{'贝叶斯估计':<12} {rmse_bayes[0]:<12.3f} {rmse_bayes[1]:<12.3f} "
      f"{np.sqrt((rmse_bayes**2).sum()):<12.3f}")
print(f"{'卡尔曼滤波':<12} {rmse_kf[0]:<12.3f} {rmse_kf[1]:<12.3f} "
      f"{np.sqrt((rmse_kf**2).sum()):<12.3f}")


# ============================================================
# 模块5：可视化（3张图）
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# ---- 图1：2D轨迹对比 ----
ax = axes[0]
ax.plot(true_traj[:, 0], true_traj[:, 1], 'k-', linewidth=2, label='真实轨迹')
ax.plot(est_avg[:, 0], est_avg[:, 1], 'r--', linewidth=1, alpha=0.7, label='综合平均法')
ax.plot(est_bayes[:, 0], est_bayes[:, 1], 'b--', linewidth=1, alpha=0.7, label='贝叶斯估计')
ax.plot(est_kf[:, 0], est_kf[:, 1], 'g--', linewidth=1, alpha=0.7, label='卡尔曼滤波')
ax.scatter(anchors[:, 0], anchors[:, 1], c='orange', marker='s', s=80,
           edgecolors='black', linewidths=1, label='锚节点', zorder=5)
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_title('图1：2D轨迹对比')
ax.legend(loc='upper right', fontsize=8)
ax.set_xlim(0, AREA_SIZE)
ax.set_ylim(0, AREA_SIZE)
ax.grid(True, alpha=0.3)

# ---- 图2：位置误差随时间变化 ----
ax = axes[1]
error_avg = np.sqrt(((true_traj - est_avg)**2).sum(axis=1))
error_bayes = np.sqrt(((true_traj - est_bayes)**2).sum(axis=1))
error_kf = np.sqrt(((true_traj - est_kf)**2).sum(axis=1))
ax.plot(error_avg, 'r-', alpha=0.6, label=f'综合平均(RMSE={error_avg.mean():.2f})')
ax.plot(error_bayes, 'b-', alpha=0.6, label=f'贝叶斯(RMSE={error_bayes.mean():.2f})')
ax.plot(error_kf, 'g-', alpha=0.8, label=f'卡尔曼(RMSE={error_kf.mean():.2f})')
ax.set_xlabel('时间步')
ax.set_ylabel('位置误差 (m)')
ax.set_title('图2：位置误差随时间变化')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# ---- 图3：RMSE柱状图对比 ----
ax = axes[2]
methods = ['综合平均法', '贝叶斯估计', '卡尔曼滤波']
rmse_x = [rmse_avg[0], rmse_bayes[0], rmse_kf[0]]
rmse_y = [rmse_avg[1], rmse_bayes[1], rmse_kf[1]]
x_pos = np.arange(len(methods))
width = 0.35
bars1 = ax.bar(x_pos - width/2, rmse_x, width, label='X方向', color='steelblue')
bars2 = ax.bar(x_pos + width/2, rmse_y, width, label='Y方向', color='darkorange')
# 在柱状图上标注数值
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f'{bar.get_height():.2f}', ha='center', fontsize=8)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f'{bar.get_height():.2f}', ha='center', fontsize=8)
ax.set_xticks(x_pos)
ax.set_xticklabels(methods)
ax.set_ylabel('RMSE (m)')
ax.set_title('图3：RMSE对比')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()