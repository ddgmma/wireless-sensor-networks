# WSN 数据融合算法设计与仿真

> 《无线传感网络》期末综合性实验考核

## 项目简介

基于 RSSI 测距的 WSN 移动目标定位，实现并对比三种数据融合算法：
- **综合平均法**（加权最小二乘）
- **贝叶斯估计**（高斯先验×高斯似然 → 后验更新）
- **卡尔曼滤波**（CV 匀速运动模型）

## 环境配置

```bash
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境（Windows）
venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

## 运行

在 VS Code 中打开 `main.ipynb`，选择 `venv` 内核，运行全部 Cell。

## 实验结果

| 方法       |  综合 RMSE  |
| ---------- | :---------: |
| 综合平均法 |   6.666 m   |
| 贝叶斯估计 | **4.511 m** |
| 卡尔曼滤波 |   4.703 m   |

## 文件说明

```
wsn-data-fusion/
├── main.ipynb          # 主程序（报告 + 代码）
├── main_backup.ipynb   # 备份文件
├── requirements.txt    # Python 依赖
└── README.md           # 本文件
```
