# WSN 数据融合算法设计与仿真

> 《无线传感网络》期末综合性实验考核
> 学号：223062136
> 姓名：杨文焕

## 项目简介

基于 RSSI 测距的 WSN 移动目标定位，实现并对比三种数据融合算法：

- **综合平均法**（加权最小二乘）
- **贝叶斯估计**（高斯先验×高斯似然 → 后验更新）
- **卡尔曼滤波**（CV 匀速运动模型）

## 环境配置

```bash
# 进入项目文件夹 wireless-sensor-networks
cd wireless-sensor-networks/
# 安装依赖
pip install -r requirements.txt
```

## 运行

打开 `main.ipynb`，选择 `venv` 内核，运行全部 Cell。
