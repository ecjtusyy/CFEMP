# Devlog

## 2026-07-26

新增 Taichi 三维主路径：

- 左板改为 1512 个 HEX8，八点 Gauss 积分；
- 右板改为 12096 个三维粒子和三线性背景网格；
- 状态改为三维速度和六分量应力；
- 49 个 FEM 表面节点映射为 hybrid nodes；
- 三维 P2G、G2P、应力和内力计算放进 Taichi CPU 内核；
- 生成三维离散图、中截面应力、能量和分离曲线；
- 二维版本保留作快速回归和尺寸比扫描。

三维默认结果：

```text
HEX8 elements      1512
MPM particles      12096
stress error       1.66 %
separation error   0.48 %
max energy error   2.55 %
tests              23 passed
```

## 2026-07-26 - 2D

第一版把板碰撞直接约化成了一维杆。应力和分离时间能对上，但它不能验证
二维离散和二维接触，这个版本不再作为主实现。

本次改动：

- FEM 改为二维 Q4，\(2\times2\) Gauss 积分；
- MPM 改为二维粒子和双线性背景网格；
- 状态量改为二维速度与
  \([\sigma_{xx},\sigma_{yy},\tau_{xy}]\)；
- hybrid nodes 映射到背景网格后保留 FEM/MPM 两套动量；
- 接触改为二维质量范数法向投影；
- 冲量回映射增加质量归一化，界面动量残差降到舍入误差；
- 增加 Q4 patch、MPM 仿射场和斜向接触测试；
- 增加二维应力场和时间步加密结果。

当前默认结果：

```text
Q4 elements       252
MPM particles     1008
stress error      1.66 %
separation error  0.48 %
max energy error  2.55 %
time slope        0.86
tests             16 passed
```

补做论文图 6 的二维尺寸比扫描。MPM 网格固定 0.5 mm，FEM 网格取
\(R=0.5,1,2,3\)，结果写入 `mesh_ratio_study.*`。

还没做：

- 曲面/多界面搜索；
- 摩擦；
- 有限变形本构；
- 弹塑性和损伤；
- 三维穿甲算例。
