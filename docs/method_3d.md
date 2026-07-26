# Taichi 3D notes

## 算例

两块 \(21\times3\times3\) mm 弹性板以 \(\pm100\) m/s 正碰。左板使用
三维 FEM，右板使用三维 MPM。初始间隙和摩擦系数均为零。

论文在四个横向侧面施加平面应变约束，因此解析解是纵向应力波：

\[
c=\sqrt{\frac{E}{\rho}},
\qquad
\sigma_c=-\rho c v_0,
\qquad
t_\mathrm{sep}=\frac{2L}{c}.
\]

对应值为 \(-1336.97\) MPa 和 \(8.6389\,\mu s\)。

## 三维状态

FEM 节点和 MPM 粒子均保存

\[
\boldsymbol x=(x,y,z),
\qquad
\boldsymbol v=(v_x,v_y,v_z).
\]

小应变和 Cauchy 应力按

\[
[\varepsilon_{xx},\varepsilon_{yy},\varepsilon_{zz},
\gamma_{xy},\gamma_{yz},\gamma_{zx}]
\]

及对应的六个应力分量存储。三维各向同性弹性本构使用 Lamé 常数
\(\lambda,\mu\)。

## HEX8 FEM

左板按 0.5 mm 划分为

\[
42\times6\times6=1512
\]

个 HEX8，节点数为 \(43\times7\times7=2107\)。每个单元使用
\(2\times2\times2\) Gauss 积分，质量集总到八个节点。Taichi 内核完成
速度梯度、应力更新和内力原子装配。

原文公开实现使用单点积分并增加 hourglass 抑制。这里改用全积分，避开未公开
的 hourglass 参数。对于本算例的均匀纵向模态，两者给出相同解析目标；但这项
积分选择是实现差异，不声称逐行复刻原程序。

## 三维 MPM

粒子间距为 0.25 mm，因此右板含

\[
84\times12\times12=12096
\]

个粒子。背景网格尺寸为 0.5 mm，使用八节点三线性形函数。每步执行：

1. 清空背景网格；
2. P2G 映射质量和三维动量；
3. USF 应力更新；
4. 计算三维网格内力；
5. 98% FLIP 与 2% PIC 的 G2P；
6. 丢弃本步网格状态。

## 接触

FEM 接触面含 \(7\times7=49\) 个节点。它们映射到 MPM 网格后形成
hybrid nodes。网格点保存 FEM 与 MPM 两套状态：

\[
(m_I^r,\boldsymbol p_I^r),
\qquad
(m_I^s,\boldsymbol p_I^s).
\]

闭合速度为

\[
c_I=
(\boldsymbol v_I^r-\boldsymbol v_I^s)\cdot\boldsymbol n_I.
\]

当 \(c_I>0\) 时施加最小法向冲量

\[
\boldsymbol J_I=
\frac{m_I^rm_I^s}{m_I^r+m_I^s}
c_I\boldsymbol n_I.
\]

MPM 得到 \(+\boldsymbol J_I\)，FEM 得到
\(-\boldsymbol J_I\)。质量归一化回映射保证两侧总冲量等大反向。每步在
应力更新前和自由推进后各投影一次，对应原文接触力的两部分。

三维 P2G、G2P、FEM 装配、接触面映射、动量投影和冲量回映射均由 Taichi
内核执行；NumPy 只用于初始化、快照整理和绘图。

## 验证边界

当前三维实现覆盖线弹性、无摩擦平直界面和小应变板碰撞。它还不是论文中
的三维穿甲求解器，没有 Johnson-Cook、Mie-Grüneisen EOS、损伤、断裂、
摩擦或曲面多体搜索。

二维版本和尺寸比扫描保留在 `cfemp/plate_impact.py`；`mpm_fem/` 仍是独立
惩罚法路径。
