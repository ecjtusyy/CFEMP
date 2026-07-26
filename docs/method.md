# NumPy 2D notes

## 算例

复现对象是论文第 5.1 节的对称弹性板碰撞。原文使用三维板，并约束横向
应变，使结果成为一维纵波。这里计算 \(x-y\) 截面：板长 21 mm、宽 3 mm，
另给 3 mm 厚度，采用二维平面应变本构。

这不是一维杆套接口。FEM 节点、MPM 粒子和背景网格都保存二维位置及速度，
应力状态为

\[
\boldsymbol\sigma=
[\sigma_{xx},\sigma_{yy},\tau_{xy}]^\mathsf T.
\]

平面应变矩阵为

\[
\mathbf D=
\begin{bmatrix}
\lambda+2\mu & \lambda & 0\\
\lambda & \lambda+2\mu & 0\\
0&0&\mu
\end{bmatrix},
\quad
\lambda=\frac{E\nu}{(1+\nu)(1-2\nu)},
\quad
\mu=\frac{E}{2(1+\nu)}.
\]

上下边界只约束 \(v_y\)。论文参数 \(\nu=0\)，所以均匀正碰仍应保持横向
均匀；这个性质作为回归检查，而不是通过删除 \(y\) 自由度得到。

## FEM

左板使用双线性 Q4 和 \(2\times2\) Gauss 积分。质量按单元质量的四分之一
集总到节点。速度给定后，

\[
\dot{\boldsymbol\varepsilon}_e=\mathbf B_e\mathbf v_e,
\qquad
\boldsymbol\sigma_e^{n+1}
=\boldsymbol\sigma_e^n
+\mathbf D\dot{\boldsymbol\varepsilon}_e\Delta t.
\]

内力直接由 Gauss 点应力积分：

\[
\mathbf f_e^{\mathrm{int}}
=-\int_{\Omega_e}\mathbf B_e^\mathsf T
\boldsymbol\sigma_e\,\mathrm d\Omega.
\]

## MPM

右板使用双线性背景网格。每步先清空网格，再映射粒子质量和动量：

\[
m_I=\sum_p N_{Ip}m_p,
\qquad
\mathbf p_I=\sum_p N_{Ip}m_p\mathbf v_p.
\]

粒子速度梯度和网格内力为

\[
\mathbf L_p=\sum_I\mathbf v_I\otimes\nabla N_{Ip},
\qquad
\mathbf f_I^{\mathrm{int}}
=-\sum_pV_p\boldsymbol\sigma_p\nabla N_{Ip}.
\]

速度使用 98% FLIP 和 2% PIC；位置用 PIC 网格速度推进。少量 PIC 只用于
压住线性 MPM 的网格穿越噪声。

## 接触

FEM 右边界节点作为 hybrid nodes 映射到 MPM 背景网格。网格点同时保留
两套状态

\[
(m_I^r,\mathbf p_I^r),\qquad
(m_I^s,\mathbf p_I^s),
\]

其中 \(r\) 是 FEM，\(s\) 是 MPM。接触法向
\(\mathbf n_I\in\mathbb R^2\) 由 FEM 表面法向映射并归一化。

无摩擦非穿透条件写成

\[
\left(\mathbf v_I^r-\mathbf v_I^s\right)\cdot\mathbf n_I\le0.
\]

若试算速度违反该条件，施加最小法向冲量

\[
\mathbf J_I=
\frac{m_I^rm_I^s}{m_I^r+m_I^s}
\left[
\left(\mathbf v_I^r-\mathbf v_I^s\right)\cdot\mathbf n_I
\right]_+\mathbf n_I.
\]

MPM 得到 \(+\mathbf J_I\)，FEM 得到 \(-\mathbf J_I\)。网格冲量回映射
使用质量归一化权重，因此两侧动量增量在舍入误差内等大反向。

一次时间步做两次投影：

1. 应力更新前投影，消除论文讨论的半步速度扰动；
2. 内力自由推进后再投影，修正本步内力重新产生的闭合速度。

固定质量和法向时，第二次投影与论文式 (39) 的第二段接触力等价。投影只
在闭合速度为正时激活，所以界面不能传递拉力。

## 验证

默认运行到 15 μs。自动检查包括：

- Q4 常应变 patch；
- MPM 仿射速度场；
- 任意斜向法向的二维动量投影；
- 接触作用量与反作用量；
- \(3\,\mu s\) 应力平台；
- 8.6389 μs 解析分离时间；
- 总动量、总能量和横向均匀性；
- 固定空间网格下的时间步加密。

时间加密使用 40、20、10 ns，并以 5 ns 解为参考。当前应力剖面相对
\(L_2\) 误差拟合斜率为 0.86，只记为“接近一阶”。

论文图 6 的尺寸比实验固定 MPM 网格为 0.5 mm，取
\(R=h_\mathrm{FEM}/h_\mathrm{MPM}=0.5,1,2,3\)。它不作为精度调参：
保留失配网格下的过冲和波前展宽，用来检查网格变粗时误差放大的原文结论。
各组还单独检查总动量和接触冲量平衡。

## 边界

当前二维实现只覆盖线弹性、平面应变、无摩擦、单个平直界面。它不是三维穿甲
求解器，也没有 Johnson-Cook、本构损伤、断裂或多接触搜索。

`mpm_fem/` 是单独的惩罚函数路径，用于比较穿透和参数敏感性。
Taichi 三维板碰撞见 [`method_3d.md`](method_3d.md)。
