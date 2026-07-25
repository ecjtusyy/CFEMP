# Devlog

## 2026-07-26

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
tests             15 passed
```

还没做：

- 曲面/多界面搜索；
- 摩擦；
- 有限变形本构；
- 弹塑性和损伤；
- 三维穿甲算例。
