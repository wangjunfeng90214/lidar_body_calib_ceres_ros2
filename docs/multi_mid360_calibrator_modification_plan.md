# multi_mid360_calibrator.cpp 改造说明（双通道 + 多次统计 + chordal mean + 8字复核接口）

## 1. 这次文件改了什么

这版修改是围绕当前主节点在线流程做的工程化增强。当前主流程本来是：

- 点云累计到 `accumulation_time_sec`
- 调 `extractPlaneMeasurementsWithPadding(...)`
- 调 `solveCeres(...)`
- `saveSolverResultYaml(...)`

这次改造成：

- 同一批累计点云同时跑 `raw` 和 `padding` 两个通道
- 两个通道各自独立提面、独立求解、独立严格验收
- 通过统一评分器选择单次最优结果
- 节点支持 `num_repeats` 多次重复标定
- 对多次结果做统计聚合
- 旋转聚合不再平均欧拉角，而是用四元数 chordal mean
- 8 字轨迹法不直接并入静态求解器，而是作为第二阶段动态复核接口挂到批量结果后面

---

## 2. 新增的核心结构体

### 2.1 双通道求解

新增：

- `enum class MeasurementChannel`
- `ChannelSolveResult`
- `DualChannelSolveResult`

作用：

- `raw` 与 `padding` 通道分别保存各自提面是否成功、求解是否成功、是否 strict accept、每平面误差、总点数、评分值
- `DualChannelSolveResult` 负责保存两个通道结果和最终被选中的通道

### 2.2 多次标定统计

新增：

- `CalibrationRunRecord`
- `BatchCalibrationSummary`

作用：

- `CalibrationRunRecord` 保存每一次 run 的完整结果
- `BatchCalibrationSummary` 负责对 accepted runs 做统计聚合、稳定性判断、最终代表 run 选择

### 2.3 8 字轨迹复核

新增：

- `Figure8VerificationConfig`
- `Figure8VerificationResult`

作用：

- 作为动态复核接口，不改静态主求解器
- 当前版本先把参数和接口挂好，后续可接 bag 读轨迹与 IMU/odom 一致性分析

---

## 3. 为什么旋转聚合不能直接平均欧拉角

欧拉角直接平均有几个问题：

- 角度周期性会导致均值跳变
- 不同顺序的欧拉角不满足线性平均的几何意义
- 接近奇异位姿时很不稳定

因此这版改成：

1. 把每次 accepted run 的旋转转成四元数
2. 做符号统一（与参考四元数同半球）
3. 构造 `M = Σ q q^T`
4. 取最大特征值对应特征向量作为平均四元数

这就是 chordal mean。

优点：

- 实现简单
- 比直接平均欧拉角稳
- 对当前这种“小偏差聚合”场景很合适

---

## 4. 双通道是如何工作的

### 4.1 两条通道

- `raw`：直接用原始 ROI 提面
- `padding`：ROI 四周外扩 `padding_roi_margin_m` 再提面

### 4.2 每个通道内部流程

- `extractPlaneMeasurements(...)` / `extractPlaneMeasurementsWithPadding(...)`
- `solveCeres(...)`
- `computeChannelDiagnostics(...)`
- `computeChannelScore(...)`

### 4.3 统一评分

评分逻辑优先级：

1. 没提成功最差
2. 求解不可用次差
3. 未 strict accept 仍保留，但分数明显更差
4. strict accept 后，再根据以下指标精细比较：
   - 最大法向角误差
   - 最大平面距离误差
   - `final_cost`
   - 总 inlier 数

### 4.4 为什么不直接平均 raw 和 padding 的结果

因为两条通道并不是统计独立观测：

- padding 本质上是原始 ROI 的扩展版本
- 误差高度相关
- 一旦 padding 吃进边缘杂点，平均会把偏差写进最终结果

所以这里采取：

- 独立求解
- 独立验收
- 统一评分
- 竞争选择

---

## 5. 多次标定统计是如何工作的

### 5.1 节点行为变化

新增参数：

- `num_repeats`
- `inter_run_idle_sec`
- `max_accepted_final_cost`
- `aggregate_max_std_t_xy_m`
- `aggregate_max_std_t_z_m`
- `aggregate_max_std_rpy_deg`

节点不再是一轮点云累计后立即结束，而是：

- 每累计一轮点云，完成一次 `processOneRun(...)`
- 把结果塞进 `batch_records_`
- 达到 `num_repeats` 后做 `aggregateBatchResults(...)`

### 5.2 accepted run 判据

当前骨架里 `isAcceptedRun(...)` 的判据是：

- `run_success == true`
- `strict_accept == true`
- `final_cost <= max_accepted_final_cost`

后面你还可以继续加：

- 每平面最大误差门限
- raw/padding 两通道差异门限
- ROI 点数 / inlier 比例门限

### 5.3 聚合方式

平移：

- `x/y/z` 分量分别取中位数作为 `robust_t_m`

旋转：

- accepted runs 的四元数做 chordal mean，得到 `robust_q`

最终正式外参：

- 不直接用数学平均结果落库
- 选“最接近 robust 中心”的真实 run，作为 `representative_run`

这样做的优点是：

- 有真实 run 可追溯
- 不会把数学平均带来的虚拟姿态直接写入生产标定值

---

## 6. YAML 输出是怎么扩展的

这版输出从“单次单结果”改成“批量结果 + 运行明细 + 复核结果”。

新增顶层字段：

- `format_version`
- `calibration_mode`
- `final_result`
- `aggregate_statistics`
- `verification`
- `runs`
- `figure8_verification`

### 6.1 final_result

保存最终推荐落库的外参。

### 6.2 aggregate_statistics

保存：

- requested / completed / accepted / rejected
- 平移均值、标准差
- 姿态均值、标准差
- robust median / chordal mean 结果
- pairwise 最大差异
- batch 是否稳定

### 6.3 runs

每个 run 里又保存：

- 当前 run 是否成功
- strict_accept
- selected_channel
- raw_channel 详情
- padding_channel 详情
- 两通道差异

这对后续排查非常关键。

---

## 7. 8 字轨迹法为什么要做成第二阶段复核接口

静态三平面法与 8 字轨迹法解决的问题不同：

- 三平面法：解决静态几何外参
- 8 字法：更适合做动态一致性检查、时间偏移检查、安装松动告警

所以这版采用的策略是：

- 静态批量三平面法给正式标定值
- figure8 只做验证，不直接覆盖静态结果

当前代码里：

- `verifyFigure8(...)` 已经把接口、参数和结果写好
- 现在是 skeleton 版本：先校验路径和挂接位置
- 下一步可继续补成：读取 rosbag、解析 body/lidar odom 与 imu、做时间对齐和轨迹一致性评估

---

## 8. 这版代码里最值得你后续继续补的点

### 8.1 需要尽快补成正式实现的

1. `verifyFigure8(...)`
   - 现在只是骨架
   - 后续要真正读 bag、做轨迹一致性检查

2. `passesStrictAcceptance(...)`
   - 现在是合理骨架，但还可以继续加入：
     - 每平面均值残差
     - 反投影误差
     - 通道差异门限

3. `isAcceptedRun(...)`
   - 现在只用了 cost 和 strict_accept
   - 后续建议再加：
     - 最大法向误差
     - 最大距离误差
     - raw/padding 一致性

### 8.2 若想进一步上生产的增强项

1. 输出 CSV/JSON 调试报告
2. 保存每次 run 的点云快照
3. 增加 “代表 run 与 robust 中心偏差” 字段
4. 给 figure8 加结果等级：pass / warn / fail
5. 支持“先三平面正式初值，再 figure8 局部微调”的可选链路

---

## 9. 推荐的运行参数

建议先用：

```yaml
num_repeats: 8
enable_dual_channel: true
enable_padding_channel: true
padding_roi_margin_m: 0.02
max_accepted_final_cost: 0.05
aggregate_max_std_t_xy_m: 0.003
aggregate_max_std_t_z_m: 0.005
aggregate_max_std_rpy_deg: 0.2
enable_figure8_verification: false
```

如果现场环境 ROI 先验很准，可以把 `padding_roi_margin_m` 缩小到 `0.01`。

如果现场噪声大、安装误差较大，可临时用 `0.02 ~ 0.03`，但要注意 padding 可能引入边缘杂点。

---

## 10. 最后的工程建议

正式生产流程建议固定成：

1. 静态三平面法批量采集 5~10 次
2. raw / padding 双通道竞争
3. 仅保留 strict accept + 统计合格 runs
4. 用 chordal mean + median 得到 robust 中心
5. 选择最接近 robust 中心的真实 run 作为正式外参
6. 上车后再做 8 字轨迹动态复核
7. 若复核失败，先告警，不直接覆盖正式标定值

这套流程比“单次求解 + 直接保存 YAML”更接近生产可用。
