**项目 C：Communication–Compute Overlap——验证重叠何时产生收益**

问题：通信与计算同时发起可能争抢 SM、内存带宽或 proxy CPU，最终比串行更慢；同步遗漏则可能偶发读到旧数据。

范围：MSCCL++ 中选择一个两卡模式，例如分块计算后传输，或一个小型 all-reduce；采用项目已有 channel 与示例。只比较一条串行基线与两种 chunk/同步配置，不从零设计通用 collective DSL。

结构：host bootstrap → registered buffer/channel → GPU 计算块 → 明确信号与完成条件 → 消费端校验。写清 `put`、`signal`、`wait`、`flush` 各自保证什么，以及何时允许源缓冲区复用；不能用延时碰巧规避竞态。

阶段：CPU 先画 happens-before 和 buffer 生命周期；双卡先用重复序号及 checksum 检测陈旧数据，再引入计算；最后扫描消息与计算强度，分别记录串行耗时、融合耗时、GPU/CPU 占用及 profile。正确性不过即停止性能比较。

验收：不同执行交错与迭代中无错读/死锁；所有记录可定位实际通道和硬件；报告有利区、无收益区和退化区。交付设计说明、一个可运行扩展示例、独立参考、时序与性能证据。规划 70–130 小时及 10–25 双卡节点小时；需受支持的 P2P/NVLink 或目标 NIC，CPU 模型不能证明设备内存顺序。[MSCCL++](https://github.com/microsoft/mscclpp)

