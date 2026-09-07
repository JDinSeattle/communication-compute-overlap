# 面试与简历证据

当前可以使用的表述：

> Implemented and compiled an MSCCL++ PortChannel extension comparing serial execution with two chunking configurations; developed buffer-lifetime counterexamples and passed 72 real GPU compute/reference cases under NVIDIA Compute Sanitizer. Two-GPU communication validation remains pending.

面试可演示：让 CPU 模型给出缺少 acknowledgement 时的覆盖反例；逐条解释 `put`、`signal`、`wait`、`flush` 的作用；展示 host 参考采用仿射复合而非照抄 GPU 循环；运行真实单卡 sanitizer；展示双卡门禁的阻断日志。

需要主动讲清：只有一个 CTA，避免依赖 occupancy 的全 grid 自旋同步；完整 source slice 在 DMA 完成前不可改写；source local completion 与 destination consumer completion 是不同的生命周期边界；CPU 模型不能证明设备内存顺序。

当前不能填写：双卡正确性已通过、通信计算重叠加速比、真实训练吞吐提升、RDMA/NVLink 优化或硬件 profile 结论。完成目标环境验收后，用真实拓扑、样本规模和有利/无收益/退化区域替换这些待办项。
