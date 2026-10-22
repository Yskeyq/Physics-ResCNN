# Physics-ResCNN，1.2.0

本仓库是论文第一轮返修对应的软件发布版。相关论文仍在审稿中。

- 代码仓库：[Yskeyq/Physics-ResCNN](https://github.com/Yskeyq/Physics-ResCNN)
- 数据与模型权重：[IEEE DataPort 数据集，DOI：10.21227/4k9h-8z46](https://doi.org/10.21227/4k9h-8z46)

自有代码采用 MIT；自有合成数据和模型权重采用 CC BY 3.0 US，第三方材料保留原许可。数据与权重已存放于 IEEE DataPort Standard Dataset，文件下载可能需要订阅；这不等于所有人均可免费下载的开放访问数据集。私有 PE 源码、原始 ERA5 下载、第三方 COARE 源码、内部通信和返修 PDF 均不在代码包内。

## 快速使用

1. 安装 Python 3.10+、适合本机的 PyTorch 和 `requirements.txt` 中的依赖。
2. 把单独取得的数据包解压至本仓库 `data/`，或用环境变量 `PHYSICS_RESCNN_DATA` 指向其解压目录。
3. 运行 `python -m unittest discover -s tests -v`。
4. 运行 `python src/predict.py --input data/examples/clutter_validation_first8.npz --output artifacts/predictions.npz`。

推理只读取 500 点中心化杂波，输出 101 点掩膜后 EDH、二值 mask 和概率；不读取标签、PE 或前向模型。中心化指每条 500 点曲线减自身均值，不能先用验证／测试数据重新拟合归一化。

模型张量与原检查点逐项核对一致；公开副本剔除了优化器状态和本机路径，所以整个 checkpoint 文件的哈希与旧文件不同。冻结主模型不会被新训练覆盖。

英文 README 提供正式训练、独立前向训练、检查点审核和绘图入口。所有训练默认只读开发角色，联合测试必须显式确认才能打开，不用于调参或选点。短流程测试与正式实验结果严格区分。

使用本项目时，请引用本代码仓库与上述 DataPort 数据集，并遵守各自许可及第三方署名要求。该 DOI 标识数据集，不是论文 DOI；相关论文仍在审稿中。

## 补充实验与缓存

数据包含名义比较、消融、18 组敏感性检查点、空间实验匹配的区域正演/逆网络、
挑战数据、方向响应原始缓存和 120 剖面直接 PE 重构轨迹。共 48 个检查点导出文件，
其中部分是同一冻结模型在不同目录下的重复组织，不等于 48 次独立初始化。

```
python src/audit_cached_results.py
python src/reproduce_training.py --suite sensitivity
python src/reproduce_training.py --suite spatial
python src/reproduce_training.py --suite fixed-reference
```

第一条只复算冻结缓存，不执行新测试推理或 PE；后三条只打印训练计划。
显式增加 --execute 才执行完整训练，拒绝覆盖已有输出目录。不同硬件重新训练
未必逐位一致，论文结果以冻结检查点和记录为准。固定雷达输入与物理损失内部
的固定参考雷达项是两个不同控制实验，不能混淆。
