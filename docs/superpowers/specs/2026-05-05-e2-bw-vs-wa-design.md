# E2 BW vs WA Design

## Goal

在现有本地环境、预训练模型和本地 LibriMix 数据可用的前提下，尽快搭建一个最小研究骨架，完成 Conv-TasNet 的 E2 对比实验：

- `fp32_baseline`
- `bw_tcn`
- `wa_tcn`

该版本的目标不是论文级完整复现，而是快速产出可比较结果，为后续 E3/E4 提供稳定起点。

## Confirmed Constraints

- 运行环境是 Windows，项目路径为 `D:\Paper\01`
- Python 环境固定使用 `D:\Paper\01\.venv`
- 深度学习栈固定为：
  - `torch 2.5.1+cu124`
  - `torchaudio 2.5.1+cu124`
  - `asteroid 0.7.0`
- 预训练模型固定为 `JorisCos/ConvTasNet_Libri2Mix_sepclean_8k`
- 数据集固定走 `D:\Paper\datasets\torchaudio_root`
- 优先目标是“尽快跑出可比结果”，不是第一版就完整复现论文流程

## Scope

本次实现只覆盖 E2 所需的最小功能：

1. 安全加载 Asteroid 预训练 Conv-TasNet
2. 基于 `torchaudio.datasets.LibriMix` 读取本地数据
3. 建立统一训练/验证入口
4. 实现三种实验模式：
   - `fp32_baseline`
   - `bw_tcn`
   - `wa_tcn`
5. 输出统一可比指标和 checkpoint

本次不实现：

- BTCN
- 蒸馏
- 模块敏感度分析
- 消融
- 论文级实验编排系统
- 大规模日志平台集成

## Approach Options

### Option 1: 单脚本快跑

把数据加载、模型改造、训练循环都塞进一个入口脚本。

优点：

- 初始开发最快

缺点：

- 很快会在 E3/E4 扩展时失控
- 不利于拆分模型改造和训练逻辑

### Option 2: 最小研究骨架

建立轻量但清晰的目录结构，把数据、预训练加载、二值化补丁、训练引擎分开。

优点：

- 当前版本仍然足够快
- 后续扩展到 BTCN、蒸馏和消融时不需要推倒重来
- 风险和复杂度都可控

缺点：

- 比单脚本多一点初始结构工作

### Option 3: 论文全流程框架

从第一版开始就补全配置系统、日志系统、实验注册和结果聚合。

优点：

- 长期最规范

缺点：

- 明显偏离“先尽快跑出可比结果”的当前目标

## Recommended Approach

采用 Option 2，也就是最小研究骨架。

原因：

- 它保留了当前阶段所需的速度
- 它为后续 E3/E4 留出清晰接口
- 它避免把 E2 的短期实现做成一次性脚本

## Architecture

### Data Layer

由独立模块负责：

- 创建 `LibriMix` 训练集和验证集
- 创建 DataLoader
- 统一 batch 组装格式

第一版固定使用：

- `train-360` 作为训练集
- `dev` 作为验证集
- `task='sep_clean'`
- `sample_rate=8000`
- `mode='min'`

### Model Loading Layer

由独立模块负责：

- 加载 Asteroid 的 `ConvTasNet.from_pretrained(...)`
- 处理预训练 checkpoint 兼容加载
- 返回可用于训练或改造的基模型

这里不在调用处重复兼容逻辑，避免后续每个实验入口各自修补。

### Model Patching Layer

由独立模块负责：

- 定位 Asteroid `model.masker` 中的 TCN 主体
- 只修改 TCN 内部相关卷积或激活
- 保持 Encoder、Decoder 和掩码头为 FP32

三种模式定义如下：

- `fp32_baseline`：不改模型结构
- `bw_tcn`：对 TCN 内卷积权重做二值化前向
- `wa_tcn`：在 `bw_tcn` 基础上，把 TCN 激活替换为 sign 路线

第一版不重写完整 Asteroid 模型，只在最小边界内 patch。

### Training Layer

统一训练引擎负责：

- 单轮训练
- 单轮验证
- AMP 自动混合精度
- checkpoint 保存
- 最优指标追踪

训练策略采用短程微调，不从零训练。

### Evaluation Layer

第一版统一输出：

- 训练 loss
- 验证 loss
- SI-SNR
- SI-SNRi

其中 SI-SNRi 以混合语音作为基线计算提升值，保证 `fp32_baseline`、`bw_tcn`、`wa_tcn` 可直接横向比较。

## File Structure

计划采用以下结构：

```text
src/
  data/
    librimix.py
  models/
    pretrained.py
    binarize.py
    patch_tcn.py
  train/
    losses.py
    engine.py
  utils/
    seed.py
    io.py
configs/
  e2_fp32.yaml
  e2_bw.yaml
  e2_wa.yaml
scripts/
  run_experiment.py
```

各文件职责：

- `src/data/librimix.py`
  - 构建数据集与 DataLoader
- `src/models/pretrained.py`
  - 安全加载预训练 Conv-TasNet
- `src/models/binarize.py`
  - 定义权重二值化和激活二值化的最小算子
- `src/models/patch_tcn.py`
  - 实现 E2 所需的 TCN patch
- `src/train/losses.py`
  - 放训练损失与评估指标计算
- `src/train/engine.py`
  - 训练/验证循环
- `src/utils/seed.py`
  - 固定随机种子
- `src/utils/io.py`
  - 输出目录、checkpoint、结果写盘
- `configs/*.yaml`
  - 管理三类实验配置
- `scripts/run_experiment.py`
  - 唯一实验入口

## Data Flow

每次实验执行流程：

1. 读取配置文件
2. 固定随机种子
3. 创建训练集和验证集
4. 加载预训练 Conv-TasNet
5. 根据实验模式选择是否 patch 为 `BW` 或 `WA`
6. 创建优化器
7. 训练若干 epoch
8. 每个 epoch 在验证集上评估
9. 保存最优 checkpoint
10. 输出最终结果摘要

## Error Handling

第一版要显式处理以下错误：

- 预训练模型加载失败
- 数据根目录不存在或结构不匹配
- 设备不可用时自动退回 CPU 或直接报清晰错误
- patch TCN 时找不到目标模块
- checkpoint 输出目录不存在

错误信息必须直指失败点，避免只抛出底层栈。

## Testing Strategy

当前阶段不追求大规模测试覆盖，但必须有最小验证：

1. 预训练模型能成功加载
2. `LibriMix` 能取到一个 batch
3. `fp32_baseline` 前向通过
4. `bw_tcn` 前向通过
5. `wa_tcn` 前向通过
6. 单个训练 step 能正常反向传播

测试形式可以先用轻量脚本或最小 pytest 用例，重点是防止第一轮训练才暴露结构错误。

## Non-Goals

这一版明确不做：

- 复刻论文所有超参数
- 追求最终最优分数
- 支持多机训练
- 支持所有 LibriMix 变体
- 同时实现 BTCN 和 LSAD

## Exit Criteria

达到以下条件就算 E2 最小实现完成：

1. 三种实验模式都能启动并完成训练
2. 能在统一验证集上输出可比较指标
3. 能保存每组实验最优 checkpoint
4. 结果目录里有结构化摘要，足够支持人工对比 BW 与 WA

