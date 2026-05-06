# Conv-TasNet 实验项目

本项目用于整理基于 Conv-TasNet 的语音分离实验代码。当前仓库中，`E1` 路径已经可运行，训练入口已迁移到 PyTorch Lightning；`E2-E6` 仍是占位入口，后续再补。

## 目录结构

```text
configs\       训练配置（含 configs\kaggle\ 下的 Kaggle 路径模板）
data\          数据集与索引构建
losses\        损失函数
models\        模型定义与 teacher 加载
trainers\      训练与评估入口
utils\         通用工具
tests\         测试
DataIndex\     JSON 索引目录
Experiments\   训练和评估输出
```

## 环境依赖

推荐使用 **Python 3.12**（与当前 Kaggle / Colab 基础镜像一致）。

`requirements.txt` 当前内容（**与 Kaggle 基础镜像同源 Colab 栈**：`release-colab-external-images_20260416`，**PyTorch 2.10.0**；另含本仓库训练/测试所需的固定版本）：

```text
numpy==2.4.4
torch==2.10.0
torchvision==0.25.0
torchaudio==2.10.0
lightning==2.5.1.post0
soundfile==0.13.1
PyYAML==6.0.3
pytest==7.4.4
huggingface_hub==1.14.0
asteroid_filterbanks==0.4.0
tqdm==4.67.1
```

**在 Kaggle Notebook 里**请勿直接全量 `pip install -r requirements.txt`。镜像已在 [Kaggle/docker-python](https://github.com/Kaggle/docker-python) 中合并 Colab 运行时与 `kaggle_requirements.txt`（TensorFlow、s3fs、gcsfs、ydata-profiling 等）；再强行安装其中的 `numpy` / `torch` / `fsspec` 等会与这些预装包的**声明依赖**冲突，pip 会打印一长串 “dependency conflicts”（多为解析器无法在同一环境里同时满足所有元数据，而非你少装了某个包）。请改用 **`requirements-kaggle.txt`**，只补充本项目用到的 `lightning`（`lightning.pytorch`，与镜像里的 `pytorch-lightning` 包名不同）、`asteroid_filterbanks` 等：

```bash
pip install -q -r requirements-kaggle.txt
```

说明：官方镜像构建时会 `uninstall google-cloud-bigquery-storage`（见上述仓库的 `Dockerfile.tmpl`），因此与 `bigquery-storage` 相关的提示可忽略，除非你要自行启用 BigQuery Storage API。若仍遇 pip 解析问题，可尝试 `pip install -r requirements-kaggle.txt --use-deprecated=legacy-resolver`（权宜之计，可能掩盖真实冲突）。

### 与 Kaggle 对齐（本机安装）

在 Kaggle Notebook 中 **`torch` / `numpy` 等已由镜像提供**；仅需按上一节安装 `requirements-kaggle.txt`。在 **Windows 本机**若需 **CUDA 12.8** 的 PyTorch wheel（与 Colab/Kaggle 常用的 cu128 构建一致），请加上 PyTorch 官方额外索引，避免只装到 CPU 轮子：

```powershell
python -m pip install -U pip
python -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128
```

仅 CPU 或已手动装好匹配版本的 `torch` / `torchaudio` 时，可直接：

```powershell
python -m pip install -r requirements.txt
```

### Kaggle 上遇到 `torchvision::nms does not exist`

这通常表示 `torch` 与 `torchvision` 的 wheel 不配套，或其中一个被单独升级过。仓库依赖现在显式固定为：

```text
torch==2.10.0
torchvision==0.25.0
torchaudio==2.10.0
```

在 Kaggle Notebook 里建议直接强制重装这三个包，再启动训练：

```bash
pip install -U --no-cache-dir --force-reinstall torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --extra-index-url https://download.pytorch.org/whl/cu128
```

装完后先做一次导入检查：

```bash
python -c "import torch, torchvision, torchaudio; print(torch.__version__, torchvision.__version__, torchaudio.__version__)"
```

如果这里能正常输出版本号，再运行训练入口：

```bash
python -m trainers.e1_smoke_train --config configs/kaggle/e1_teacher_smoke_train-kaggle.yaml
```

## 数据准备

默认数据根目录示例：

```text
D:\Paper\datasets\MiniLibriMix
```

目录结构要求：

```text
D:\Paper\datasets\MiniLibriMix\
├── train\
│   ├── mix_both\
│   ├── s1\
│   └── s2\
├── val\
│   ├── mix_both\
│   ├── s1\
│   └── s2\
└── test\
    ├── mix_both\
    ├── s1\
    └── s2\
```

## 数据加载（与 TIGER 一致）

与 `D:\Paper\TIGER\look2hear\datas\Libri2Mix16.py` 中 **`Libri2MixDataset` / `Libri2MixModuleRemix`** 相同：**每个 split 对应一个目录 `json_dir`，其下直接放置 `mix_both.json`、`s1.json`、`s2.json`**，wav 的绝对路径写在 JSON 条目中。训练 YAML 使用 **`data.train_dir`**、**`data.valid_dir`**（与 TIGER 配置里的 `train_dir` / `valid_dir` 含义一致）。可选 **`data.root`**：当上述目录里还没有 JSON 时，用原始 MiniLibriMix 根目录自动生成该 split 的索引（等价于先跑 `data.build_index`）。

仍支持 **legacy**：`data.index_root` + `data.index_name` + `data.train_split` / `data.val_split`，解析为 `index_root/index_name/<split>/`。

## 构建索引

`data.libri_mix.build_mini_librimix_index` 与 TIGER 仓库 `DataPreProcess/build_mini_librimix_index.py` 对齐：有限样本时用「各 speaker 取前 N 条 wav 且文件名一致」的选取规则；帧数用 `soundfile.SoundFile` 的长度写入 JSON（`indent=4`）。**默认混合条件子目录为 `mix_both`**（与 `process_librimix.py` 一致），训练配置里使用 `data.speakers: [mix_both, s1, s2]`。若你的数据只有 `mix_clean` 目录，请加 `--speakers mix_clean s1 s2` 并把 YAML 中 `data.speakers` 改为对应三项，然后重新建索引。此前生成的 `mix_clean.json` 需删除或换输出目录后重建，否则会与默认 `mix_both.json` 不一致。

完整索引：

```powershell
python -m data.build_index --in_dir D:\Paper\datasets\MiniLibriMix --out_dir DataIndex\default
```

调试用小索引：

```powershell
python -m data.build_index --in_dir D:\Paper\datasets\MiniLibriMix --out_dir DataIndex\mini_debug --train_count 10 --val_count 5 --test_count 5
```

仅含 `mix_clean` 时的建索引示例：

```powershell
python -m data.build_index --in_dir D:\path\to\MiniLibriMix --out_dir DataIndex\default --speakers mix_clean s1 s2
```

索引目录示例（默认 `mix_both`）：

```text
DataIndex\default\
├── train\
│   ├── mix_both.json
│   ├── s1.json
│   └── s2.json
├── val\
│   ├── mix_both.json
│   ├── s1.json
│   └── s2.json
└── test\
    ├── mix_both.json
    ├── s1.json
    └── s2.json
```

单个 JSON 条目格式：

```json
[
  ["D:\\Paper\\datasets\\MiniLibriMix\\test\\mix_both\\a.wav", 32000]
]
```

## E1 训练

训练配置文件：

- `configs\e1_teacher_smoke_train.yaml`

默认训练命令：

```powershell
python -m trainers.e1_smoke_train --config configs\e1_teacher_smoke_train.yaml
```

快速 CPU smoke：

```powershell
python -m trainers.e1_smoke_train --config configs\e1_teacher_smoke_train.yaml --runtime.device cpu --data.train_max_examples 2 --data.val_max_examples 1
```

### Kaggle Notebook（`configs/kaggle/`）

约定与 `D:\Paper\TIGER\configs` 下 Kaggle 模板一致：**原始数据**挂在 `/kaggle/input/...`，**预处理结果、索引与实验输出**写在 `/kaggle/working/...`；适当提高 `runtime.num_workers`、开启 `pin_memory: true`，GPU 上使用 `runtime.precision: 16-mixed`。

**1）依赖**（在已添加本仓库为 Notebook 数据源、工作目录为仓库根的前提下）：

```bash
pip install -q -r requirements-kaggle.txt
```

**2）预处理 / 建索引示例**（路径按你的 Input 数据集名称修改；`--speakers` 须与数据与 YAML 一致）：

```bash
python -m data.build_index --in_dir /kaggle/input/datasets/qjhkaggle/minilibrimix/MiniLibriMix --out_dir /kaggle/working/DataPreProcess/MiniLibriMix --speakers mix_clean s1 s2
```

**3）建索引**（`--in_dir` 必须与 YAML 里 `data.root` 或索引根一致；`--speakers` 须与 `process_librimix.py` 一致，默认 `mix_both s1 s2`）：

```bash
python -m data.build_index \
  --in_dir /kaggle/working/DataPreProcess/MiniLibriMix \
  --out_dir /kaggle/working/DataIndex/default \
  --speakers mix_both s1 s2
```

快速 mini 索引示例：

```bash
python -m data.build_index \
  --in_dir /kaggle/working/DataPreProcess/MiniLibriMix \
  --out_dir /kaggle/working/DataIndex/mini_debug \
  --speakers mix_both s1 s2 \
  --train_count 10 --val_count 5 --test_count 5
```

| 文件 | 说明 |
|------|------|
| `configs/kaggle/e1_teacher_smoke_train-kaggle.yaml` | E1：`data.train_dir` / `data.valid_dir` 指向含 JSON 的 split 目录（与 TIGER `train_dir`/`valid_dir` 一致） |
| `configs/kaggle/e1_teacher_smoke_train-kaggle-mini.yaml` | 快速联调：`mini_debug` + 限制 train/val 样本数 |
| `configs/kaggle/e2_*` … `e7_*` | 与本地 `configs/e2`–`e7` 占位配置对齐，checkpoint 等路径改为 `/kaggle/working/...` |


训练（在仓库根目录执行，配置使用 POSIX 路径）：

```bash
python -m trainers.e1_smoke_train --config configs/kaggle/e1_teacher_smoke_train-kaggle.yaml
```

评估示例（推荐 TIGER 风格 **`--json_dir`**，指向含 `mix_both.json` 的 test 目录）：

```bash
python -m trainers.evaluate \
  --checkpoint /kaggle/working/Experiments/e1_teacher_smoke_train/best_model.pth \
  --json_dir /kaggle/working/DataIndex/default/test \
  --batch_size 4 \
  --device auto \
  --speakers mix_both s1 s2
```

Legacy：`--index_root` + `--index_name` + `--split` 仍可组合出同一目录。若 test 下尚无 JSON，可加 `--data_root` 指向原始 MiniLibriMix 根目录以触发生成。

若 Kaggle 上 Input 数据集目录名不是 `MiniLibriMix`，请改写预处理命令里的 `--in_dir`；`data.build_index` 的 `--in_dir` 须为含 `train/val/test` 子目录的 MiniLibriMix 根目录。

说明：

- 训练仍由 `python -m trainers.e1_smoke_train` 启动；**迭代与验证流程**由 PyTorch Lightning `Trainer` 驱动（优化步进、验证频率、进度条、`runtime.precision` 等）。
- **产物路径**为固定实验目录布局（实验根目录下的 `best_model.pth`、`checkpoints\`、每次运行的 `results\<时间戳>\` 等），便于对接 `evaluate` 与既有脚本；当前未启用 Lightning 默认的 `lightning_logs\version_*` 与内置 `CSVLogger`。
- `runtime.progress_bar: true`（默认）时只在 **tqdm 进度条** 上展示每个 epoch 的指标，不再额外打印一行文本摘要，避免与进度条重复。
- `runtime.progress_bar: false` 时关闭进度条，改为在验证结束后打印一行 epoch 摘要（`Epoch N: train_loss=..., val_loss=...`）。
- GPU 上可将 `runtime.precision` 设为 `16-mixed`；在 CPU 上会回落为全精度以保证稳定。

当前 `configs\e1_teacher_smoke_train.yaml` 的关键字段（节选）：

```yaml
experiment:
  name: e1_teacher_smoke_train

data:
  train_dir: "DataIndex/mini_debug/train"
  valid_dir: "DataIndex/mini_debug/val"
  root: null
  speakers: [mix_both, s1, s2]

runtime:
  device: auto
  precision: 32
  batch_size: 1
  num_workers: 0
  seed: 42

train:
  lr: 1.0e-5
  epochs: 2
  monitor: val_loss
  monitor_mode: min

output:
  root_dir: "Experiments"
```

命令结束时的简短汇总（与 `trainers.e1_smoke_train` 的 `main()` 一致）：

```text
Done: train_loss=-0.4535, val_loss=-0.4956, best_model_path=Experiments\e1_teacher_smoke_train\best_model.pth
```

## E1 评估

评估入口：

```powershell
python -m trainers.evaluate --checkpoint Experiments\e1_teacher_smoke_train\best_model.pth --json_dir DataIndex\default\test --batch_size 4 --device auto --speakers mix_both s1 s2
```

调试用较小评估：

```powershell
python -m trainers.evaluate --checkpoint Experiments\e1_teacher_smoke_train\best_model.pth --json_dir DataIndex\default\test --batch_size 2 --max_examples 10 --device auto --speakers mix_both s1 s2
```

说明：

- `--checkpoint` 必填
- **`--json_dir`**（TIGER）：含 `mix_both.json`、`s1.json`、`s2.json` 的目录，例如 `DataIndex\default\test`
- 未指定 `--json_dir` 时可用 legacy：`--index_root`、`--index_name`、`--split`
- 可选 **`--data_root`**：仅当对应目录下缺少 JSON 且需从 wav 自动生成索引时使用
- `--speakers` 需与建索引一致，默认 `mix_both s1 s2`
- `--start_index` 和 `--max_examples` 可控制评估子集

## 输出目录

训练与评估输出默认放在 `Experiments\` 下。

E1 训练目录（实验名来自配置中的 `experiment.name`，根目录来自 `output.root_dir`）通常类似：

```text
Experiments\e1_teacher_smoke_train\
├── conf.yml
├── final_metrics.json
├── history.jsonl
├── best_k_models.json
├── best_model.pth          # 评估常用：trainers.evaluate --checkpoint 指向此文件
├── checkpoints\
│   ├── best.ckpt
│   └── last.ckpt
└── results\
    └── 20260506-213320\    # 每次运行一个时间戳子目录
        ├── history.csv
        ├── summary.json
        └── checkpoints\
            └── epoch=001-step=000020-val_loss=-0.4535.ckpt
```

说明：`results\<时间戳>\checkpoints\` 下为按验证 epoch 归档的检查点；实验根目录下的 `best.ckpt` / `last.ckpt` 与 `best_model.pth` 由训练回调同步更新。不会出现 Lightning 默认的 `version_0` 目录。

E1 评估输出通常类似：

```text
Experiments\e1_teacher_eval\
├── final_metrics.json
├── history.jsonl
└── results\
    └── 20260506-XXXXXX_metrics.json
```

## 测试

E1 相关测试：

```powershell
pytest tests\test_e1_smoke_train.py -v
```

轻量回归测试：

```powershell
pytest tests\test_precision.py tests\test_progress.py tests\test_training_artifacts.py -v
```

## 后续入口

当前仓库仍保留以下 trainer 入口：

```powershell
python -m trainers.e2_trainer --config configs\e2_naive_bw_wa.yaml
python -m trainers.e3_trainer --config configs\e3_module_sensitivity.yaml
python -m trainers.e4_trainer --config configs\e4_btcn_train.yaml
python -m trainers.e5_trainer --config configs\e5_output_distill.yaml
python -m trainers.e6_trainer --config configs\e6_lsad_train.yaml
```

这些入口目前仍是占位实现。
