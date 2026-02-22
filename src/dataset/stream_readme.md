# RE10K Stream Dataset (流式数据切分器)

## 简介
这是一个适用于 RealEstate10K 数据集的 PyTorch Dataset 实现。它旨在模拟**流式 (Streaming) 3D 重建**场景，将长视频切分为连续的时间步 (Time Steps)，并提供符合特定规则的输入 (Context) 和目标 (Target) 帧。

## 核心功能
1. **流式切分**：将视频的前 5 秒划分为 5 个独立的时间步 (T1-T5)。
* Input Shape: [5, input_num, 3, H, W]（5秒，每秒 input_num 帧）。
* Target Shape: [5, 15, 3, H, W]（5秒，每秒固定 15 帧）。

2. **输入策略**：每秒随机抽取 `X` 帧作为input输入 (默认为 4)。
* 参数化：支持通过 input_num 参数灵活调整每秒输入帧数（默认 4 帧）。
* 随机性：每秒内的输入帧是从该秒的所有可用帧中随机抽取的，模拟真实场景中的不确定性。

3. **目标策略**：每秒固定抽取单数帧 [1, 3, ..., 29] 共 15 帧作为pool备选池，并按照 `1+2+4+8` 的滑动窗口逻辑进行跨秒组合。
* 均匀采样：采用查表法实现最大间隔采样，确保目标帧在时间轴上均匀分布。

4. **鲁棒性**：自动跳过长度不足 5 秒（150 帧）的短视频，兼容不同格式的索引文件 (index.json) 和原始二进制数据 (BytesIO 解码)。

## 文件结构
* `re10k_dataset.py`: 核心代码文件，包含 `RE10KDataset` 类。

## 数据格式说明
从 DataLoader 中取出的 batch 是一个字典，包含以下关键字段：
* input_images: [B, 5, 4, 3, H, W]. B: Batch Size; 5: 5 个时间步 (T1...T5); 4: 每秒的输入帧数
* target_images: [B, 5, 15, 3, H, W]. 15: 每秒的目标帧数 (固定)
* input_cams: [B, 5, 4, ...] 相机参数; target_cams: [B, 5, 15, ...] 相机参数

ps: 保存图片逻辑在代码中注释部分，如需保存可自行启用。

## 如何在您的训练代码中使用？

### 1. 引入文件
将 `re10k_dataset.py` 复制到您的项目目录下。

### 2. 替换 Dataset
在您的训练脚本 (如 `train.py`) 中，找到定义 Dataset 的地方，替换为本 Dataset。

```python
from re10k_dataset import RE10KDataset
from torch.utils.data import DataLoader

# 初始化数据集
# data_root: 指向包含 .torch 文件和 index.json 的目录
# input_num: 每秒输入的帧数 (默认 4)
train_dataset = RE10KDataset(
    data_root="/path/to/re10k", 
    input_num=4
)

# 创建 DataLoader 
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4) 