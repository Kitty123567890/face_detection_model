# Face Recognition and MegaFace Evaluation Project

## 项目概述

该项目提供了一个完整的人脸识别和MegaFace评估流程，使用InsightFace库进行人脸特征提取，并支持在FaceScrub和MegaFace数据集上进行模型性能评估。

## 功能特性

- 基于InsightFace的高精度人脸识别
- 支持FaceScrub和MegaFace数据集处理
- 批量处理图像并提取人脸特征
- 噪声数据过滤和清理功能
- 完整的MegaFace评估流程

## 环境要求

- Python 3.6+
- CUDA (如使用GPU加速)
- 支持的操作系统: Linux, Windows

## 安装依赖

```bash
pip install -r requirements.txt
```



## 项目结构

```
.
├── FaceRecognizer.py      # 人脸识别核心类
├── gen_magaface.py        # 特征提取脚本
├── remove_noises.py       # 噪声数据清理脚本
├── run_test.sh           # 运行测试的Shell脚本
├── data/                 # 数据目录(需自行创建)
│   ├── facescrub_lst     # FaceScrub图像列表
│   ├── megaface_lst      # MegaFace图像列表
│   ├── facescrub_noises.txt # FaceScrub噪声数据列表
│   └── megaface_noises.txt  # MegaFace噪声数据列表
├── feature_out/          # 特征输出目录
├── feature_out_clean/    # 清理后的特征目录
└── results/             # 评估结果目录
```

## 使用方法

### 1. 数据准备

下载并准备以下数据集：
- FaceScrub数据集
- MegaFace数据集

将数据集放置在适当位置，并创建相应的文件列表。

### 2. 运行特征提取

```bash
# 使用GPU加速
python gen_magaface.py --gpu 0 --algo "buffalo_l" \
    --facescrub-root "/path/to/facescrub_images" \
    --megaface-root "/path/to/megaface_images" \
    --output "./feature_out" \
    --facescrub-lst "/path/to/facescrub_lst" \
    --megaface-lst "/path/to/megaface_lst"
```

### 3. 清理噪声数据

```bash
python remove_noises.py --algo "buffalo_l" \
    --feature-dir-input "./feature_out" \
    --feature-dir-out "./feature_out_clean"
```

### 4. 运行评估

使用提供的Shell脚本运行完整评估流程：

```bash
chmod +x run_test.sh
./run_test.sh
```

或者手动运行各个步骤。

## 参数说明

### gen_magaface.py 参数

- `--batch_size`: 批处理大小 (默认: 8)
- `--det_size`: 检测尺寸 (默认: 640)
- `--gpu`: GPU设备ID (默认: 0)
- `--algo`: 模型名称 (默认: 'buffalo_l')
- `--facescrub-lst`: FaceScrub列表文件路径
- `--megaface-lst`: MegaFace列表文件路径
- `--facescrub-root`: FaceScrub图像根目录
- `--megaface-root`: MegaFace图像根目录
- `--output`: 输出目录
- `--nomf`: 是否跳过MegaFace处理

### remove_noises.py 参数

- `--facescrub-noises`: FaceScrub噪声文件路径
- `--megaface-noises`: MegaFace噪声文件路径
- `--algo`: 算法名称 (默认: 'buffalo_l')
- `--facescrub-lst`: FaceScrub列表文件路径
- `--megaface-lst`: MegaFace列表文件路径
- `--feature-dir-input`: 输入特征目录
- `--feature-dir-out`: 输出特征目录

## 注意事项

1. 首次运行时会自动下载InsightFace模型权重
2. 确保有足够的磁盘空间存储提取的特征(约数十GB)
3. 使用GPU可以显著加速处理过程
4. 处理大规模数据集可能需要数小时甚至数天时间

## 结果解读

评估完成后，结果将保存在`results/`目录中，包含各种评估指标和排名信息，可用于模型性能分析和比较。

## 技术支持

如有问题，请参考InsightFace官方文档或提交Issue。

## 许可证

该项目基于Apache 2.0许可证开源。
