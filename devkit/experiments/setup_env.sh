#!/bin/bash

# 设置项目根目录 - 使用 MINGW64 路径格式
export MEGAFACE_ROOT="/f/insightface"

# 检查目录是否存在
if [ ! -d "$MEGAFACE_ROOT" ]; then
    echo "错误: 目录 $MEGAFACE_ROOT 不存在"
    exit 1
fi

# 添加二进制文件到 PATH（如果 bin 目录存在）
if [ -d "$MEGAFACE_ROOT/bin" ]; then
    export PATH="$MEGAFACE_ROOT/bin:$PATH"
    echo "已添加 bin 目录到 PATH: $MEGAFACE_ROOT/bin"
else
    echo "警告: bin 目录不存在: $MEGAFACE_ROOT/bin"
fi

# 在 MINGW64 中，LD_LIBRARY_PATH 可能不起作用
# 改用 PATH 来包含库文件目录
if [ -d "$MEGAFACE_ROOT/lib" ]; then
    export PATH="$MEGAFACE_ROOT/lib:$PATH"
    echo "已添加 lib 目录到 PATH: $MEGAFACE_ROOT/lib"
else
    echo "警告: lib 目录不存在: $MEGAFACE_ROOT/lib"
fi

# 设置 Python 路径（如果 devkit 目录存在）
if [ -d "$MEGAFACE_ROOT/devkit" ]; then
    export PYTHONPATH="$MEGAFACE_ROOT/devkit:$PYTHONPATH"
    echo "已添加 devkit 到 PYTHONPATH: $MEGAFACE_ROOT/devkit"
else
    echo "警告: devkit 目录不存在: $MEGAFACE_ROOT/devkit"
fi

echo "MegaFace 环境已设置"
echo "MEGAFACE_ROOT: $MEGAFACE_ROOT"
echo "PATH: $PATH"
echo "PYTHONPATH: $PYTHONPATH"