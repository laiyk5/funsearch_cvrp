# FunSearch CVRP 开发者文档

本项目使用 Sphinx 构建开发者文档。

## 安装依赖

```bash
cd /Users/laiyk/Dev/Master/AI/AI-Project/AI-Project
uv pip install -e ".[dev]"
# 或: pip install -e ".[dev]"
```

## 构建文档

### macOS/Linux

```bash
cd docs/dev
make html
```

### Windows

```bash
cd docs/dev
make.bat html
```

## 查看文档

构建完成后，打开浏览器访问：

```
build/html/index.html
```

例如：

```bash
# macOS
open build/html/index.html

# Linux
xdg-open build/html/index.html

# Windows
start build/html/index.html
```

## 文档结构

```
source/
├── conf.py          # Sphinx 配置文件
├── index.rst        # 文档主页
├── data.rst         # 数据集说明
└── ...              # 其他章节
```

## 添加新章节

1. 在 `source/` 目录下创建新的 `.rst` 文件
2. 在 `source/index.rst` 的 `toctree` 中添加新文件
3. 重新构建文档

## 常用命令

| 命令 | 说明 |
|------|------|
| `make html` | 构建 HTML 文档 |
| `make clean` | 清理构建文件 |
| `make help` | 显示所有可用命令 |
