# Smart Doc Parser

本地文档智能解析工具 — 拖拽上传，一键识别。PP-OCRv6 + PaddleOCR-VL 统一流水线，自动还原排版，支持图片 / PDF 转 Markdown。

## 功能

- **拖拽上传**：图片 (PNG/JPG) 或 PDF 直接拖入
- **统一流水线**：v6 快速扫描 → 自动检测复杂区域 → VL 深度解析 → 合并输出
- **排版还原**：表格自动输出 Markdown 表格格式，段落按原排版保留
- **PDF 转 MD**：整本书 PDF 逐页识别，输出完整 Markdown 文件
- **桌面应用**：Tkinter 原生窗口，双击即用

## 技术栈

- [PaddleOCR 3.7](https://github.com/PaddlePaddle/PaddleOCR) — PP-OCRv6 (ONNX Runtime) + PaddleOCR-VL
- PyMuPDF — PDF 渲染
- Tkinter — 桌面 GUI

## 安装

```bash
pip install paddlepaddle paddleocr onnxruntime pymupdf pillow
```

## 使用

### 桌面应用

```bash
python desktop.py
```

### 命令行

```python
from pipeline import smart_parse, pdf_to_md

# 单张图片
result = smart_parse("document.png")
print(result)

# PDF 整书转换
output = pdf_to_md("book.pdf")
print(f"输出: {output}")
```
