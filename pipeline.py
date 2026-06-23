"""smart-doc-parser 统一流水线：PP-OCRv6 快速扫描 + Unlimited-OCR 深度解析。

流水线逻辑：
  - 快速模式 (quick=True)：PP-OCRv6 ONNX Runtime，秒出结果，适合简单文字
  - 深度模式 (默认)：Unlimited-OCR (3B VLM)，单图 gundam 增强，多页连续理解
  - 自动模式：先跑 v6 扫描，检测到复杂内容则唤醒 Unlimited-OCR

ponytail: Paddle 原生推理在 Windows 上有 OneDNN bug，v6 必须用 ONNX Runtime。
  Unlimited-OCR 需要 CUDA GPU（RTX 4070 8GB 够用）。
"""
import json
import os
import tempfile
from pathlib import Path
from typing import Any

ENGINE = "onnxruntime"

# ── Unlimited-OCR 模型管理 ────────────────────────────────────

_ocr_model = None
_ocr_tokenizer = None


def _load_unlimited_ocr():
    """懒加载 Unlimited-OCR 模型，加载后常驻 GPU 内存。"""
    global _ocr_model, _ocr_tokenizer
    if _ocr_model is not None:
        return _ocr_model, _ocr_tokenizer

    import torch
    from transformers import AutoModel, AutoTokenizer

    model_name = "baidu/Unlimited-OCR"
    _ocr_tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    _ocr_model = AutoModel.from_pretrained(
        model_name,
        trust_remote_code=True,
        use_safetensors=True,
        dtype=torch.bfloat16,
    )
    _ocr_model = _ocr_model.eval().cuda()
    return _ocr_model, _ocr_tokenizer


def _parse_unlimited_ocr(image_path: str) -> dict[str, Any]:
    """Unlimited-OCR: 单图深度解析（gundam 模式 + 裁切增强）。

    gundam: base_size=1024, image_size=640, crop_mode=True
    → 对复杂排版（表格/公式/嵌套）效果最好。
    """
    model, tokenizer = _load_unlimited_ocr()
    output_dir = tempfile.mkdtemp(prefix="ocr_")

    import sys, io
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        model.infer(
            tokenizer,
            prompt="<image>document parsing.",
            image_file=image_path,
            output_path=output_dir,
            base_size=1024, image_size=640, crop_mode=True,
            max_length=32768,
            no_repeat_ngram_size=35, ngram_window=128,
            save_results=True,
        )
    finally:
        sys.stdout = old_stdout

    # Unlimited-OCR 输出 result.md 到 output_path 目录
    result_md = Path(output_dir) / "result.md"
    md_content = result_md.read_text(encoding="utf-8").strip() if result_md.exists() else ""

    try:
        import shutil
        shutil.rmtree(output_dir, ignore_errors=True)
    except Exception:
        pass

    return {"markdown": md_content}


def _parse_unlimited_ocr_multi(image_paths: list[str]) -> dict[str, Any]:
    """Unlimited-OCR: 多页连续解析（base 模式，跨页语义关联）。

    base: image_size=1024, crop_mode=False
    → 多页一起理解，输出是连贯的完整文档。
    """
    model, tokenizer = _load_unlimited_ocr()
    output_dir = tempfile.mkdtemp(prefix="ocr_multi_")

    import sys, io
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        model.infer_multi(
            tokenizer,
            prompt="<image>Multi page parsing.",
            image_files=image_paths,
            output_path=output_dir,
            image_size=1024,
            max_length=32768,
            no_repeat_ngram_size=35, ngram_window=1024,
            save_results=True,
        )
    finally:
        sys.stdout = old_stdout

    result_md = Path(output_dir) / "result.md"
    md_content = result_md.read_text(encoding="utf-8").strip() if result_md.exists() else ""

    try:
        import shutil
        shutil.rmtree(output_dir, ignore_errors=True)
    except Exception:
        pass

    return {"markdown": md_content}


# ── v6 快速 OCR ──────────────────────────────────────────────

def _parse_v6(image_path: str) -> dict[str, Any]:
    """PP-OCRv6: 快速文字检测 + 识别（ONNX Runtime 后端）。"""
    from paddleocr import PaddleOCR
    ocr = PaddleOCR(
        engine=ENGINE,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )
    result = list(ocr.predict(image_path))
    texts = []
    for res in result:
        rec_texts = res.get("rec_texts", [])
        rec_scores = res.get("rec_scores", [])
        dt_polys = res.get("dt_polys", [])
        for i, txt in enumerate(rec_texts):
            score = float(rec_scores[i]) if i < len(rec_scores) else 0.0
            poly = dt_polys[i] if i < len(dt_polys) else None
            texts.append({
                "text": txt,
                "confidence": score,
                "poly": poly,
                "complexity": _classify_region(txt, poly),
            })
    return {"texts": texts, "total": len(texts)}


# ── 区域复杂度判断 ────────────────────────────────────────────

FORMULA_CHARS = frozenset({
    "$", "∫", "∑", "√", "∂", "∞", "±", "×", "÷", "=",
    "α", "β", "γ", "δ", "θ", "λ", "μ", "π", "σ", "φ", "ω",
    "Δ", "Ω", "Σ", "Π", "∈", "∉", "⊂", "⊃", "∪", "∩",
})


def _classify_region(text: str, poly: Any) -> str:
    if not text:
        return "text"
    if any(c in text for c in FORMULA_CHARS):
        return "formula"
    if poly is not None:
        try:
            xs = poly[:, 0]; ys = poly[:, 1]
            w = float(xs.max() - xs.min()); h = float(ys.max() - ys.min())
            if h > 30 and w / h > 12:
                return "table_cell"
        except Exception:
            pass
    return "text"


# ── 排版还原（v6 only 时使用）─────────────────────────────────

def _bbox_from_poly(poly: Any) -> tuple[float, float, float, float] | None:
    if poly is None:
        return None
    try:
        xs = poly[:, 0]; ys = poly[:, 1]
        return (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))
    except Exception:
        return None


def _restore_layout(texts: list[dict]) -> str:
    if not texts:
        return ""

    items = []
    for t in texts:
        bbox = _bbox_from_poly(t["poly"])
        if bbox is None:
            items.append({"text": t["text"], "x": 0, "y": 0, "w": 0, "h": 0})
        else:
            x_min, y_min, x_max, y_max = bbox
            items.append({"text": t["text"], "x": x_min, "y": y_min, "w": x_max - x_min, "h": y_max - y_min})

    if not items:
        return ""

    avg_h = sum(it["h"] for it in items) / len(items) if items else 20
    row_tolerance = max(avg_h * 0.6, 15)

    sorted_items = sorted(items, key=lambda it: (it["y"], it["x"]))
    rows: list[list[dict]] = []
    current_row: list[dict] = [sorted_items[0]]
    for it in sorted_items[1:]:
        if abs(it["y"] - current_row[0]["y"]) < row_tolerance:
            current_row.append(it)
        else:
            rows.append(sorted(current_row, key=lambda r: r["x"]))
            current_row = [it]
    rows.append(sorted(current_row, key=lambda r: r["x"]))

    is_table = _detect_table(rows)
    if is_table:
        return _format_as_table(rows)
    else:
        return _format_as_text(rows)


def _detect_table(rows: list[list[dict]]) -> bool:
    if len(rows) < 2:
        return False
    multi_col_rows = [r for r in rows if len(r) >= 2]
    if len(multi_col_rows) < 2:
        return False
    col_counts = [len(r) for r in multi_col_rows]
    if max(col_counts) - min(col_counts) > 1:
        return False
    n_cols = max(col_counts)
    for col_idx in range(n_cols):
        col_x = [row[col_idx]["x"] for row in multi_col_rows if col_idx < len(row)]
        if len(col_x) < 2:
            continue
        avg_x = sum(col_x) / len(col_x)
        if max(abs(x - avg_x) for x in col_x) > 30:
            return False
    return True


def _format_as_table(rows: list[list[dict]]) -> str:
    multi_col_rows = [r for r in rows if len(r) >= 2]
    single_col_rows = [r for r in rows if len(r) == 1]
    if not multi_col_rows:
        return _format_as_text(rows)

    n_cols = max(len(r) for r in multi_col_rows)
    header_cells = [multi_col_rows[0][i]["text"] if i < len(multi_col_rows[0]) else "" for i in range(n_cols)]
    col_widths = [0] * n_cols
    for row in multi_col_rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(cell["text"]))
    separators = ["-" * max(w, 3) for w in col_widths]

    data_lines = []
    for row in multi_col_rows[1:]:
        cells = [row[i]["text"] if i < len(row) else "" for i in range(n_cols)]
        data_lines.append("| " + " | ".join(cells) + " |")

    table = [
        "| " + " | ".join(header_cells) + " |",
        "| " + " | ".join(separators) + " |",
    ] + data_lines
    result = "\n".join(table)

    if single_col_rows:
        notes = "\n".join(r[0]["text"] for r in single_col_rows)
        result += "\n\n" + notes
    return result


def _format_as_text(rows: list[list[dict]]) -> str:
    lines = []
    for row in rows:
        lines.append(" ".join(cell["text"] for cell in row))
    return "\n".join(lines)


# ── 合并输出 ──────────────────────────────────────────────────

def _merge_output(v6: dict, vl: dict | None, output_format: str) -> str:
    # Unlimited-OCR 输出的 markdown 已经还原了排版结构，直接用
    if vl and vl.get("markdown"):
        vl_md = vl["markdown"].strip()
        if output_format == "json":
            return json.dumps({"markdown": vl_md}, ensure_ascii=False, indent=2)
        return vl_md

    # v6 only：用坐标还原排版
    layout_text = _restore_layout(v6["texts"])
    if output_format == "json":
        rows_data = []
        items = v6["texts"]
        sorted_items = sorted(
            [(t, _bbox_from_poly(t["poly"])) for t in items],
            key=lambda pair: (pair[1][1] if pair[1] else 0, pair[1][0] if pair[1] else 0)
        )
        for t, bbox in sorted_items:
            entry = {"text": t["text"]}
            if bbox:
                entry["bbox"] = list(bbox)
            rows_data.append(entry)
        return json.dumps(rows_data, ensure_ascii=False, indent=2)

    return layout_text


# ── 公共入口 ──────────────────────────────────────────────────

def smart_parse(
    image_path: str,
    output_format: str = "markdown",
    mode: str = "auto",  # "auto" | "quick" | "deep"
    on_progress=None,
) -> str:
    """统一 OCR 流水线入口。

    mode:
      - "quick": PP-OCRv6 only，秒出，适合简单文字
      - "deep": Unlimited-OCR only，深度解析，适合复杂文档
      - "auto": 先 v6 扫描，检测到复杂内容则唤醒 Unlimited-OCR
    """
    if not Path(image_path).exists():
        return f"文件不存在: {image_path}"

    def progress(msg):
        if on_progress:
            on_progress(msg)

    # 快速模式：只跑 v6
    if mode == "quick":
        progress("PP-OCRv6 快速识别...")
        v6 = _parse_v6(image_path)
        result = _merge_output(v6, None, output_format)
        progress("完成")
        return result

    # 深度模式：只跑 Unlimited-OCR
    if mode == "deep":
        progress("Unlimited-OCR 深度解析（gundam 模式）...")
        vl = _parse_unlimited_ocr(image_path)
        progress("完成")
        return vl.get("markdown", "") if output_format == "markdown" else json.dumps(vl, ensure_ascii=False, indent=2)

    # 自动模式：先 v6 扫描，有复杂内容则跑 Unlimited-OCR
    progress("阶段 1: PP-OCRv6 快速扫描...")
    v6 = _parse_v6(image_path)

    complex_count = sum(1 for t in v6["texts"] if t["complexity"] != "text")
    if complex_count > 0:
        progress(f"检测到 {complex_count} 个复杂区域，启动 Unlimited-OCR 深度解析...")
        vl = _parse_unlimited_ocr(image_path)
    else:
        progress("未检测到复杂内容，使用 v6 结果")
        vl = None

    result = _merge_output(v6, vl, output_format)
    progress("完成")
    return result


# ── PDF → MD ──────────────────────────────────────────────────

def pdf_to_md(
    pdf_path: str,
    output_dir: str | None = None,
    on_progress=None,
    on_page_done=None,
    mode: str = "deep",  # PDF 默认用 deep（多页连续理解）
) -> str:
    """PDF 书籍 → Markdown 文件。

    mode="deep": Unlimited-OCR 多页连续理解（跨页语义关联，推荐）
    mode="quick": PP-OCRv6 逐页独立 OCR（快但无跨页关联）
    mode="auto": 先 v6 扫描，有复杂内容则 Unlimited-OCR
    """
    import fitz

    pdf_path = str(Path(pdf_path).resolve())
    if not Path(pdf_path).exists():
        raise FileNotFoundError(f"PDF 不存在: {pdf_path}")

    doc = fitz.open(pdf_path)
    total_pages = len(doc)

    def progress(msg):
        if on_progress:
            on_progress(msg)

    if output_dir is None:
        output_dir = str(Path(pdf_path).parent)
    output_path = str(Path(output_dir) / (Path(pdf_path).stem + ".md"))

    progress(f"PDF 共 {total_pages} 页...")

    # deep 模式：Unlimited-OCR infer_multi，多页一起理解
    if mode == "deep":
        progress("Unlimited-OCR 多页连续解析...")

        # 逐页渲染为图片
        image_paths = []
        tmp_dir = tempfile.mkdtemp(prefix="pdf_ocr_")
        for i, page in enumerate(doc):
            out = os.path.join(tmp_dir, f"page_{i+1:04d}.png")
            mat = fitz.Matrix(300 / 72, 300 / 72)
            page.get_pixmap(matrix=mat).save(out)
            image_paths.append(out)
        doc.close()

        vl = _parse_unlimited_ocr_multi(image_paths)

        # 清理临时图片
        try:
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            pass

        md_content = vl.get("markdown", "")
        if md_content:
            # 加上书名标题
            full_md = f"# {Path(pdf_path).stem}\n\n{md_content}"
            Path(output_path).write_text(full_md, encoding="utf-8")
        else:
            Path(output_path).write_text("# 空结果", encoding="utf-8")

        progress(f"完成！已保存 → {output_path}")
        if on_page_done:
            on_page_done(total_pages, total_pages, md_content)
        return output_path

    # quick/auto 模式：逐页独立 OCR
    all_md: list[str] = [f"# {Path(pdf_path).stem}", ""]

    for page_num in range(total_pages):
        page = doc.load_page(page_num)
        pix = page.get_pixmap(dpi=300)
        img_bytes = pix.tobytes("png")
        temp_img = str(Path(output_dir) / f"_page_{page_num + 1:04d}.png")
        Path(temp_img).write_bytes(img_bytes)

        try:
            progress(f"[{page_num + 1}/{total_pages}] 识别中...")
            page_result = smart_parse(temp_img, mode=mode)
        finally:
            try:
                Path(temp_img).unlink()
            except Exception:
                pass

        all_md.append(f"## 第 {page_num + 1} 页")
        all_md.append("")
        all_md.append(page_result)
        all_md.append("")
        all_md.append("---")
        all_md.append("")

        if on_page_done:
            on_page_done(page_num + 1, total_pages, page_result)

    doc.close()
    md_content = "\n".join(all_md)
    Path(output_path).write_text(md_content, encoding="utf-8")

    progress(f"完成！已保存 → {output_path}")
    return output_path
