"""smart-doc-parser 统一流水线：v6 快速扫描 + VL 深度解析，还原排版。

流水线逻辑：
  1. PP-OCRv6 (ONNX Runtime) 快速扫描全部文字区域
  2. 根据区域特征判断是否包含复杂内容（表格/公式/图表）
  3. 如有复杂区域，PaddleOCR-VL 深度解析整页结构
  4. 合并输出：VL 提供结构布局，v6 提供字符级精度 + 坐标还原排版

ponytail: Paddle 原生推理在 Windows PaddlePaddle 3.x 上有 OneDNN bug，
  必须用 engine='onnxruntime'。这是实锤，不是假设。
"""
import json
from pathlib import Path
from typing import Any

ENGINE = "onnxruntime"  # ponytail: Paddle Inference 在 Windows 崩，ONNX 是唯一选择


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


# ── VL 深度解析 ──────────────────────────────────────────────

def _parse_vl(image_path: str) -> dict[str, Any]:
    """PaddleOCR-VL 1.6: 深度文档解析（表格/公式/图表/印章）。

    ponytail: PaddleOCRVL 是独立类，不是 PaddleOCR(pipeline="doc_parse")。
    实锤：PaddleOCRVL() 构造需要 paddlex[ocr] 依赖。
    """
    from paddleocr import PaddleOCRVL
    vl = PaddleOCRVL()  # 默认 v1.6
    result = list(vl.predict(image_path))
    markdown_content = ""
    for res in result:
        # VL 输出 markdown 格式的结构化内容
        md = getattr(res, "markdown", None) or res.get("markdown", "")
        if not md:
            md = res.get("content", str(res))
        markdown_content += md
    return {"markdown": markdown_content.strip()}


# ── 区域复杂度判断 ────────────────────────────────────────────

FORMULA_CHARS = frozenset({
    "$", "∫", "∑", "√", "∂", "∞", "±", "×", "÷", "=",
    "α", "β", "γ", "δ", "θ", "λ", "μ", "π", "σ", "φ", "ω",
    "Δ", "Ω", "Σ", "Π", "∈", "∉", "⊂", "⊃", "∪", "∩",
})


def _classify_region(text: str, poly: Any) -> str:
    """判断文字区域是否属于复杂内容。

    启发式（ponytail: 实锤——普通文字 ratio 也能到 6~8，阈值不能太低）:
      - 公式符号 → formula
      - 宽高比 > 12 且高度 > 30px → table_cell
      - 其他 → text
    """
    if not text:
        return "text"

    if any(c in text for c in FORMULA_CHARS):
        return "formula"

    if poly is not None:
        try:
            xs = poly[:, 0]
            ys = poly[:, 1]
            w = float(xs.max() - xs.min())
            h = float(ys.max() - ys.min())
            if h > 30 and w / h > 12:
                return "table_cell"
        except Exception:
            pass

    return "text"


# ── 排版还原 ──────────────────────────────────────────────────

def _bbox_from_poly(poly: Any) -> tuple[float, float, float, float] | None:
    """从 poly (4,2) 提取 (x_min, y_min, x_max, y_max)。"""
    if poly is None:
        return None
    try:
        xs = poly[:, 0]
        ys = poly[:, 1]
        return (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))
    except Exception:
        return None


def _restore_layout(texts: list[dict]) -> str:
    """用坐标信息还原原始排版。

    核心思路：把文字按 y 坐标分行，按 x 坐标排列，
    然后根据列分布判断是否是表格结构。
    """
    if not texts:
        return ""

    # 1. 提取 bbox + 文字
    items = []
    for t in texts:
        bbox = _bbox_from_poly(t["poly"])
        if bbox is None:
            # 没有 bbox 的文字单独一行
            items.append({"text": t["text"], "x": 0, "y": 0, "w": 0, "h": 0})
        else:
            x_min, y_min, x_max, y_max = bbox
            items.append({
                "text": t["text"],
                "x": x_min, "y": y_min,
                "w": x_max - x_min, "h": y_max - y_min,
            })

    if not items:
        return ""

    # 2. 按 y 坐标分行（容差 = 平均高度的 0.6）
    avg_h = sum(it["h"] for it in items) / len(items) if items else 20
    row_tolerance = max(avg_h * 0.6, 15)

    # 排序后按 y 分组
    sorted_items = sorted(items, key=lambda it: (it["y"], it["x"]))
    rows: list[list[dict]] = []
    current_row: list[dict] = [sorted_items[0]]

    for it in sorted_items[1:]:
        # 如果 y 差距 < 容差，归入同一行
        if abs(it["y"] - current_row[0]["y"]) < row_tolerance:
            current_row.append(it)
        else:
            rows.append(sorted(current_row, key=lambda r: r["x"]))
            current_row = [it]
    rows.append(sorted(current_row, key=lambda r: r["x"]))

    # 3. 判断是否是表格结构
    # 表格特征：多行多列，列位置对齐（每行的 x 起始位置有规律重复）
    is_table = _detect_table(rows)

    # 4. 格式化输出
    if is_table:
        return _format_as_table(rows)
    else:
        return _format_as_text(rows)


def _detect_table(rows: list[list[dict]]) -> bool:
    """检测行结构是否是表格。

    表格特征：
      - 至少 2 行（核心行）有 2+ 列
      - 核心行列数一致
      - 列的 x 位置有规律对齐
    ponytail: 不要求所有行都有 N 列——备注行/标题行只有 1 列也正常。
      只看多列行是否构成网格。
    """
    if len(rows) < 2:
        return False

    # 只看有 2+ 列的行
    multi_col_rows = [r for r in rows if len(r) >= 2]
    if len(multi_col_rows) < 2:
        return False

    # 核心行列数一致（允许 1 列偏差）
    col_counts = [len(r) for r in multi_col_rows]
    max_cols = max(col_counts)
    min_cols = min(col_counts)
    if max_cols - min_cols > 1:
        return False

    # 检查核心行各列的 x 是否对齐（容差 30px）
    n_cols = max_cols
    for col_idx in range(n_cols):
        col_x_positions = []
        for row in multi_col_rows:
            if col_idx < len(row):
                col_x_positions.append(row[col_idx]["x"])
        if len(col_x_positions) < 2:
            continue
        avg_x = sum(col_x_positions) / len(col_x_positions)
        max_dev = max(abs(x - avg_x) for x in col_x_positions)
        if max_dev > 30:
            return False

    return True


def _format_as_table(rows: list[list[dict]]) -> str:
    """Markdown 表格格式输出。单列行（备注/标题）跟在表格后面。"""
    if not rows:
        return ""

    # 分出多列行（表格主体）和单列行（备注/标题）
    multi_col_rows = [r for r in rows if len(r) >= 2]
    single_col_rows = [r for r in rows if len(r) == 1]

    if not multi_col_rows:
        # 没有真正的表格行，退回纯文本
        return _format_as_text(rows)

    n_cols = max(len(r) for r in multi_col_rows)

    # 表头行（多列行的第一行）
    header = multi_col_rows[0]
    header_cells = [header[i]["text"] if i < len(header) else "" for i in range(n_cols)]

    # 分隔线
    col_widths = [0] * n_cols
    for row in multi_col_rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(cell["text"]))
    separators = ["-" * max(w, 3) for w in col_widths]

    # 数据行
    data_lines = []
    for row in multi_col_rows[1:]:
        cells = [row[i]["text"] if i < len(row) else "" for i in range(n_cols)]
        data_lines.append("| " + " | ".join(cells) + " |")

    # 组合
    table = [
        "| " + " | ".join(header_cells) + " |",
        "| " + " | ".join(separators) + " |",
    ] + data_lines

    result = "\n".join(table)

    # 单列行追加在表格后面
    if single_col_rows:
        notes = "\n".join(r[0]["text"] for r in single_col_rows)
        result += "\n\n" + notes

    return result


def _format_as_text(rows: list[list[dict]]) -> str:
    """普通文字排版——同行文字用空格连接，不同行换行。"""
    lines = []
    for row in rows:
        # 同行文字用空格连接
        parts = []
        for i, cell in enumerate(row):
            parts.append(cell["text"])
            # 如果下一个文字的 x 起始远于当前文字的 x 结尾，
            # 加空格表示间隔（ponytail: 不用精确间距，空格够用）
        lines.append(" ".join(parts))
    return "\n".join(lines)


# ── 合并输出 ──────────────────────────────────────────────────

def _merge_output(v6: dict, vl: dict | None, output_format: str) -> str:
    """合并输出 — 还原排版，只输出识别内容。"""
    # VL 优先：VL 输出的 markdown 已经还原了排版结构
    if vl and vl.get("markdown"):
        vl_md = vl["markdown"].strip()
        if output_format == "json":
            return json.dumps({"vl_markdown": vl_md}, ensure_ascii=False, indent=2)
        return vl_md

    # v6 only：用坐标还原排版
    layout_text = _restore_layout(v6["texts"])

    if output_format == "json":
        # JSON 输出保留结构化信息
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
    force_vl: bool = False,
    on_progress=None,
) -> str:
    """统一 OCR 流水线入口。

    v6 先跑，检测到复杂区域自动唤醒 VL，最后合并输出。
    """
    if not Path(image_path).exists():
        return f"文件不存在: {image_path}"

    def progress(msg):
        if on_progress:
            on_progress(msg)

    # 阶段 1: v6 快速扫描
    progress("阶段 1/3: PP-OCRv6 快速扫描文字区域...")
    v6 = _parse_v6(image_path)

    # 阶段 2: 判断是否需要 VL
    complex_count = sum(1 for t in v6["texts"] if t["complexity"] != "text")
    needs_vl = force_vl or complex_count > 0

    vl = None
    if needs_vl:
        reason = "强制启用" if force_vl else f"检测到 {complex_count} 个复杂区域"
        progress(f"阶段 2/3: {reason}，启动 PaddleOCR-VL 深度解析...")
        try:
            vl = _parse_vl(image_path)
        except Exception as e:
            progress(f"阶段 2/3: VL 解析失败 ({e})，跳过，仅使用 v6 结果")
    else:
        progress("阶段 2/3: 未检测到复杂内容，跳过 VL")

    # 阶段 3: 合并输出
    progress("阶段 3/3: 合并结果，还原排版...")
    result = _merge_output(v6, vl, output_format)

    progress("完成")
    return result


# ── PDF → MD 批量转换 ─────────────────────────────────────────

def pdf_to_md(
    pdf_path: str,
    output_dir: str | None = None,
    on_progress=None,
    on_page_done=None,
) -> str:
    """PDF 书籍 → Markdown 文件。逐页 OCR → 拼接保存。

    返回输出 .md 文件路径。
    """
    import fitz  # pymupdf

    pdf_path = str(Path(pdf_path).resolve())
    if not Path(pdf_path).exists():
        raise FileNotFoundError(f"PDF 文件不存在: {pdf_path}")

    doc = fitz.open(pdf_path)
    total_pages = len(doc)

    def progress(msg):
        if on_progress:
            on_progress(msg)

    # 输出路径：同目录同名 .md
    if output_dir is None:
        output_dir = str(Path(pdf_path).parent)
    output_path = str(Path(output_dir) / (Path(pdf_path).stem + ".md"))

    progress(f"PDF 共 {total_pages} 页，开始逐页转换...")

    all_md: list[str] = []
    # 书名作为标题
    all_md.append(f"# {Path(pdf_path).stem}")
    all_md.append("")

    for page_num in range(total_pages):
        page = doc.load_page(page_num)
        # 渲染为图片（300 DPI 保证 OCR 精度）
        pix = page.get_pixmap(dpi=300)
        img_bytes = pix.tobytes("png")

        # 保存临时图片
        temp_img = str(Path(output_dir) / f"_page_{page_num + 1:04d}.png")
        Path(temp_img).write_bytes(img_bytes)

        try:
            progress(f"[{page_num + 1}/{total_pages}] 识别中...")
            page_result = smart_parse(temp_img, output_format="markdown")
        finally:
            # 清理临时图片
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

    # 写入 .md 文件
    md_content = "\n".join(all_md)
    Path(output_path).write_text(md_content, encoding="utf-8")

    progress(f"完成！已保存 → {output_path}")
    return output_path
