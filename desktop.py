"""smart-doc-parser 桌面应用 — 拖拽上传，一键识别，v6+VL 自动组合。

Launch: python desktop.py
"""
import os
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from pipeline import smart_parse, pdf_to_md


class SmartDocParserApp:
    """桌面 OCR 工具主窗口。"""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Smart Doc Parser — PP-OCRv6 + PaddleOCR-VL")
        self.root.geometry("900x700")
        self.root.minsize(700, 500)
        self.root.configure(bg="#1e1e2e")

        self.output_md_path: str | None = None  # PDF 转 MD 的输出路径

        # 样式
        style = ttk.Style()
        style.theme_use("clam")
        self._setup_styles(style)

        # 布局
        self._build_ui()

        # 支持拖拽
        self._setup_drag_drop()

    # ── 样式 ──────────────────────────────────────────────────

    def _setup_styles(self, style: ttk.Style):
        bg = "#1e1e2e"
        fg = "#cdd6f4"
        accent = "#89b4fa"
        btn_bg = "#313244"

        style.configure("TFrame", background=bg)
        style.configure("TLabel", background=bg, foreground=fg, font=("Microsoft YaHei UI", 11))
        style.configure("Title.TLabel", font=("Microsoft YaHei UI", 20, "bold"),
                        foreground=accent, background=bg)
        style.configure("Status.TLabel", font=("Microsoft YaHei UI", 9),
                        foreground="#a6adc8", background=bg)
        style.configure("Drop.TLabel", font=("Microsoft YaHei UI", 13),
                        foreground="#585b70", background="#313244")
        style.configure("Primary.TButton", font=("Microsoft YaHei UI", 12, "bold"),
                        background=accent, foreground="#1e1e2e")
        style.map("Primary.TButton",
                  background=[("active", "#74c7ec")])
        style.configure("TButton", font=("Microsoft YaHei UI", 10))
        style.configure("TProgressbar", troughcolor="#313244",
                        background="#a6e3a1", thickness=8)

    # ── UI 构建 ───────────────────────────────────────────────

    def _build_ui(self):
        # 主容器
        main = ttk.Frame(self.root, padding=20)
        main.pack(fill="both", expand=True)

        # 标题
        ttk.Label(main, text="Smart Doc Parser", style="Title.TLabel").pack(anchor="w")
        ttk.Label(main, text="PP-OCRv6 快速扫描 + PaddleOCR-VL 深度解析 · 自动组合",
                  style="Status.TLabel").pack(anchor="w", pady=(2, 20))

        # 文件区域
        file_frame = ttk.Frame(main)
        file_frame.pack(fill="x", pady=(0, 12))

        self.file_var = tk.StringVar(value="未选择文件")
        file_entry = ttk.Entry(file_frame, textvariable=self.file_var,
                               font=("Consolas", 10), state="readonly")
        file_entry.pack(side="left", fill="x", expand=True, padx=(0, 8))

        ttk.Button(file_frame, text="选择文件", command=self._pick_file).pack(side="right")

        # 控制按钮行
        ctrl_frame = ttk.Frame(main)
        ctrl_frame.pack(fill="x", pady=(0, 12))

        self.run_btn = ttk.Button(ctrl_frame, text="开始识别",
                                   command=self._start_ocr, style="Primary.TButton")
        self.run_btn.pack(side="left", padx=(0, 10))

        self.force_vl_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(ctrl_frame, text="强制启用 VL 深度解析",
                        variable=self.force_vl_var).pack(side="left")

        # 进度条
        self.progress = ttk.Progressbar(main, mode="indeterminate")
        self.progress.pack(fill="x", pady=(0, 4))

        self.status_var = tk.StringVar(value="就绪")
        ttk.Label(main, textvariable=self.status_var, style="Status.TLabel").pack(
            anchor="w", pady=(0, 8))

        # 结果区域
        result_frame = ttk.Frame(main)
        result_frame.pack(fill="both", expand=True)

        self.result_text = tk.Text(
            result_frame,
            font=("Consolas", 10),
            bg="#313244", fg="#cdd6f4",
            insertbackground="#cdd6f4",
            relief="flat", borderwidth=0,
            wrap="word", state="disabled",
        )
        scrollbar = ttk.Scrollbar(result_frame, command=self.result_text.yview)
        self.result_text.configure(yscrollcommand=scrollbar.set)

        self.result_text.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Tag 样式（给结果加颜色标记）
        self.result_text.tag_configure("heading1", font=("Microsoft YaHei UI", 14, "bold"),
                                       foreground="#89b4fa")
        self.result_text.tag_configure("heading2", font=("Microsoft YaHei UI", 12, "bold"),
                                       foreground="#74c7ec")
        self.result_text.tag_configure("meta", foreground="#a6adc8")
        self.result_text.tag_configure("flag", foreground="#f9e2af")

        # 底部操作栏
        bottom = ttk.Frame(main)
        bottom.pack(fill="x", pady=(8, 0))

        ttk.Button(bottom, text="复制结果", command=self._copy_result).pack(side="left", padx=(0, 8))
        self.open_file_btn = ttk.Button(bottom, text="打开文件", command=self._open_output, state="disabled")
        self.open_file_btn.pack(side="left", padx=(0, 8))
        ttk.Button(bottom, text="清空", command=self._clear).pack(side="left")

    # ── 拖拽支持 ──────────────────────────────────────────────

    def _setup_drag_drop(self):
        """Windows 拖拽支持（通过 tkinterdnd2 或手动注册）。"""
        # ponytail: tkinterdnd2 需要额外安装。
        #   fallback: 用户用"选择文件"按钮即可，拖拽是锦上添花。
        #   upgrade: 安装 tkinterdnd2 后解锁拖拽。
        try:
            from tkinterdnd2 import DND_FILES, TkinterDnD
            # 如果已经用了 TkinterDnD.Tk，注册拖拽
            if hasattr(self.root, "drop_target_register"):
                self.root.drop_target_register(DND_FILES)
                self.root.dnd_bind("<<Drop>>", self._on_drop)
        except ImportError:
            pass  # 没有 tkinterdnd2，只用按钮选择文件

    def _on_drop(self, event):
        filepath = event.data.strip("{}")
        self._set_file(filepath)

    # ── 文件操作 ──────────────────────────────────────────────

    def _pick_file(self):
        path = filedialog.askopenfilename(
            title="选择图片或 PDF",
            filetypes=[
                ("图片和 PDF", "*.png *.jpg *.jpeg *.bmp *.tiff *.pdf"),
                ("所有文件", "*.*"),
            ],
        )
        if path:
            self._set_file(path)

    def _set_file(self, path: str):
        self.file_var.set(path)
        self.file_path = path

    # ── OCR 执行 ──────────────────────────────────────────────

    def _start_ocr(self):
        if not hasattr(self, "file_path") or not self.file_path:
            messagebox.showwarning("提示", "请先选择文件")
            return

        self.run_btn.configure(state="disabled")
        self.open_file_btn.configure(state="disabled")
        self.output_md_path = None
        self.progress.start()
        self._set_result("", clear_only=True)

        ext = os.path.splitext(self.file_path)[1].lower()
        is_pdf = ext == ".pdf"

        thread = threading.Thread(
            target=self._run_pdf if is_pdf else self._run_pipeline, daemon=True
        )
        thread.start()

    def _run_pipeline(self):
        try:
            def on_progress(msg):
                self.root.after(0, lambda: self.status_var.set(msg))

            result = smart_parse(
                self.file_path,
                output_format="markdown",
                force_vl=self.force_vl_var.get(),
                on_progress=on_progress,
            )
            self.root.after(0, lambda: self._set_result(result))
        except Exception as e:
            self.root.after(0, lambda: self._set_result(f"识别出错:\n{e}"))
        finally:
            self.root.after(0, self._done)

    def _done(self):
        self.progress.stop()
        self.run_btn.configure(state="normal")
        if self.output_md_path:
            self.open_file_btn.configure(state="normal")
        self.status_var.set("就绪")

    # ── PDF 批量转换 ──────────────────────────────────────────

    def _run_pdf(self):
        try:
            self.root.after(0, lambda: self._set_result(f"正在解析 PDF...\n文件: {self.file_path}\n\n"))

            def on_progress(msg):
                self.root.after(0, lambda: self.status_var.set(msg))

            def on_page_done(page, total, result):
                preview = result[:80].replace("\n", " ") + ("..." if len(result) > 80 else "")
                self.root.after(0, lambda: self.status_var.set(
                    f"[{page}/{total}] {preview}"
                ))
                # 实时追加到文本框
                self.root.after(0, lambda p=page, r=result: self._append_page(p, r))

            output_path = pdf_to_md(
                self.file_path,
                on_progress=on_progress,
                on_page_done=on_page_done,
            )
            self.output_md_path = output_path
            self.root.after(0, lambda: self._set_result(
                f"转换完成！\n输出文件: {output_path}\n\n"
                f"内容已逐页显示在下方。"
            ))
        except Exception as e:
            self.root.after(0, lambda: self._set_result(f"PDF 转换出错:\n{e}"))
        finally:
            self.root.after(0, self._done)

    def _append_page(self, page: int, text: str):
        self.result_text.configure(state="normal")
        # 如果已经有一些内容，追加分隔
        current = self.result_text.get("1.0", "end-1c")
        if current.strip():
            self.result_text.insert("end", f"\n\n## 第 {page} 页\n\n")
        else:
            self.result_text.insert("end", f"# PDF 逐页结果\n\n## 第 {page} 页\n\n")
        self.result_text.insert("end", text)
        self.result_text.configure(state="disabled")
        # 滚动到末尾
        self.result_text.see("end")

    def _open_output(self):
        if self.output_md_path and os.path.exists(self.output_md_path):
            os.startfile(self.output_md_path)
            self.status_var.set(f"已打开: {os.path.basename(self.output_md_path)}")

    # ── 结果显示 ──────────────────────────────────────────────

    def _set_result(self, text: str, clear_only: bool = False):
        self.result_text.configure(state="normal")
        self.result_text.delete("1.0", "end")
        if not clear_only:
            self.result_text.insert("1.0", text)
        self.result_text.configure(state="disabled")

    def _copy_result(self):
        text = self.result_text.get("1.0", "end-1c")
        if text:
            self.root.clipboard_clear()
            self.root.clipboard_append(text)
            self.status_var.set("已复制到剪贴板")
            self.root.after(2000, lambda: self.status_var.set("就绪"))

    def _clear(self):
        self.file_var.set("未选择文件")
        if hasattr(self, "file_path"):
            del self.file_path
        self._set_result("", clear_only=True)
        self.status_var.set("就绪")


def main():
    root = tk.Tk()
    app = SmartDocParserApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
