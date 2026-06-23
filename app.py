"""smart-doc-parser WebUI — 拖拽上传，点击识别，直接看结果。"""
import gradio as gr
from parse import parse_v6, parse_vl, to_markdown, to_json


def ocr_image(image_path: str, mode: str, fmt: str):
    """Run OCR and return formatted output + status."""
    if image_path is None:
        return "请先上传图片或 PDF", ""
    try:
        parsed = parse_vl(image_path) if mode == "vl" else parse_v6(image_path)
        output = to_json(parsed) if fmt == "json" else to_markdown(parsed)
        count = len(parsed.get("texts", [])) if mode == "v6" else "—"
        status = f"识别完成 | 模式: {mode.upper()} | 文字段: {count} | 源文件: {image_path}"
        return output, status
    except Exception as e:
        return f"识别出错: {e}", ""


with gr.Blocks(title="Smart Doc Parser") as demo:
    gr.Markdown("# Smart Doc Parser\n拖拽图片或 PDF，选择模式，点击识别。")

    with gr.Row():
        with gr.Column(scale=2):
            file_input = gr.File(label="上传文件", file_types=["image", ".pdf"])
            with gr.Row():
                mode_choice = gr.Radio(
                    choices=[("快速模式 (PP-OCRv6)", "v6"), ("深度模式 (PaddleOCR-VL)", "vl")],
                    value="v6", label="识别模式"
                )
                fmt_choice = gr.Radio(
                    choices=[("Markdown", "markdown"), ("JSON", "json")],
                    value="markdown", label="输出格式"
                )
            run_btn = gr.Button("开始识别", variant="primary", size="lg")

        with gr.Column(scale=3):
            status_text = gr.Textbox(label="状态", interactive=False)
            result_text = gr.Textbox(
                label="识别结果", lines=20, max_lines=40,
                interactive=False
            )

    run_btn.click(
        fn=ocr_image,
        inputs=[file_input, mode_choice, fmt_choice],
        outputs=[result_text, status_text],
    )


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)
