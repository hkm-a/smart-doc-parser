"""smart-doc-parser: v6 for speed, VL for depth. One script, two modes.

Usage:
  python parse.py input.png              # fast: PP-OCRv6
  python parse.py input.png --deep       # deep: PaddleOCR-VL
  python parse.py input.pdf --deep       # deep: VL handles PDF
  python parse.py input.png --format json  # output as JSON instead of MD
"""
import argparse
import json
import sys
from pathlib import Path


def parse_v6(image_path: str) -> dict:
    """PP-OCRv6: fast text detection + recognition."""
    from paddleocr import PaddleOCR
    ocr = PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )
    result = ocr.predict(image_path)
    # ponytail: flatten PaddleOCR result into simple dict.
    #   ceiling: PaddleOCR may change result format across versions.
    #   upgrade: wrap in adapter if format breaks.
    texts = []
    for res in result:
        for item in res.get("rec_text", []):
            texts.append({
                "text": item.get("text", ""),
                "confidence": item.get("score", 0.0),
                "bbox": item.get("det_box", []),
            })
    return {"source": image_path, "mode": "v6", "texts": texts}


def parse_vl(image_path: str) -> dict:
    """PaddleOCR-VL: deep document parsing (tables, formulas, charts)."""
    from paddleocr import PaddleOCR
    ocr = PaddleOCR(
        pipeline="doc_parse",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )
    result = ocr.predict(image_path)
    # ponytail: extract markdown content from VL result.
    #   ceiling: VL result structure may vary by doc_parse version.
    #   upgrade: add structured extraction if markdown isn't enough.
    markdown_content = ""
    for res in result:
        markdown_content += res.get("markdown", res.get("content", str(res)))
    return {"source": image_path, "mode": "vl", "markdown": markdown_content}


def to_markdown(parsed: dict) -> str:
    if parsed["mode"] == "vl":
        return parsed["markdown"]
    lines = []
    for t in parsed["texts"]:
        lines.append(t["text"])
    return "\n".join(lines)


def to_json(parsed: dict) -> str:
    return json.dumps(parsed, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description="smart-doc-parser: v6 fast, VL deep")
    parser.add_argument("input", help="Image or PDF path")
    parser.add_argument("--deep", action="store_true", help="Use PaddleOCR-VL for complex docs")
    parser.add_argument("--format", choices=["markdown", "json"], default="markdown",
                        help="Output format (default: markdown)")
    parser.add_argument("--output", help="Write to file instead of stdout")
    args = parser.parse_args()

    if not Path(args.input).exists():
        sys.exit(f"File not found: {args.input}")

    parsed = parse_vl(args.input) if args.deep else parse_v6(args.input)

    output = to_json(parsed) if args.format == "json" else to_markdown(parsed)

    if args.output:
        Path(args.output).write_text(output, encoding="utf-8")
        print(f"Written to {args.output}")
    else:
        print(output)


if __name__ == "__main__":
    main()
