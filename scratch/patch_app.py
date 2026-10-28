from pathlib import Path
import re

p = Path(r"C:\Users\gnana\OneDrive\Desktop\5thyear\MINI PROJECT\Project\final\Agentic-RAG\app.py")
text = p.read_text(encoding="utf-8")

# 1. Fix CSS duplicate Choose PDF Files button
old_css_btn = """.st-key-kb_upload_card [data-testid="stFileUploaderDropzone"] button:not([aria-label]),
    .kb-card [data-testid="stFileUploaderDropzone"] button:not([aria-label]) {{
        background: #102442 !important;"""

new_css_btn = """.st-key-kb_upload_card [data-testid="stFileUploaderDropzone"] button:first-of-type,
    .kb-card [data-testid="stFileUploaderDropzone"] button:first-of-type {{
        background: #102442 !important;"""

text = text.replace(old_css_btn, new_css_btn)

old_css_after = """.st-key-kb_upload_card [data-testid="stFileUploaderDropzone"] button:not([aria-label])::after,
    .kb-card [data-testid="stFileUploaderDropzone"] button:not([aria-label])::after {{
        content: "Choose PDF Files" !important;"""

new_css_after = """.st-key-kb_upload_card [data-testid="stFileUploaderDropzone"] button:first-of-type::after,
    .kb-card [data-testid="stFileUploaderDropzone"] button:first-of-type::after {{
        content: "Choose PDF Files" !important;"""

text = text.replace(old_css_after, new_css_after)

# Hide any secondary/duplicate buttons inside dropzone
hide_dup_btn = """
    .st-key-kb_upload_card [data-testid="stFileUploaderDropzone"] button:not(:first-of-type),
    .kb-card [data-testid="stFileUploaderDropzone"] button:not(:first-of-type) {{
        display: none !important;
    }}
"""
if hide_dup_btn.strip() not in text:
    text = text.replace(new_css_after, hide_dup_btn + "\n    " + new_css_after)

# 2. Replace use_container_width=True with width="stretch" everywhere
text = text.replace("use_container_width=True", 'width="stretch"')

p.write_text(text, encoding="utf-8")
print("Part 1: CSS and width='stretch' patched successfully!")
