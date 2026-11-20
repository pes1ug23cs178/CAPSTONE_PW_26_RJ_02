"""Render the exported PDF to PNG pages and contact sheets for visual checking (scratch output in build/pages/)."""
import sys
from pathlib import Path

import pypdfium2 as pdfium
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
pdf = HERE.parent / "output" / "PW26_RJ_02_Phase3_Report.pdf"
out = HERE / "pages"
out.mkdir(exist_ok=True)
doc = pdfium.PdfDocument(str(pdf))
pages = []
for i in range(len(doc)):
    img = doc[i].render(scale=1.4).to_pil()
    img.save(out / f"p{i + 1:03d}.png")
    pages.append(img)
per, cols = 12, 6
for s in range(0, len(pages), per):
    chunk = pages[s:s + per]
    w, h = 300, 424
    sheet = Image.new("RGB", (cols * w, ((len(chunk) + cols - 1) // cols) * (h + 20)), "white")
    for k, im in enumerate(chunk):
        t = im.resize((w - 8, h - 8))
        x, y = (k % cols) * w, (k // cols) * (h + 20)
        sheet.paste(t, (x + 4, y + 18))
        ImageDraw.Draw(sheet).text((x + 6, y + 2), f"PDF page {s + k + 1}", fill="red")
    sheet.save(out / f"sheet_{s // per + 1:02d}.png")
print(len(pages), "pages rendered")
