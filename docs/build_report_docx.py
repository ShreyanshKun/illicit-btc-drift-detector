"""Convert reports/REPORT.md (a simple Markdown subset) into reports/REPORT.docx with figures.

Usage:  python docs/build_report_docx.py
"""
import re
import sys
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "reports" / "REPORT.md"
OUT = ROOT / "reports" / "REPORT.docx"
if len(sys.argv) == 3:                      # optional: python docs/build_report_docx.py in.md out.docx
    SRC, OUT = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()


def add_runs(par, text):
    """Handle **bold**, *italic* and `code` inline."""
    for tok in re.split(r"(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`)", text):
        if not tok:
            continue
        if tok.startswith("**"):
            par.add_run(tok[2:-2]).bold = True
        elif tok.startswith("`"):
            r = par.add_run(tok[1:-1])
            r.font.name = "Consolas"
            r.font.size = Pt(9.5)
        elif tok.startswith("*") and len(tok) > 2:
            par.add_run(tok[1:-1]).italic = True
        else:
            par.add_run(tok)


def add_table(doc, rows):
    header, body = rows[0], rows[2:]
    t = doc.add_table(rows=1, cols=len(header))
    t.style = "Table Grid"      # plain black/grey grid, no theme colours
    for i, h in enumerate(header):
        t.rows[0].cells[i].text = ""
        add_runs(t.rows[0].cells[i].paragraphs[0], h)
        for r in t.rows[0].cells[i].paragraphs[0].runs:
            r.bold = True
            r.font.size = Pt(9)
            r.font.color.rgb = RGBColor(0, 0, 0)
        shd = OxmlElement("w:shd")
        shd.set(qn("w:val"), "clear"); shd.set(qn("w:color"), "auto"); shd.set(qn("w:fill"), "EDEDED")
        t.rows[0].cells[i]._tc.get_or_add_tcPr().append(shd)
    for row in body:
        cells = t.add_row().cells
        for i, c in enumerate(row):
            cells[i].text = ""
            add_runs(cells[i].paragraphs[0], c)
            for r in cells[i].paragraphs[0].runs:
                r.font.size = Pt(9)
    doc.add_paragraph()


def neutralise_styles(doc):
    """Strip the blue theme colours from headings, title, quotes and any border on them."""
    for name in ("Title", "Heading 1", "Heading 2", "Heading 3", "Intense Quote", "Quote"):
        try:
            st = doc.styles[name]
        except KeyError:
            continue
        st.font.color.rgb = RGBColor(0, 0, 0)
        st.font.name = "Calibri"
        rpr = st.element.get_or_add_rPr()
        rf = rpr.find(qn("w:rFonts"))
        if rf is not None:
            for a in ("w:asciiTheme", "w:hAnsiTheme", "w:eastAsiaTheme", "w:cstheme"):
                rf.attrib.pop(qn(a), None)
        ppr = st.element.find(qn("w:pPr"))
        if ppr is not None:
            for tag in ("w:pBdr",):
                el = ppr.find(qn(tag))
                if el is not None:
                    ppr.remove(el)


def main():
    doc = Document()
    neutralise_styles(doc)
    st = doc.styles["Normal"]
    st.font.name = "Calibri"
    st.font.size = Pt(11)
    for s in doc.sections:
        s.left_margin = s.right_margin = Inches(0.9)
        s.top_margin = s.bottom_margin = Inches(0.8)
    lines = SRC.read_text(encoding="utf-8").splitlines()
    i, table = 0, []
    while i < len(lines):
        ln = lines[i].rstrip()
        if ln.startswith("|"):
            table.append([c.strip() for c in ln.strip("|").split("|")])
            if i + 1 >= len(lines) or not lines[i + 1].startswith("|"):
                add_table(doc, table)
                table = []
            i += 1
            continue
        if not ln or ln == "---":
            i += 1
            continue
        m = re.match(r"!\[(.*?)\]\((.*?)\)", ln)
        if m:
            doc.add_picture(str(SRC.parent / m.group(2)), width=Inches(6.0))
            doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
            cap = doc.add_paragraph()
            r = cap.add_run(f"Figure: {m.group(1)}")
            r.italic = True
            r.font.size = Pt(9)
            r.font.color.rgb = RGBColor(0x55, 0x55, 0x55)
            cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        elif ln.startswith("# "):
            doc.add_heading(ln[2:], 0)
        elif ln.startswith("## "):
            doc.add_heading(ln[3:], 1)
        elif ln.startswith("### "):
            doc.add_heading(ln[4:], 2)
        elif ln.startswith("> "):
            p = doc.add_paragraph(style="Intense Quote")
            add_runs(p, ln[2:].lstrip("*").rstrip("*") if ln.startswith("> **") is False else ln[2:])
        elif re.match(r"^\* ", ln) or ln.startswith("- "):
            add_runs(doc.add_paragraph(style="List Bullet"), ln[2:])
        elif re.match(r"^\d+\. ", ln):
            add_runs(doc.add_paragraph(style="List Number"), re.sub(r"^\d+\. ", "", ln))
        else:
            add_runs(doc.add_paragraph(), ln)
        i += 1
    doc.save(OUT)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
