"""Render the separate v3 rationale; manuscript remains authored in LaTeX."""
from pathlib import Path
import re
from html import escape
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, LongTable, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.colors import HexColor, white
from reportlab.lib.enums import TA_LEFT

root = Path(__file__).resolve().parent.parent
src = root/'versions/v3_revision_notes.md'
dest = root/'versions/v3_revision_notes.pdf'
styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name='BodyV3', fontName='Helvetica', fontSize=10, leading=13.4, spaceAfter=7.5))
styles.add(ParagraphStyle(name='TitleV3', fontName='Helvetica-Bold', fontSize=23, leading=27, textColor=HexColor('#123c53'), spaceAfter=12))
styles.add(ParagraphStyle(name='SectionV3', fontName='Helvetica-Bold', fontSize=15, leading=19, textColor=HexColor('#123c53'), spaceBefore=10, spaceAfter=9, keepWithNext=True))
styles.add(ParagraphStyle(name='SubV3', fontName='Helvetica-Bold', fontSize=10.5, leading=14, spaceBefore=8, spaceAfter=5, keepWithNext=True))
styles.add(ParagraphStyle(name='CellV3', fontName='Helvetica', fontSize=8.2, leading=10.2, alignment=TA_LEFT))
styles.add(ParagraphStyle(name='HeadV3', parent=styles['CellV3'], fontName='Helvetica-Bold', textColor=white))

def markup(s):
    s = escape(s)
    s = re.sub(r'\[([^]]+)\]\((https://[^)]+)\)', r'<link href="\2" color="#185d80">\1</link>', s)
    return s

lines = src.read_text(encoding='utf-8').splitlines()
story = []
i = 0
while i < len(lines):
    line = lines[i].strip()
    if not line:
        i += 1; continue
    if line == '<!-- PAGEBREAK -->':
        story.append(PageBreak()); i += 1; continue
    if line.startswith('### I.'):
        story.append(PageBreak())
    if line.startswith('|'):
        rows = []
        while i < len(lines) and lines[i].startswith('|'):
            fields = [c.strip() for c in lines[i].strip('|').split('|')]
            if not all(re.fullmatch(r'[:\- ]+', c) for c in fields):
                st = styles['HeadV3'] if not rows else styles['CellV3']
                rows.append([Paragraph(markup(c), st) for c in fields])
            i += 1
        table = LongTable(rows, colWidths=[110, 38, 175, 193], repeatRows=1, hAlign='LEFT')
        table.setStyle(TableStyle([
            ('BACKGROUND',(0,0),(-1,0),HexColor('#123c53')),
            ('ROWBACKGROUNDS',(0,1),(-1,-1),[white,HexColor('#f0f5f7')]),
            ('VALIGN',(0,0),(-1,-1),'TOP'),
            ('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5),
            ('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),4),
        ]))
        story.extend([table, Spacer(1,9)]); continue
    if line.startswith('# '):
        style='TitleV3'; content=line[2:]
    elif line.startswith('## '):
        style='SectionV3'; content=line[3:]
    elif line.startswith('### '):
        style='SubV3'; content=line[4:]
    else:
        style='BodyV3'; content=line
    story.append(Paragraph(markup(content),styles[style]))
    i += 1

def footer(canvas, doc):
    canvas.setFont('Helvetica',8)
    canvas.setFillColor(HexColor('#5a6871'))
    canvas.drawString(48,25,'CIS 6270 | Manuscript v3 revision rationale | 27 September 2026')
    canvas.drawRightString(564,25,str(doc.page))

doc=SimpleDocTemplate(str(dest),pagesize=(612,792),rightMargin=48,leftMargin=48,topMargin=42,bottomMargin=43,
    title='Why the paper was revised: Manuscript v3',author='CIS 6270 Group 2')
doc.build(story,onFirstPage=footer,onLaterPages=footer)
print(dest)
