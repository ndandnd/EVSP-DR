#!/usr/bin/env python3
"""Vector figure plus native PDF itinerary tables."""
from pathlib import Path
import csv,html
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,PageBreak
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib.pagesizes import landscape,A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from pypdf import PdfReader,PdfWriter
HERE=Path(__file__).resolve().parent
FONT=Path('/Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages/matplotlib/mpl-data/fonts/ttf')
pdfmetrics.registerFont(TTFont('DejaVu',str(FONT/'DejaVuSans.ttf')));pdfmetrics.registerFont(TTFont('DejaVuBold',str(FONT/'DejaVuSans-Bold.ttf')))
styles=getSampleStyleSheet();styles.add(ParagraphStyle(name='Cell',fontName='DejaVu',fontSize=9,leading=11,textColor=colors.HexColor('#253d4e')));styles.add(ParagraphStyle(name='Head',fontName='DejaVuBold',fontSize=15,leading=20));styles.add(ParagraphStyle(name='Note',fontName='DejaVu',fontSize=8.5,leading=12));styles.add(ParagraphStyle(name='TH',fontName='DejaVuBold',fontSize=9,leading=11))
def rows(name):return list(csv.DictReader((HERE/name).open()))
def p(s,sty='Cell'):return Paragraph(html.escape(str(s)).replace('\n','<br/>'),styles[sty])
def table(head,data,widths):
 t=Table([[p(x,'TH') for x in head]]+[[p(x) for x in row] for row in data],colWidths=widths,repeatRows=1,hAlign='LEFT')
 t.setStyle(TableStyle([('VALIGN',(0,0),(-1,-1),'TOP'),('BACKGROUND',(0,0),(-1,0),colors.HexColor('#edf3f6')),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f7f9fb')]),('GRID',(0,0),(-1,-1),.3,colors.HexColor('#d4dfe5')),('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5),('TOPPADDING',(0,0),(-1,-1),3),('BOTTOMPADDING',(0,0),(-1,-1),3)]));return t
W,H=1080,720;content=W-64
story=[p('Duty 13309 — chronological leg key','Head'),Spacer(1,7),p('L = passenger-leg order; M = inter-area empty-move order. Trip numbers are stable prepared-input labels, not GIRO-supplied journey numbers. The graph groups platforms through documented reference areas; the codes below retain actual recorded endpoints.','Note'),Spacer(1,10)]
legs=rows('diagram_leg_key.csv');story.append(table(['Step','Graph','Trip','Recorded endpoints','Departure','Arrival','Minutes'],[[x['step'],x['diagram_label'],x['prepared_trip_id'],x['from_code']+' → '+x['to_code'],x['departure'],x['arrival'],x['duration_minutes']] for x in legs],[36,42,45,content-36-42-45-73-73-58,73,73,58]))
story+=[PageBreak(),p('Duty 13309 — numbered area visits','Head'),Spacer(1,7),p('Visit bounds include local platform moves and dwells. The full event ledger follows. V numbering counts repeated arrivals to an area; it is separate from L/M/C numbering.','Note'),Spacer(1,10)]
visits=rows('diagram_visit_key.csv');story.append(table(['Visit','Area / recorded platform codes','Arrival','Departure','Charges / visit detail'],[[x['visit'],x['area']+' · '+x['raw_codes'],x['arrival'],x['departure'],x['activities']] for x in visits],[40,190,65,75,content-370]))
story+=[PageBreak(),p('Duty 13309 — complete event ledger','Head'),Spacer(1,7),p('Includes preparation, recorded platform movements, waits and gaps between platform codes without an explicit movement record. Such gaps are not assigned an invented travel time or energy. Source rows refer to Par_VehicleDetails.xlsx, Data worksheet.','Note'),Spacer(1,10)]
events=rows('diagram_event_key.csv');story.append(table(['Event','Graph','Kind','Recorded endpoints','Start','End','Trip','Recharge kWh','Source row'],[[x['event'],x['diagram_label'],x['kind'].replace('_',' '),x['from_code']+' → '+x['to_code'],x['start'],x['end'],x['prepared_trip_id'],f"{float(x['recorded_recharge_kwh']):.3f}" if x['recorded_recharge_kwh'] else '',x['source_pointer'].replace('Data!','')] for x in events],[45,45,100,130,49,49,39,72,content-629]))
story+=[Spacer(1,12),p('Scope: recorded GIRO original only; no optimized counterpart or new feasibility certificate. The four recharge amounts were independently checked against raw workbook cells and this duty’s 239.01 kWh18E2 capacity. The primary schematic displaces PARX for readability; coordinates.csv retains actual documented proxies and source URLs. Full extraction, source hashes and validation are in schedules.json and validation.json.','Note')]
def footer(canvas,doc):
 canvas.setFont('DejaVu',7);canvas.setFillColor(colors.HexColor('#71808c'));canvas.drawString(32,18,'GIRO13309 · recorded schedule · 21September2026 · source tables in complex_route_graphs');canvas.drawRightString(W-32,18,str(doc.page))
tables=HERE/'duty_13309_itinerary.pdf';SimpleDocTemplate(str(tables),pagesize=(W,H),leftMargin=32,rightMargin=32,topMargin=26,bottomMargin=30).build(story,onFirstPage=footer,onLaterPages=footer)
w=PdfWriter();w.append(str(HERE/'duty_13309_graph.pdf'),outline_item='Spatial graph');w.append(str(tables),outline_item='Recorded itinerary');w.write(str(HERE/'duty_13309_daybook.pdf'))
print('Daybook pages:',len(PdfReader(str(HERE/'duty_13309_daybook.pdf')).pages))
