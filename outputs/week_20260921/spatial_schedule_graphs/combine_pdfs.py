#!/usr/bin/env python3
"""Combine the five vector spatial figures and five detailed itineraries."""
from pathlib import Path
import json
from pypdf import PdfReader, PdfWriter

HERE=Path(__file__).resolve().parent
data=json.loads((HERE/'schedules.json').read_text())
writer=PdfWriter()
for pair in sorted(data['pairs'],key=lambda p:p['visual_priority']):
    parent=writer.add_outline_item('Original duty '+pair['original_duty']+' and paired schedules',len(writer.pages))
    for kind,label in [('graph','Fixed spatial graph'),('itinerary','Complete chronological itinerary')]:
        page=len(writer.pages)
        writer.append(HERE/(pair['pair_id']+'_'+kind+'.pdf'),import_outline=False)
        writer.add_outline_item(label,page,parent=parent)
writer.add_metadata({'/Title':'Five matched bus-day comparisons','/Subject':'Spatial graphs and complete visit itineraries; saved-sequence charging comparison','/Author':'EVSP–DR research'})
with (HERE/'all_comparisons.pdf').open('wb') as f:
    writer.write(f)
assert len(PdfReader(HERE/'all_comparisons.pdf').pages)==10
print('Combined 10 vector pages with duty and view bookmarks')
