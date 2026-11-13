"""Optional, strictly local OCR of character-sized vector paths in PDFs.

This is a review aid for outlined digits that normal PDF text extraction misses.
OCR output is ALWAYS unreviewed. It must never silently replace native text or a
visually checked transcription. The detector is heuristic and not full-page OCR.
"""
from __future__ import annotations

import csv
import io
from pathlib import Path
import re
import shutil
import subprocess


def _candidate_regions(page):
    """Group small filled non-rectangle paths by nearby original geometry."""
    shapes = []
    for drawing in page.get_drawings():
        rect = drawing.get('rect')
        items = drawing.get('items') or []
        if rect is None or drawing.get('type') not in {'f', 'fs'}:
            continue
        if not (0.25 <= rect.width <= 22 and 1.2 <= rect.height <= 26):
            continue
        if not items or all(item[0] == 're' for item in items):
            continue
        shapes.append({'bbox': list(rect), 'curved': any(item[0] == 'c' for item in items)})

    # Connected components retain commas and straight glyphs beside curved digits.
    unseen = set(range(len(shapes)))
    groups = []
    while unseen:
        todo = [min(unseen)]
        unseen.remove(todo[0])
        group = []
        while todo:
            index = todo.pop()
            group.append(shapes[index])
            a = shapes[index]['bbox']
            neighbours = []
            for other in unseen:
                b = shapes[other]['bbox']
                horizontal_gap = max(b[0]-a[2], a[0]-b[2], 0)
                vertical_gap = max(b[1]-a[3], a[1]-b[3], 0)
                vertical_overlap = min(a[3],b[3])-max(a[1],b[1])
                if horizontal_gap <= 4 and vertical_gap <= 1.5 and vertical_overlap >= -1.5:
                    neighbours.append(other)
            for other in neighbours:
                unseen.remove(other)
                todo.append(other)
        # A curve is an indicator rather than proof of a character outline.
        if any(s['curved'] for s in group):
            groups.append({'bbox': [min(s['bbox'][0] for s in group),
                                    min(s['bbox'][1] for s in group),
                                    max(s['bbox'][2] for s in group),
                                    max(s['bbox'][3] for s in group)],
                           'path_count':len(group)})
    return sorted(groups, key=lambda r: (round(r['bbox'][1],1), r['bbox'][0]))


def _native_context(page, bbox):
    """Native nearby line text is context, not an inferred value/label join."""
    y0,y1 = bbox[1],bbox[3]
    lines=[]
    for block in page.get_text('dict').get('blocks',[]):
        for line in block.get('lines',[]):
            box=line.get('bbox')
            if box and min(y1,box[3])-max(y0,box[1]) > 0:
                text=''.join(span.get('text','') for span in line.get('spans',[])).strip()
                if text:lines.append(text)
    return ' | '.join(lines)


def _ocr_region(page, bbox, executable, scale, timeout):
    import fitz
    # Narrow x padding avoids cutting into an adjacent native Korean glyph.
    crop=fitz.Rect(bbox[0]-.75,bbox[1]-2.5,bbox[2]+.75,bbox[3]+2.5) & page.rect
    pix=page.get_pixmap(matrix=fitz.Matrix(scale,scale),clip=crop,alpha=False)
    result=subprocess.run([
        executable,'stdin','stdout','-l','eng','--psm','7',
        '-c','tessedit_char_whitelist=0123456789,.-%+/():~','tsv',
    ],input=pix.tobytes('png'),stdout=subprocess.PIPE,stderr=subprocess.PIPE,
      timeout=timeout,check=False)
    if result.returncode:
        raise RuntimeError('Local tesseract failed: '+result.stderr.decode('utf-8','replace')[:300])
    words=[]
    for row in csv.DictReader(io.StringIO(result.stdout.decode('utf-8','replace')),delimiter='\t'):
        text=(row.get('text') or '').strip()
        if not text:continue
        try:confidence=float(row.get('conf','-1'))
        except ValueError:confidence=-1.
        words.append((text,confidence))
    text=' '.join(word[0] for word in words)
    valid=[word[1] for word in words if word[1]>=0]
    confidence=sum(valid)/len(valid) if valid else None
    return {'text':text,'ocr_confidence':confidence,
            'numeric_candidates':re.findall(r'\d[\d,]*(?:\.\d+)?',text),
            'crop_bbox':list(crop)}


def inspect_vector_regions(path, pages=None, max_regions=50, run_ocr=True,
                           scale=4.0, timeout=15):
    """Inspect selected PHYSICAL pages (one-based), optionally using local OCR.

    Dependencies are PyMuPDF and, only when run_ocr=True, the local ``tesseract``
    binary with eng traineddata. No network or OCR service is used. ``pages=None``
    scans the complete file geometrically but performs at most max_regions OCRs.
    Returned numeric tokens are candidate strings, never verified financial data.
    """
    import fitz
    path=Path(path)
    if max_regions<0 or scale<=0 or timeout<=0:
        raise ValueError('max_regions must be non-negative; scale and timeout positive')
    executable=shutil.which('tesseract') if run_ocr else None
    blocks=[];regions=[];warnings=[];attempted=0;failures=0
    if run_ocr and not executable:
        warnings.append({'code':'LOCAL_OCR_UNAVAILABLE','message':'Local tesseract executable not found; candidate regions only.'})
    with fitz.open(path) as pdf:
        selected=list(range(1,len(pdf)+1)) if pages is None else sorted(set(pages))
        if any(not isinstance(p,int) or p<1 or p>len(pdf) for p in selected):
            raise ValueError('pages must contain physical page numbers inside the file')
        for number in selected:
            page=pdf[number-1]
            for index,region in enumerate(_candidate_regions(page),1):
                locator=f'PDF p.{number} / vector region {index}'
                region.update(page=number,locator=locator,
                              context_text=_native_context(page,region['bbox']),
                              review_status='unreviewed',detection_method='small_filled_vector_paths')
                regions.append(region)
                if not executable or attempted>=max_regions:continue
                attempted+=1
                try:
                    extracted=_ocr_region(page,region['bbox'],executable,scale,timeout)
                except (OSError,RuntimeError,subprocess.TimeoutExpired) as exc:
                    failures+=1
                    warnings.append({'code':'LOCAL_OCR_REGION_FAILED','message':str(exc),'locator':locator})
                    continue
                region.update(extracted)
                if extracted['numeric_candidates']:
                    blocks.append({'block_id':f'pdf-p{number:04d}-vocr{index:04d}',
                                   'kind':'ocr_region','locator':locator,'page':number,
                                   'bbox':region['bbox'],'text':extracted['text'],
                                   'numeric_candidates':extracted['numeric_candidates'],
                                   'ocr_confidence':extracted['ocr_confidence'],
                                   'context_text':region['context_text'],
                                   'source_method':'local_tesseract_eng_numeric_ocr',
                                   'review_status':'unreviewed'})
        if regions:
            warnings.append({'code':'VECTOR_OUTLINED_TEXT_SUSPECTED',
                             'message':'Small filled vector paths may encode text missing from native extraction; regions require visual review.'})
        if run_ocr and executable and len(regions)>max_regions:
            warnings.append({'code':'LOCAL_OCR_REGION_LIMIT',
                             'message':f'{len(regions)-max_regions} candidate regions not OCRed due to max_regions limit.'})
    return {'parser':'local_vector_numeric_ocr_v1','blocks':blocks,'regions':regions,
            'warnings':warnings,'stats':{'pages_scanned':len(selected),'physical_pages':selected,
                'candidate_regions':len(regions),'ocr_attempted':attempted,
                'numeric_blocks':len(blocks),'ocr_failures':failures,
                'unreviewed_blocks':len(blocks),'local_only':True},
            'limitations':['Heuristic detection does not find all outlined or image text.',
                           'English digit OCR does not recover Korean labels or semantics.',
                           'OCR confidence is not factual verification; all output remains unreviewed.',
                           'Candidate strings never replace native text or visually checked transcriptions.']}
