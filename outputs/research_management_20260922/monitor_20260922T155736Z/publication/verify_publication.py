#!/usr/bin/env python3
"""Read-only export and source audit. No cluster, UI, or live document mutations.

Run after root supplies final current_after.md/PDF and slides_after.pptx/PDF.
Visual review is recorded separately and is never inferred from XML checks.
"""
from pathlib import Path
import csv
import difflib
import hashlib
import json
import posixpath
import re
import xml.etree.ElementTree as E
import zipfile

P = Path(__file__).resolve().parent
OPS = P.parent / 'operations'
PRIOR = P.parents[1] / 'monitor_20260922T115605Z/operations'
NS = {'a': 'http://schemas.openxmlformats.org/drawingml/2006/main',
      'p': 'http://schemas.openxmlformats.org/presentationml/2006/main'}
checks = []


def ck(name, ok, detail=None):
    checks.append({'name': name, 'pass': bool(ok), 'detail': detail})


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canon(data):
    root = E.fromstring(data)
    for elem in root.iter():
        if elem.tag.endswith('}tableStyleId'):
            elem.text = 'STYLE_ID'
        for attr in ['styleId', 'def']:
            if attr in elem.attrib:
                elem.attrib[attr] = 'STYLE_ID'
    return E.tostring(root)


def txt(root):
    return '\n'.join(elem.text or '' for elem in root.findall('.//a:t', NS))


def cells(root):
    return [['\n'.join(elem.text or '' for elem in cell.findall('.//a:t', NS))
             for cell in row.findall('a:tc', NS)] for row in root.findall('a:tr', NS)]


def tables(markdown):
    return [match.group(0).strip().splitlines()
            for match in re.finditer(r'(?m)^\|.*(?:\n\|.*)*', markdown)]


def md_cells(table):
    return [[cell.strip() for cell in row.strip('|').split('|')]
            for row in table if not re.fullmatch(r'[| :\-]+', row)]


def image_refs(markdown):
    return re.findall(r'(?im)!\[[^\]]*\]\([^\n]*\)|^\[image\d+\]:.*$|!\[\]\[image\d+\]', markdown)


def image_rels(z, slide):
    path = f'ppt/slides/_rels/slide{slide}.xml.rels'
    if path not in z.namelist():
        return {}
    return {elem.attrib['Id']: hashlib.sha256(z.read(posixpath.normpath(
        'ppt/slides/' + elem.attrib['Target']))).hexdigest()
            for elem in E.fromstring(z.read(path)) if elem.attrib['Type'].endswith('/image')}


def notes_text(z, slide):
    rel = E.fromstring(z.read(f'ppt/slides/_rels/slide{slide}.xml.rels'))
    target = next((elem.attrib['Target'] for elem in rel if elem.attrib['Type'].endswith('/notesSlide')), None)
    return txt(E.fromstring(z.read(posixpath.normpath('ppt/slides/' + target)))) if target else ''


def theme_bindings(z):
    bindings = {}
    for name in z.namelist():
        if not name.endswith('.rels'):
            continue
        for elem in E.fromstring(z.read(name)):
            if elem.attrib['Type'].endswith('/theme'):
                owner = posixpath.dirname(posixpath.dirname(name))
                target = posixpath.normpath(posixpath.join(owner, elem.attrib['Target']))
                bindings[name + '#' + elem.attrib['Id']] = hashlib.sha256(z.read(target)).hexdigest()
    return bindings


def rows(path):
    return list(csv.DictReader(path.open()))


def without_extension_and_strict_update(markdown):
    # These are the only two parent-authorized content regions for this heartbeat.
    start = markdown.index('## **21 September Extending all six chains to 40**')
    end = markdown.index('## **21 September Eight jobs completed**')
    markdown = markdown[:start] + '<EXTENSION_REGION>\n' + markdown[end:]
    strict_start = markdown.index('The graph-save/reload fix')
    strict_end = markdown.index('[Audited MIP endpoint and full logs]', strict_start)
    return markdown[:strict_start] + '<STRICT_GRAPH_GATE_REGION> ' + markdown[strict_end:]


def main():
    required = [P / name for name in ('current_before.md', 'current_after.md',
                'slides_before.pptx', 'slides_after.pptx', 'current_after.pdf', 'slides_after.pdf')]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit('Wait for final exports; missing: ' + ', '.join(missing))
    new_cg = rows(OPS / 'cg_endpoints.csv')
    prior_cg = rows(PRIOR / 'cg_endpoints.csv')
    new_mip = rows(OPS / 'mip_endpoints.csv')
    prior_mip = rows(PRIOR / 'mip_endpoint.csv')[0]
    source_artifacts = []
    for source_root, source_rows, stage, filename in [
        (OPS, new_cg, 'cg', 'cg.json'), (PRIOR, prior_cg, 'cg', 'cg.json'),
        (OPS, new_mip, 'mip', 'result.json'), (PRIOR, [prior_mip], 'mip', 'result.json')
    ]:
        for row in source_rows:
            source = source_root / 'baseline/cases' / row['case'] / stage / (row['job'] + '_r0') / filename
            source_artifacts.append(source)
            ck(f"{row['case']} {stage} cited CSV hash matches saved endpoint", source.exists() and sha(source) == row['result_sha256'])
    k33 = {row['case']: row for row in prior_cg + new_cg if row['case'].endswith('_k33')}
    fleet = {row['case']: [str(round(float(row['fleet']))), str(round(float(row['finite_pool_fleet_bound'])))] for row in new_mip}
    fleet[prior_mip['case']] = [prior_mip['buses'], str(round(float(prior_mip['finite_pool_bound'])))]
    expected = {f'C{case[1]}': [row['trips'], f"{float(row['reported_cg_wall_s']) / 60:.1f}",
                               str(round(float(row['route_weight']))), *fleet.get(case, ['MIP running', '—'])]
                for case, row in k33.items()}
    ck('all eight cited CG sources have wall limits and no pricing certificate',
       all(row['pricing_certified'] == 'False' and row['stop_reason'] == 'wall_limit'
           for row in prior_cg + new_cg))
    ck('all three new MIPs time out twice with no fleet proof or target attainment',
       all(row['stage1_status'] == 'TIME_LIMIT' and row['stage2_status'] == 'TIME_LIMIT'
           and row['fleet_proven'] == 'False' and row['target_attained'] == 'False' for row in new_mip))
    ck('previous C3 MIP retains 34 buses, pool bound 33 and unresolved proof',
       prior_mip['buses'] == '34' and round(float(prior_mip['finite_pool_bound'])) == 33
       and prior_mip['fleet_proven'] == 'False' and prior_mip['target_attained'] == 'False')
    before = (P / 'current_before.md').read_text()
    after = (P / 'current_after.md').read_text()
    bt, at = tables(before), tables(after)
    old_target = [table for table in bt if 'Chain, k=33' in table[0]]
    new_target = [table for table in at if 'Chain, k=33' in table[0]]
    ck('exactly one existing k33 Doc table retained', len(old_target) == len(new_target) == 1)
    ck('Doc table count unchanged', len(bt) == len(at), {'before': len(bt), 'after': len(at)})
    ck('all other Doc table bytes unchanged', [t for t in bt if t not in old_target] == [t for t in at if t not in new_target])
    ck('Doc figure references unchanged', image_refs(before) == image_refs(after))
    ck('Doc content outside extension and strict gate region unchanged',
       without_extension_and_strict_update(before) == without_extension_and_strict_update(after))
    extension_start = '## **21 September Extending all six chains to 40**'
    extension_end = '## **21 September Eight jobs completed**'
    before_extension = before.split(extension_start, 1)[1].split(extension_end, 1)[0]
    after_extension = after.split(extension_start, 1)[1].split(extension_end, 1)[0]
    ck('extension historical setup preserved', before_extension.split('**22 September,', 1)[0] == after_extension.split('**22 September,', 1)[0])
    ck('extension baseline physics, full40 scope and geography links preserved',
       before_extension.split('This extension retains', 1)[1] == after_extension.split('This extension retains', 1)[1])
    doc_table = md_cells(new_target[0]) if len(new_target) == 1 else []
    ck('Doc existing column headings retained', bool(doc_table) and doc_table[0] == md_cells(old_target[0])[0])
    ck('Doc has exactly five k33 data rows in chain order', [row[0] for row in doc_table[1:]] == sorted(expected))
    doc_rows = {row[0]: row[1:] for row in doc_table[1:]}
    for chain, values in sorted(expected.items()):
        actual = doc_rows.get(chain, [])
        ck(f'Doc {chain} k33 trips, time and restricted route weight match source', actual[:3] == values[:3], {'actual': actual, 'expected': values})
        ck(f'Doc {chain} k33 integer endpoint matches source', len(actual) == 4 and (
            (' / '.join(values[3:]) in actual[3]) if chain in ['C1', 'C2', 'C3', 'C4'] else 'MIP running' in actual[3]))
    changes = [{'tag': tag, 'before_range': [i + 1, j], 'after_range': [k + 1, l],
                'before': before.splitlines()[i:j], 'after': after.splitlines()[k:l]}
               for tag, i, j, k, l in difflib.SequenceMatcher(None, before.splitlines(), after.splitlines()).get_opcodes() if tag != 'equal']
    with zipfile.ZipFile(P / 'slides_before.pptx') as bz, zipfile.ZipFile(P / 'slides_after.pptx') as az:
        count = lambda z: len([name for name in z.namelist() if re.fullmatch(r'ppt/slides/slide\d+.xml', name)])
        ck('slide count remains 42', count(bz) == count(az) == 42)
        raw_changes = [i for i in range(1, 43) if bz.read(f'ppt/slides/slide{i}.xml') != az.read(f'ppt/slides/slide{i}.xml')]
        semantic = [i for i in range(1, 43) if canon(bz.read(f'ppt/slides/slide{i}.xml')) != canon(az.read(f'ppt/slides/slide{i}.xml'))]
        ck('only authorized slides 10 and 42 change semantically', 42 in semantic and set(semantic) <= {10, 42}, {'raw': raw_changes, 'semantic': semantic})
        notes_changes = [i for i in range(1, 43) if notes_text(bz, i) != notes_text(az, i)]
        ck('only authorized slide notes change', set(notes_changes) <= {10, 42}, notes_changes)
        media = lambda z: sorted(hashlib.sha256(z.read(name)).hexdigest() for name in z.namelist() if name.startswith('ppt/media/'))
        ck('all slide media bytes preserved', media(bz) == media(az))
        ck('all original slide image bindings preserved', all(image_rels(bz, i) == image_rels(az, i) for i in range(1, 43)))
        master_parts = [name for name in bz.namelist() if name.endswith('.xml') and name.startswith(('ppt/slideMasters/', 'ppt/slideLayouts/', 'ppt/notesMasters/'))]
        master_changes = [name for name in master_parts if name not in az.namelist() or canon(bz.read(name)) != canon(az.read(name))]
        ck('masters and layouts preserve existing headers and footers', not master_changes, master_changes)
        ck('theme bytes preserved despite export filename permutation',
           sorted(hashlib.sha256(bz.read(n)).hexdigest() for n in bz.namelist() if n.startswith('ppt/theme/') and n.endswith('.xml')) ==
           sorted(hashlib.sha256(az.read(n)).hexdigest() for n in az.namelist() if n.startswith('ppt/theme/') and n.endswith('.xml')))
        ck('all effective theme bindings preserved', theme_bindings(bz) == theme_bindings(az), theme_bindings(az))
        oldstyles = E.fromstring(bz.read('ppt/tableStyles.xml'))
        newstyles = E.fromstring(az.read('ppt/tableStyles.xml'))
        ck('original table style definitions preserved', all(any(canon(E.tostring(x)) == canon(E.tostring(y)) for y in newstyles) for x in oldstyles))
        slide = E.fromstring(az.read('ppt/slides/slide42.xml'))
        stables = slide.findall('.//a:tbl', NS)
        ck('slide 42 remains one native editable table', len(stables) == 1)
        slide_table = cells(stables[0]) if stables else []
        ck('slide 42 existing column headings retained', bool(slide_table) and slide_table[0] == cells(E.fromstring(bz.read('ppt/slides/slide42.xml')).find('.//a:tbl', NS))[0])
        ck('slide 42 has exactly five k33 data rows in chain order', [row[0] for row in slide_table[1:]] == sorted(expected))
        slide_rows = {row[0]: row[1:] for row in slide_table[1:]}
        for chain, values in sorted(expected.items()):
            ck(f'slide 42 {chain} all data cells match exact endpoint sources', slide_rows.get(chain) == values,
               {'actual': slide_rows.get(chain), 'expected': values})
        size = E.fromstring(az.read('ppt/presentation.xml')).find('p:sldSz', NS)
        bounds = []
        for frame in slide.findall('.//p:graphicFrame', NS):
            table = frame.find('.//a:tbl', NS)
            off = frame.find('p:xfrm/a:off', NS)
            if table is None or off is None:
                continue
            x, y = int(off.attrib['x']), int(off.attrib['y'])
            width = sum(int(elem.attrib['w']) for elem in table.findall('a:tblGrid/a:gridCol', NS))
            height = sum(int(elem.attrib['h']) for elem in table.findall('a:tr', NS))
            bounds.append([x, y, width, height])
            ck('slide 42 native table grid fits slide', x >= 0 and y >= 0 and x + width <= int(size.attrib['cx']) and y + height <= int(size.attrib['cy']))
        s42 = txt(slide)
        s42notes = notes_text(az, 42)
        s10b = txt(E.fromstring(bz.read('ppt/slides/slide10.xml')))
        s10a = txt(E.fromstring(az.read('ppt/slides/slide10.xml')))
        s10notes = notes_text(az, 10)
        ck('slide 10 prior scientific endpoint facts preserved', s10b.split('Duplicates and charger conflicts remain.')[0] == s10a.split('Duplicates and charger conflicts remain.')[0])
    ck('both publications retain uncertified RMP scope',
       all(term in after for term in ['final restricted-master route weight', 'without a pricing certificate', 'not certified LP lower bounds'])
       and 'not certified LP lower bounds' in s42 and 'restricted-master route weights' in s42notes
       and 'neither the weighted objective nor a certified full-model LP lower bound' in s42notes)
    ck('all four MIPs remain open with physical limitations',
       all(term in after for term in ['All four completed MIPs missed 33', 'both time limits', 'duplicate removal and shared capacity remain unvalidated'])
       and all(term in s42notes for term in ['Both stages TIME_LIMIT in every case', 'none is a target hit or a fleet proof', 'finite-pool bounds']))
    ck('extra trip assignments agree with new and prior MIP sources',
       'C1 315, C2 225, C3 118, C4 285' in after and '315/225/118/285' in s42notes)
    ck('current queue snapshot is scoped and numerically consistent',
       all(term in after for term in ['11:58 EDT', '33/44 graphs', '20 jobs run (11 graphs, five CGs, four MIPs)', '75 pending solver dependencies'])
       and all(term in s42notes for term in ['15:58:38 UTC (11:58 EDT)', '33/44 graphs ready', '20 running jobs (11 graph, 5 CG, 4 MIP)', '75 genuine pending dependencies']))
    ck('unfinished C5 graph and C6 MIP remain pending and running respectively',
       'C5 k33 still awaits its graph' in after and 'C5 awaits its graph' in s42
       and 'C6 MIP is running' in s42notes)
    build_times = [float(row['graph_original_build_s']) / 3600 for row in k33.values()]
    graph_range = f'{min(build_times):.2f}–{max(build_times):.2f}'
    ck('graph timing range and prior-CG exclusion match k33 sources',
       graph_range in after and graph_range in s42 and 'CG minutes exclude earlier smaller instances' in after
       and 'prior CG stages excluded' in s42, graph_range)
    ck('k34 endpoints retain separate target and uncertified scope',
       'At k34, C1/C3/C4 also reached four hours without certificates; fractional weights are 33/34/33' in after
       and 'New k34 CGs C1/C3/C4 also stopped at four hours, uncertified' in s42notes)
    ck('new precise weighted objectives preserved in slide notes',
       all(f"{float(row['weighted_RMP']):.6f}" in s42notes or f"{float(row['weighted_RMP']):.9f}" in s42notes for row in new_cg))
    ck('C2 positive-only numerical discrepancy disclosed',
       '0.038 cost units' in after and '0.037537633 higher' in s42notes and 'no tolerance or certificate was changed' in s42notes)
    ck('unchanged baseline and source dirty-flag scope retained',
       all(term in s42notes for term in ['240 kWh battery/initial SOC', 'constant 240 kW charging', 'zero reserve',
                                       'no terminal floor/shared capacity', 'flat tariff', 'start fee 5', 'covering',
                                       '871d057 is clean', 'a0e0bb7 records git_dirty=true']))
    gate_root = P.parents[1] / 'strict_graph_reuse/production_gate'
    gate_verification = json.loads((gate_root / 'verification.json').read_text())
    gate_dir = gate_root / 'collections/768638_r0'
    gate = json.loads((gate_dir / 'gate_result.json').read_text())
    ck('native gate receipt has nine passing checks, separate from old 13 cache tests',
       gate_verification['status'] == 'passed' and len(gate_verification['checks']) == 9 and all(gate_verification['checks'].values()))
    ck('native gate receipts match actual collected file hashes',
       all(sha(gate_dir / name) == digest for name, digest in gate_verification['file_sha256'].items()))
    ck('native gate validates exactly inherited and saved singleton records without graph or solver',
       gate['status'] == 'passed' and gate['inherited_routes'] == gate['inherited_routes_replayed'] == 8343
       and gate['all_routes_physically_replayed'] == 8397 and gate['singleton_routes'] == 54
       and gate['inherited_full_record_equality'] is True and gate['comparison_normalized_fields'] == ['cg_checkpoint_id']
       and gate['graph_built'] is False and gate['solver_started'] is False
       and gate['fresh_singleton_optimum_equality_proved'] is False and gate['cg_pricing_certificate'] is None)
    gate_snapshot = json.loads((gate_root / 'native_snapshot.json').read_text())
    ck('native gate 35-second completion is scheduler time',
       '768638|COMPLETED|0:0|00:00:35|' in gate_snapshot['accounting']
       and '35 seconds' in after and '35 scheduler seconds' in s10notes)
    ck('published native gate counts and limitations match receipts',
       all(term in after for term in ['8,343 inherited records', '54 new singleton routes', 'job 768638', 'not a new CG result',
                                    'Full 331-trip graph preparation and reload comparison must pass before new CG'])
       and 'All 8,397 initial routes pass native replay' in s10a and 'full-size graph validation remains required' in s10a
       and all(term in s10notes for term in ['normalizing only cg_checkpoint_id', 'No graph or solver was run', 'does not prove']))
    live_receipt = json.loads((P / 'live_ui_verification.json').read_text())
    ck('root live UI receipt records saved state and seven visible Doc tabs',
       live_receipt['doc_save_status'] == 'Saved to Drive' and len(live_receipt['observed_tabs']) == 7
       and live_receipt['other_doc_tabs_edited'] is False and live_receipt['historical_decks_edited'] is False)
    inputs = required + [OPS / name for name in ('README.md', 'cg_endpoints.csv', 'mip_endpoints.csv', 'verified_summary.json', 'queue.txt')]
    inputs += [PRIOR / 'cg_endpoints.csv', PRIOR / 'mip_endpoint.csv']
    inputs += source_artifacts
    inputs += [gate_root / 'verification.json', gate_root / 'native_snapshot.json', P / 'live_ui_verification.json']
    inputs += [gate_dir / name for name in gate_verification['file_sha256']]
    review_path = P / 'audit_renders/visual_review.json'
    visual = json.loads(review_path.read_text()) if review_path.exists() else {'status': 'pending', 'reason': 'Actual page and slide inspection required.'}
    ck('final exports visually inspected', visual.get('status') == 'pass' and
       all(sha(P / name) == digest for name, digest in visual.get('export_sha256', {}).items()) and
       set(visual.get('export_sha256', {})) == {'current_after.pdf', 'slides_after.pdf'}, visual)
    report = {'status': 'pass' if all(check['pass'] for check in checks) else 'fail',
              'check_count': len(checks), 'checks': checks,
              'failed': [check['name'] for check in checks if not check['pass']],
              'source_expected_k33': expected, 'slide42_table': slide_table, 'doc_k33_table': doc_table,
              'front_diff': changes, 'slide10_before': s10b, 'slide10_after': s10a,
              'slide42_text': s42, 'table_bounds_emu': bounds, 'visual_review': visual,
              'slide42_notes': s42notes, 'slide10_notes': s10notes, 'root_live_ui_receipt': live_receipt,
              'source_hashes': {str(path): sha(path) for path in inputs if path.exists()},
              'limitations': ['Markdown verifies front-tab content, not native live Doc schema or other tabs.',
                             'Only table-style identifiers are normalized; style definitions are separately checked. Theme filenames were permuted, so every resolved relationship and complete theme byte hash is checked.',
                             'Figures and historical text in the front-tab exports and every slide are compared. Historical Doc tabs require separate publication UI evidence.',
                             'A source CSV audit is not a new pricing certificate or independent whole-journal physical replay.']}
    (P / 'verification.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'status': report['status'], 'checks': report['check_count'], 'failed': report['failed']}, indent=2))


if __name__ == '__main__':
    main()
