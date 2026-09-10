import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { Workbook, SpreadsheetFile } from '@oai/artifact-tool';

const root = path.dirname(fileURLToPath(import.meta.url));
const out = path.resolve(root, '../01a07ecc-9b77-79b3-9782-e4308a80ba07');
const register = JSON.parse(await fs.readFile(path.join(root, 'register.json'), 'utf8'));
const wb = Workbook.create();
const previews = path.join(out, 'previews');
await fs.mkdir(previews, { recursive: true });
const definitions = [
  ['Record', 'One source artifact, solver stage, comparison arm, or execution observation. Record counts are not independent experiment counts.'],
  ['LP objective', 'The recorded objective. Read its objective type and source scope; some prototypes minimize fleet only.'],
  ['Fractional buses', 'Sum of LP route weights. It is not the weighted objective divided by 100,000 and is not automatically a fleet-only lower bound.'],
  ['Pricing certified', 'Source-reported certificate within its graph, assumptions, and reduced-cost tolerance. It is not a continuous-physics certificate.'],
  ['Pool fleet proved', 'The supplied column pool cannot provide fewer buses. This is not a branch-and-price proof for all possible routes.'],
  ['Charging cost', 'Route electricity cost plus the modeled charging-start fee when the source defines it that way. Grid cost and continuous replay are distinct.'],
  ['Authority', 'Superseded and rejected artifacts remain in the record but must not be used as authoritative results.'],
  ['Blank numeric cell', 'Not recorded in the source snapshot or not applicable. A real zero remains numeric zero.'],
  ['Physical checks', 'Individual route replay, duplicate coverage removal and shared charger capacity are distinct claims; read the recorded validation scope.'],
  ['Refresh', 'Regenerate from a dated collector snapshot with build_register.py, then run build_workbook.mjs. This workbook is a timestamped view, not a live cluster connection.'],
];
const cell = value => value == null ? null : typeof value === 'object' ? JSON.stringify(value) : value;
const numeric = value => typeof value === 'number' && Number.isFinite(value) ? value : null;
const pretty = value => value == null ? 'unknown' : String(value).replaceAll('_', ' ');
const col = n => { let s=''; for (n++; n>0; n=Math.floor((n-1)/26)) s=String.fromCharCode(65+(n-1)%26)+s; return s; };
const bool = v => v === true ? 'yes' : v === false ? 'no' : 'unknown';
const field = (key, label, width=18, format=null) => ({key,label,width,format});
const common = [field('campaign_id','Campaign / source group',32),field('case_id','Case',38),field('stage','Stage',17),field('substage','Arm / substage',27),field('target_k','Target buses',12,'0'),field('trip_count','Trips',10,'0')];
const settings = [field('master_sense','Coverage rule',16),field('initialization','CG initialization',24),field('column_pool_treatment','Pool treatment',22),field('battery_kwh','Battery (kWh)',15,'0.0'),field('charge_kw','Charge power (kW)',18,'0.0'),field('parx_kw','PARX power (kW)',17,'0.0'),field('capacity_enforced','Shared capacity enforced',20),field('soc_step_kwh','SOC step (kWh)',17,'0.0'),field('block_minutes','Time step (min)',16,'0'),field('tariff_path','Tariff source',70),field('terminal_energy_policy','Terminal-energy policy',45)];
const provenance = [field('authority_role','Authority',26),field('artifact_status','Artifact state',25),field('proof_scope','Proof scope',85),field('physical_selected_validated','Selected physical check',23),field('physical_validation_scope','Physical check scope',85),field('overcovered_trips','Overcovered trips',17,'0'),field('cross_route_capacity_validated','Shared-capacity check',24),field('job_ids','Job IDs',36),field('code_commit','Execution commit',45),field('input_sha256','Input SHA-256',72),field('source_path','Original source path',100),field('source_sha256','Original SHA-256',72),field('snapshot_payload_sha256','Snapshot payload SHA-256',72),field('row_id','Stable record ID',72),field('snapshot_time_utc','Snapshot time (UTC)',28),field('limitations','Limitations',100),field('notes','Source notes',100)];
const cgFields = [...common,field('recorded_lp_objective','LP objective',19,'#,##0.000000'),field('lp_objective_kind','Objective type',25),field('fractional_fleet','Fractional buses',19,'0.000000'),field('full_model_lp_certified','Pricing certified',19),field('stop_reason','CG stopping reason',42),field('cg_iterations','CG iterations',15,'0'),field('runtime_s','Runtime (s)',17,'#,##0.00'),field('pool_size','Pool columns',15,'#,##0'),field('min_reduced_cost','Minimum reduced cost',23,'0.000000E+00'),field('artificial_total','Artificial coverage',20,'0.000000'),field('lp_bound_scope','LP certificate scope',85),...settings,...provenance];
const mipFields = [...common,field('mip_incumbent_fleet','Integer buses',15,'0'),field('mip_bound_fleet','Pool fleet bound',18,'0.000000'),field('fleet_proven','Pool fleet proved',19),field('fleet_excess','Buses above target',20,'0'),field('mip_status','MIP status',24),field('pool_size','Pool columns',15,'#,##0'),field('runtime_s','Runtime (s)',17,'#,##0.00'),field('stage1_incumbent_fleet','Stage 1 buses',18,'0'),field('stage1_bound','Stage 1 fleet bound',20,'0.000000'),field('stage1_proven','Stage 1 fleet proved',21),field('stage2_status','Stage 2 status',24),field('stage2_charging_cost','Stage 2 charging cost',24,'#,##0.000000'),field('stage2_charging_bound','Stage 2 cost bound',24,'#,##0.000000'),field('stage2_gap','Stage 2 relative gap',23,'0.00%'),field('charging_cost_continuous','Continuous replay cost',25,'#,##0.000000'),...settings,...provenance];
const chargingFields = [...common,field('mip_incumbent_fleet','Integer buses',15,'0'),field('charging_cost_grid','Grid charging cost',24,'#,##0.000000'),field('charging_cost_continuous','Continuous charging cost',26,'#,##0.000000'),field('charging_cost_exact','Exact charging cost',24,'#,##0.000000'),field('charging_cost_lower','Charging cost lower',24,'#,##0.000000'),field('charging_cost_upper','Charging cost upper',24,'#,##0.000000'),field('terminal_energy_grid_kwh','Grid end energy (kWh)',25,'#,##0.000000'),field('terminal_energy_continuous_kwh','Continuous end energy (kWh)',29,'#,##0.000000'),field('mip_gap','MIP relative gap',20,'0.00%'),field('completion_marker_matches','Completion hash matches',24),...settings,...provenance];
const attemptFields = [...common,field('result_family','Record family',32),field('artifact_status','Artifact state',28),field('workflow_state','Scheduler / workflow state',32),field('dependency','Dependency',50),field('stop_reason','Stop / error reason',65),field('runtime_s','Runtime (s)',18,'#,##0.00'),field('phase_runtime_json','Timing components (s)',90),...provenance.filter(x=>x.key!=='artifact_status')];

const groups = {CG: [], MIP: [], Charging: [], Attempts: []};
for (const r0 of register.rows) {
  const r = {...r0};
  r.recorded_lp_objective ??= r.weighted_lp_objective;
  r.lp_objective_kind ??= null;
  if (['cg','cg_arm','lp'].includes(r.stage)) groups.CG.push(r);
  else if (r.stage==='mip' || r.stage.startsWith('mip_')) groups.MIP.push(r);
  else if (['comparison','comparison_arm','frontier'].includes(r.stage)) groups.Charging.push(r);
  else groups.Attempts.push(r);
}
if (Object.values(groups).reduce((s,r)=>s+r.length,0)!==register.rows.length) throw Error('Record partition mismatch');
const overview = wb.worksheets.add('Campaigns');
const sheets = {};
for (const [name, fields] of Object.entries({CG:cgFields,MIP:mipFields,Charging:chargingFields,Attempts:attemptFields})) {
  sheets[name] = wb.worksheets.add(name);
}
const historySheet = wb.worksheets.add('History');
const readme = wb.worksheets.add('Definitions');
const sourceTime = register.source_snapshot?.timestamp_utc || register.source_snapshot?.captured_utc || register.rows[0]?.snapshot_time_utc || 'unknown';

function styleTable(sheet, fields, rows, name, firstRow=5) {
  sheet.showGridLines = false;
  const last=firstRow+rows.length;
  const range=sheet.getRange(`A${firstRow}:${col(fields.length-1)}${last}`);
  range.values=[fields.map(f=>f.label),...rows.map(r=>fields.map(f=>{
    const v=r[f.key];
    if (f.format) return numeric(v);
    return typeof v==='boolean'?bool(v):cell(v);
  }))];
  range.format.font={name:'Arial',size:10,color:'#1E293B'};
  range.format.rowHeight=24;
  range.format.verticalAlignment='center';
  range.format.wrapText=false;
  sheet.getRange(`A${firstRow}:${col(fields.length-1)}${firstRow}`).format={fill:'#233B5A',font:{name:'Arial',size:10,bold:true,color:'#FFFFFF'},rowHeight:40,wrapText:true,horizontalAlignment:'center',verticalAlignment:'center'};
  for (let i=0;i<fields.length;i++) {
    sheet.getRange(`${col(i)}${firstRow}:${col(i)}${last}`).format.columnWidth=fields[i].width;
    if (fields[i].format && rows.length) sheet.getRange(`${col(i)}${firstRow+1}:${col(i)}${last}`).setNumberFormat(fields[i].format);
  }
  const table=sheet.tables.add(`A${firstRow}:${col(fields.length-1)}${last}`,true,name);
  table.showFilterButton=true;
  sheet.freezePanes.freezeRows(firstRow);
  sheet.freezePanes.freezeColumns(2);
  const authority=fields.findIndex(f=>f.key==='authority_role');
  if (authority>=0 && rows.length) {
    const cells=sheet.getRange(`${col(authority)}${firstRow+1}:${col(authority)}${last}`);
    cells.conditionalFormats.add('containsText',{text:'superseded',format:{fill:'#FFF1D6',font:{color:'#7A4700'}}});
    cells.conditionalFormats.add('containsText',{text:'rejected',format:{fill:'#FCE4E4',font:{color:'#9B1C1C'}}});
  }
  return last;
}
function heading(sheet,title,note) {
  sheet.getRange('A2').values=[[title]];
  sheet.getRange('A2').format.font={name:'Arial',size:15,bold:true,color:'#1E293B'};
  sheet.getRange('A3').values=[[note]];
  sheet.getRange('A3').format.font={name:'Arial',size:10,italic:true,color:'#526071'};
}

for (const [name,fields] of Object.entries({CG:cgFields,MIP:mipFields,Charging:chargingFields,Attempts:attemptFields})) {
  heading(sheets[name],`${name==='CG'?'Column generation and LP':name==='MIP'?'Integer pool solves':name==='Charging'?'Charging comparisons':'Execution and audit records'}`,`Snapshot ${sourceTime}. Blank numeric cells mean unavailable; filter by campaign, case and authority.`);
  const last=styleTable(sheets[name],fields,groups[name],`Register${name}`);
  if (name==='MIP' && groups.MIP.length) {
    sheets.MIP.getRange(`J6:J${last}`).formulas=groups.MIP.map((r,i)=>[`=IF(AND(ISNUMBER(E${i+6}),ISNUMBER(G${i+6})),G${i+6}-E${i+6},"")`]);
    sheets.MIP.getRange(`J6:J${last}`).conditionalFormats.add('colorScale',{colors:['#F8FAFC','#F7B3A6','#BA2525'],thresholds:[{type:'num',value:0},{type:'num',value:3},{type:'num',value:10}]});
  }
}

heading(overview,'EVSP–DR experiment register',`Evidence captured ${sourceTime}. Counts below are records, not independent experimental samples.`);
overview.tabColor='#233B5A';
const campaignFields=[field('campaign_id','Campaign / source group',34),field('family','Family',29),field('cg','LP / CG records',19,'0'),field('mip','MIP records',18,'0'),field('charging','Charging records',21,'0'),field('attempts','Execution / audit records',26,'0'),field('row_count','Total records',18,'0'),field('root','Cluster source directory',100),field('report_links','Local report paths',115)];
styleTable(overview,campaignFields,register.campaigns,'CampaignIndex');
register.campaigns.forEach((c,i)=>{
  const row=i+6;
  const formulas=['CG','MIP','Charging','Attempts'].map(name=>`=COUNTIFS('${name}'!$A$6:$A$${Math.max(6,groups[name].length+5)},$A${row})`);
  overview.getRange(`C${row}:F${row}`).formulas=[formulas];
  overview.getRange(`G${row}`).formulas=[[`=SUM(C${row}:F${row})`]];
});

let hist=[];
try { hist=JSON.parse(await fs.readFile(path.join(root,'historical_inventory.json'),'utf8')); if (!Array.isArray(hist)) hist=hist.rows || hist.records || []; } catch (e) { if(e.code!=='ENOENT') throw e; }
if (!hist.length) {
  try { const csv=await fs.readFile(path.join(root,'historical_inventory.csv'),'utf8'); const imported=await Workbook.fromCSV(csv,{sheetName:'Import'}); const vals=imported.worksheets.getItemAt(0).getUsedRange().values; hist=vals.slice(1).filter(r=>r.some(x=>x!==null&&x!=='')).map(r=>Object.fromEntries(vals[0].map((k,i)=>[k,r[i]??null]))); } catch(e) { if(e.code!=='ENOENT') throw e; }
}
if (!hist.length) throw Error('Historical inventory missing; finish inventory before workbook export');
const histKeys=Object.keys(hist[0]);
const histFields=histKeys.map(k=>field(k,pretty(k),/path|url|source|warning|note|scope|hash|provenance/i.test(k)?85:/title|name|campaign|family/i.test(k)?40:23));
heading(historySheet,'Historical evidence inventory','These are evidence artifacts and groups, not independent runs. Read classification and comparability notes.');
styleTable(historySheet,histFields,hist,'HistoricalInventory');
heading(readme,'Definitions and refresh','Source files are preserved. The complete normalized JSON retains fields omitted from these review views.');
styleTable(readme,[field('term','Term',31),field('meaning','Meaning',115)],definitions.map(([term,meaning])=>({term,meaning})),'RegisterDefinitions');
readme.getRange(`B6:B${definitions.length+5}`).format.wrapText=true;
readme.getRange(`A6:B${definitions.length+5}`).format.rowHeight=44;
readme.tabColor='#66758A';
wb.recalculate();
const sums=overview.getRange(`G6:G${register.campaigns.length+5}`).values.flat();
if(sums.some((v,i)=>v!==register.campaigns[i].row_count)) throw Error('Workbook campaign counts differ from normalized record counts');
const scan=await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#NUM!|#NULL!|#SPILL!|#CALC!',options:{useRegex:true,maxResults:20},summary:'Formula error scan'});
await fs.writeFile(path.join(out,'workbook_checks.json'),JSON.stringify({sourceTime,sourceRows:register.rows.length,partition:Object.fromEntries(Object.entries(groups).map(([k,v])=>[k,v.length])),historicalRows:hist.length,campaignCountCheck:true,formulaScan:scan.ndjson},null,2));
for (const name of ['Campaigns','CG','MIP','Charging','Attempts','History','Definitions']) {
  const blob=await wb.render({sheetName:name,range:name==='Definitions'?'A1:B15':'A1:H14',scale:1.3,format:'png'});
  await fs.writeFile(path.join(previews,`${name}.png`),new Uint8Array(await blob.arrayBuffer()));
}
const file=await SpreadsheetFile.exportXlsx(wb);
const output=path.join(out,'EVSP_DR_Experiment_Register.xlsx');
await file.save(output);
console.log(JSON.stringify({output,sourceTime,rows:register.rows.length,historicalRows:hist.length,sheets:7}));
