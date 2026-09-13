import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {FileBlob,SpreadsheetFile} from '@oai/artifact-tool';
const root=path.dirname(fileURLToPath(import.meta.url));
const mode=process.argv[2]||'after';
const wb=await SpreadsheetFile.importXlsx(await FileBlob.load(path.join(root,'EVSP_DR_Experiment_Register.xlsx')));
const register=JSON.parse(await fs.readFile(path.join(root,'../research_register/register.json'),'utf8'));
const checks={mode,records:register.rows.length,views:[]};
const views=[['Campaigns','A1:G14']];
if(mode==='after') {
 for(const name of ['Campaigns','CG','MIP','Attempts']) {
  const sheet=wb.worksheets.getItem(name);const values=sheet.getUsedRange().values;
  const i=values.findIndex(row=>row[0]==='cumulative_budget_20260913');
  if(i>=0)views.push([name,`A${Math.max(1,i)}:${name==='CG'?'N':name==='MIP'?'L':'H'}${Math.min(values.length,i+10)}`]);
 }
 checks.formulaScan=(await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!',options:{useRegex:true,maxResults:30},summary:'Saved workbook error scan'})).ndjson;
}
for(const [name,range] of views){
 const check=await wb.inspect({kind:'table',range:`${name}!${range}`,include:'values,formulas',tableMaxRows:12,tableMaxCols:14,maxChars:12000});
 checks.views.push({name,range,inspection:check.ndjson});
 const image=await wb.render({sheetName:name,range,scale:1.2,format:'png'});
 const filename=`${mode}_${name}_${range.replace(':','_')}.png`;
 await fs.writeFile(path.join(root,'previews',filename),new Uint8Array(await image.arrayBuffer()));
}
await fs.writeFile(path.join(root,`${mode}_verification.json`),JSON.stringify(checks,null,2));
console.log(JSON.stringify({mode,views:views.length,rows:register.rows.length}));
