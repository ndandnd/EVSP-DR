import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {FileBlob,SpreadsheetFile} from '@oai/artifact-tool';
const root=path.dirname(fileURLToPath(import.meta.url));
const mode=process.argv[2]||'after';
const label=process.argv[3]||mode;
const campaigns=process.argv[4]?.split(',')||['cumulative_budget_20260913','chain_extension_20260913','overnight_diagnostics_20260914','mip_repeatability_20260914'];
const wb=await SpreadsheetFile.importXlsx(await FileBlob.load(path.join(root,'EVSP_DR_Experiment_Register.xlsx')));
const register=JSON.parse(await fs.readFile(path.join(root,'../research_register/register.json'),'utf8'));
const checks={mode,records:register.rows.length,views:[]};
const views=[['Campaigns','A1:G14']];
if(mode==='after') {
 for(const name of ['Campaigns','CG','MIP','Attempts']) {
  const sheet=wb.worksheets.getItem(name);const values=sheet.getRange('A1:A3000').values;
  for(const campaign of campaigns) {
   if(campaign==='chain_extension_20260913' && name==='Attempts') continue;
   const indices=values.flatMap((row,i)=>row[0]===campaign?[i+1]:[]);
   if(indices.length)views.push([name,`A${Math.max(1,indices[0]-1)}:${name==='CG'?'N':name==='MIP'?'L':'H'}${Math.max(...indices)}`]);
  }
 }
 checks.formulaScan=(await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!',options:{useRegex:true,maxResults:30},summary:'Saved workbook error scan'})).ndjson;
}
for(const [name,range] of views){
 const check=await wb.inspect({kind:'table',range:`${name}!${range}`,include:'values,formulas',tableMaxRows:50,tableMaxCols:14,maxChars:25000});
 checks.views.push({name,range,inspection:check.ndjson});
 const image=await wb.render({sheetName:name,range,scale:1.2,format:'png'});
 const filename=`${label}_${name}_${range.replace(':','_')}.png`;
 await fs.writeFile(path.join(root,'previews',filename),new Uint8Array(await image.arrayBuffer()));
}
await fs.writeFile(path.join(root,`${label}_verification.json`),JSON.stringify(checks,null,2));
console.log(JSON.stringify({mode,views:views.length,rows:register.rows.length}));
