import fs from 'node:fs/promises';
import path from 'node:path';
import {PresentationFile,FileBlob} from '@oai/artifact-tool';
const root='C:/Users/mooooonesy/Downloads/pennstuff/cis 6270/Project 1';
const file=path.join(root,'slides/deck_versions/pptx_v4_2026-09-29/output/BDG_Defense_Group_2_v4_revised.pptx');
const out=path.join(root,'slides/.build_v8/final_v4');await fs.mkdir(out,{recursive:true});
const p=await PresentationFile.importPptx(await FileBlob.load(file));
for(let i=0;i<p.slides.items.length;i++){
 const png=await p.slides.items[i].export({format:'png',scale:1});
 await fs.writeFile(path.join(out,`slide-${String(i+1).padStart(2,'0')}.png`),new Uint8Array(await png.arrayBuffer()));
}
console.log('Rendered final PPTX:',p.slides.items.length);
