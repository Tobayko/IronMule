import { readFileSync, readdirSync } from 'node:fs';
import { join } from 'node:path';
import { digest, sha } from '../src/util.js';
export function sealManifest():Record<string,string> {
  const paths=['package.json','package-lock.json','tsconfig.json','bench/PREREGISTRATION.md'];
  for(const dir of ['src','bench','schemas'])for(const file of readdirSync(dir).sort())
    if(/\.(ts|json|py)$/.test(file))paths.push(join(dir,file));
  const out:Record<string,string>={};for(const p of paths.sort())out[p]=sha(readFileSync(p));
  out['../friday_evidence/statistics.py']=sha(readFileSync('../friday_evidence/statistics.py'));
  return out;
}
export function sourceSeal():string {return digest(sealManifest());}
