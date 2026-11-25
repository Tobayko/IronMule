"""ADM1: a wave's requests from the client's send through the service's admit and hand-off to the engine's run and
gathering window (ms from the wave's start).  python lab/adm1/adm.py TIMELINE STABILITY_LOG TRACE"""
import json,sys,collections
tl=sys.argv[1]; st=sys.argv[2]; tr=sys.argv[3]
reqs=[json.loads(l) for l in open(tl)]
ev=collections.defaultdict(dict)
for l in open(st):
    e=json.loads(l)
    if e['event'] in('admit','join','end'):
        ev[e['id']][e['event']]=e
    if e['event']=='group':
        for i in e['ids']: ev[i]['group']=e
runs=[];gath=[]
for l in open(tr):
    if l.startswith('# run '): runs.append(float(l.split()[2]))
    if l.startswith('# gathered'): gath.append((float(l.split()[-1]),int(l.split()[2])))
waves=collections.defaultdict(list)
for r in reqs: waves[(r['repeat'],r['width'],r['wave_start'])].append(r)
for k,rs in sorted(waves.items(), key=lambda x:x[0][2]):
    w0=k[2]
    sends=sorted(r['t_send']-w0 for r in rs)
    adm=sorted(ev[r['server_id']]['admit']['t']-w0 for r in rs if 'admit' in ev[r['server_id']])
    hand=sorted(((ev[r['server_id']].get('join') or ev[r['server_id']].get('group'))['t']-w0) for r in rs)
    disp=sorted(ev[r['server_id']]['end']['t_dispatch']-w0 for r in rs)
    run=[x-w0 for x in runs if 0<x-w0<1]
    g=[(x-w0,n) for x,n in gath if 0<x-w0<1]
    f=lambda v:"%.0f/%.0f/%.0f"%(1e3*v[0],1e3*v[len(v)//2],1e3*v[-1])
    print(k[1],k[0],"send",f(sends),"admit",f(adm),"handoff",f(hand),"dispatch",f(disp),"run",[round(1e3*x) for x in run],"gathered",[(round(1e3*x),n) for x,n in g])
