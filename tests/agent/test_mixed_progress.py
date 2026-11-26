from types import SimpleNamespace

from lab.pk6.mixed_progress.schedule import serve


def test_partial_prefills_share_real_decodes_and_keep_output_ownership():
    jobs = {n: SimpleNamespace(ids=[n+100]*k, tokens=[], cancelled=None) for n,k in enumerate([3,21,4,19,5,20,6,18])}
    kept = {s:s % 2 for s in jobs}
    kv = {s:list(j.ids[:kept[s]]) for s,j in jobs.items()}
    active = dict(reversed(jobs.items()))
    calls = []

    def decode(rows):
        assert len(rows) <= 19
        calls.append(rows)
        picks = []
        for token,pos,seq,logits in rows:
            assert pos == len(kv[seq])
            expected = jobs[seq].ids[pos] if pos < len(jobs[seq].ids) else jobs[seq].tokens[-1]
            assert token == expected
            kv[seq].append(token)
            if logits:
                picks.append(1000*seq+len(kv[seq]))
        return picks

    def take(seq, job, pick):
        assert pick == 1000*seq + len(job.ids) + len(job.tokens)
        job.tokens.append(pick)
        if len(job.tokens)==3:
            del active[seq]

    e=SimpleNamespace(_decode=decode, pack_calls=0, pack_rows=0, pack_prompts=0, observer=None, batch_cap=40)
    serve(e,[(s,j,kept[s]) for s,j in jobs.items()],active,take,None,19)
    assert all(j.tokens for j in jobs.values())
    assert e.pack_rows == sum(len(j.ids)-kept[s] for s,j in jobs.items())
    assert any(any(pos>=len(jobs[s].ids) for _,pos,s,_ in call)
               and any(pos<len(jobs[s].ids) for _,pos,s,_ in call) for call in calls)


def test_cancellation_of_partially_prefilled_job_discards_its_cache():
    jobs={i:SimpleNamespace(ids=[i]*n,tokens=[],cancelled=None) for i,n in enumerate((3,4,40,60))}
    group=[(s,j,0) for s,j in jobs.items()]
    live=jobs.copy()
    cancelled=[]
    rows=[]
    jobs[2].cancelled=lambda: any(s==2 for _,_,s,_ in rows)
    def decode(step):
        rows.extend(step)
        return [s for _,_,s,logits in step if logits]
    def finish(s,j,reason,keep):
        assert reason=="cancelled" and not keep
        cancelled.append(s)
        del live[s]
    e=SimpleNamespace(_decode=decode,pack_calls=0,pack_rows=0,pack_prompts=0,observer=None,batch_cap=32)
    serve(e,group,live,lambda s,j,t:j.tokens.append(t),finish,16)
    assert cancelled==[2] and not jobs[2].tokens and jobs[3].tokens
