"""Strict, offline comparison of recorded original and simulator motion."""
import bisect
import json
import math
import uuid
from pathlib import Path

FIELDS=('x','y','z','vy','vz')
DEFAULT_TOLERANCES={'x':.5,'y':.5,'z':2.,'vy':5.,'vz':3.}


def read_trace(path):
    rows=[json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    headers=[row for row in rows if row.get('kind')=='header']
    if len(headers)!=1 or headers[0].get('schema_version')!=1:
        raise ValueError('Trace needs one schema_version=1 header')
    try:
        uuid.UUID(headers[0]['target_uuid'])
    except (KeyError, ValueError, TypeError, AttributeError) as error:
        raise ValueError('Trace needs a target build UUID') from error
    if any(row.get('kind')=='error' for row in rows):raise ValueError('Trace contains a capture error')
    samples=[row for row in rows if row.get('kind')=='sample']
    if len(samples)<3:raise ValueError('Trace needs at least three motion samples')
    if any(type(row.get(key)) is not bool for row in samples for key in ('grounded','flying')):
        raise ValueError('Samples need grounded and flying boolean state')
    times=[sample['t'] for sample in samples]
    if any(not math.isfinite(value) for row in samples for value in [row['t'],*[row[k] for k in FIELDS]]):
        raise ValueError('Trace contains nonfinite motion values')
    if any(b<=a for a,b in zip(times,times[1:])):
        raise ValueError('Capture must cover one uninterrupted run with increasing time')
    if any(b-a>.1 for a,b in zip(times,times[1:])):
        raise ValueError('Trace has a sampling gap over 100 ms')
    if any(abs(b['z']-a['z'])>100 for a,b in zip(samples,samples[1:])):
        raise ValueError('World-origin reset or discontinuity; capture a shorter scenario')
    return headers[0],samples,[row for row in rows if row.get('kind')=='action']


def compare_traces(reference_path,candidate_path,tolerances=None,action_time_tolerance=.025):
    ref,original,ref_actions=read_trace(reference_path)
    sim,candidate,sim_actions=read_trace(candidate_path)
    if ref.get('source')!='original_game':
        raise ValueError('Reference must be an original_game capture; synthetic fixtures cannot certify parity')
    if sim.get('source')!='simulator':raise ValueError('Candidate must be a simulator trace')
    if ref.get('scenario')!=sim.get('scenario'):raise ValueError('Scenario identifiers differ')
    if ref.get('target_uuid')!=sim.get('target_uuid'):raise ValueError('Target build identities differ')
    tolerances=DEFAULT_TOLERANCES if tolerances is None else tolerances
    if set(tolerances)!=set(FIELDS) or any(not math.isfinite(x) or x<0 for x in tolerances.values()):
        raise ValueError('Specify one finite, nonnegative tolerance for each motion field')
    ref_start,sim_start=original[0]['t'],candidate[0]['t']
    times=[sample['t']-sim_start for sample in candidate]
    end=original[-1]['t']-ref_start
    if times[-1]+1e-6<end:raise ValueError('Candidate does not cover the full original trace')
    errors={key:[] for key in FIELDS};grounded_errors=flying_errors=0
    for row in original:
        t=row['t']-ref_start
        i=min(len(candidate)-2,max(0,bisect.bisect_right(times,t)-1))
        a,b=candidate[i:i+2];u=(t-times[i])/(times[i+1]-times[i])
        for key in FIELDS:
            errors[key].append(abs(row[key]-(a[key]+(b[key]-a[key])*u)))
        nearest=a if u<.5 else b
        grounded_errors+=row.get('grounded')!=nearest.get('grounded')
        flying_errors+=row['flying']!=nearest['flying']
    metrics={key:{'rmse':math.sqrt(sum(x*x for x in values)/len(values)),
                  'max_error':max(values),'tolerance':tolerances[key],
                  'passed':max(values)<=tolerances[key]} for key,values in errors.items()}
    # Inputs are compared as attempted impulses, including rejected attempts.
    actions_match=len(ref_actions)==len(sim_actions) and all(
        a['action']==b['action'] and abs((a['t']-ref_start)-(b['t']-sim_start))<=action_time_tolerance
        for a,b in zip(ref_actions,sim_actions))
    return {'schema_version':1,'scenario':ref['scenario'],'target_uuid':ref['target_uuid'],
            'reference_samples':len(original),'duration':end,'time_alignment':'first sample',
            'position_alignment':'none','metrics':metrics,'grounded_mismatches':grounded_errors,
            'flying_mismatches':flying_errors,
            'actions_match':actions_match,'action_time_tolerance':action_time_tolerance,
            'passed':all(v['passed'] for v in metrics.values()) and grounded_errors==0 and flying_errors==0 and actions_match,
            'scope':'This motion scenario only; does not certify full-game fidelity'}
