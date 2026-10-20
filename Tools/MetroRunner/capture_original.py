"""Attach an observation probe to an explicitly selected original-game PID.

Optional dependency: Frida's Python SDK. No spawn, input injection, or game
state writes. The target UUID and hook bytes are checked before attachment.
"""
import argparse
import json
from pathlib import Path
import threading


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',required=True,help='Exact Frida device ID of the authorized iPhone')
    parser.add_argument('--pid',required=True,type=int,help='Already running Subway Surfers process ID')
    parser.add_argument('--scenario',required=True)
    parser.add_argument('--seconds',type=float,default=15.)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.seconds<=0:parser.error('--seconds must be positive')
    try:
        import frida
    except ImportError:
        parser.error('The optional frida Python package is needed for original-device capture')
    config={'scenario':args.scenario}
    source=Path(__file__).with_name('original_probe.js').read_text().replace(
        '/*PROBE_CONFIG*/ null',json.dumps(config))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    done=threading.Event();errors=[];counts={'sample':0}
    # Refuse to overwrite a prior evidence capture.
    with args.output.open('x') as output:
        def receive(message,data):
            if message['type']=='error':
                errors.append(message.get('description','Probe failed'));done.set();return
            if message['type']!='send':return
            payload=message['payload']
            rows=payload['rows'] if payload.get('kind')=='batch' else [payload]
            for row in rows:
                output.write(json.dumps(row,allow_nan=False)+'\n')
                if row.get('kind')=='sample':counts['sample']+=1
                if row.get('kind')=='error':errors.append(row['message']);done.set()
            output.flush()
        device=frida.get_device(args.device,timeout=5)
        session=device.attach(args.pid)
        session.on('detached',lambda *unused:done.set())
        script=session.create_script(source)
        script.on('message',receive)
        try:
            script.load()
            done.wait(args.seconds)
            script.exports_sync.stop()
        finally:
            session.detach()
    if errors:raise SystemExit('; '.join(errors))
    if not counts['sample']:raise SystemExit('No player samples captured; this file cannot validate fidelity')
    print(f'Captured {counts["sample"]} original player samples to {args.output}')


if __name__=='__main__':main()
