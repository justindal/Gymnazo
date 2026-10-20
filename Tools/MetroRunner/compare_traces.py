import argparse
import json
from pathlib import Path
from trace_tools import compare_traces


def main():
    parser=argparse.ArgumentParser(description='Compare a real original-game motion capture with a simulator trace')
    parser.add_argument('reference',type=Path)
    parser.add_argument('candidate',type=Path)
    parser.add_argument('--tolerances',type=Path,help='JSON mapping x,y,z,vy,vz to maximum errors')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    tolerances=json.loads(args.tolerances.read_text()) if args.tolerances else None
    result=compare_traces(args.reference,args.candidate,tolerances)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print('PASS' if result['passed'] else 'FAIL',args.output)
    return 0 if result['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
