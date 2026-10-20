import argparse
import hashlib
import json
from pathlib import Path


def node(value):
    behaviours = value['behaviours']
    spawn = behaviours.get('SpawnLocation')
    override = (spawn or {}).get('_spawnerOverride', {})
    lane = behaviours.get('RandomLane')
    return dict(
        name=value['name'], active=bool(value['active']),
        position=value['p'], rotation=value['q'], scale=value['s'],
        asset=behaviours.get('ChunkAssetPlacer', {}).get('_assetType', {}).get('name', ''),
        placer='ChunkAssetPlacer' in behaviours,
        lanes=None if lane is None else [x for k, x in [('_left', -20), ('_middle', 0), ('_right', 20)] if lane.get(k, True)],
        mirror=behaviours.get('MirrorRandomizer', {}).get('_mirrorProbability'),
        randomChild=behaviours.get('RandomChildRandomizer', {}).get('_activationProbability'),
        movingSpeed=behaviours.get('MovingTrain', {}).get('_speed', 1),
        spawn=spawn is not None, coin=override.get('_pickupType') == 1,
        coinCount=max(1, int(override.get('_baseNumItems', 1))),
        colliders=[dict(center=c['center'], size=c['size']) for c in value.get('colliders', [])],
        children=[node(c) for c in value['children']],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    routes = json.loads((args.source / 'routes.json').read_text())
    schedule = json.loads((args.source / 'route_schedule.json').read_text())
    root = schedule['schedulers'][schedule['routes'][schedule['default_route']]['_scheduler']['name']]
    sections = {}
    for name in schedule['core_section_names']:
        section = schedule['sections'][name]
        sections[name] = dict(kind=section['class'],
                              chunks=[c['name'] for c in section.get('Chunks', [])],
                              sections=[s['name'] for s in section.get('Sections', [])])
    result = dict(
        schemaVersion=1,
        source='Metro_Run; Subway Surfers iOS 3.70.1 build 96703 configuration export',
        sourceSHA256={f: hashlib.sha256((args.source / f).read_bytes()).hexdigest()
                      for f in ['routes.json', 'route_schedule.json']},
        chunks={k: dict(length=max(180, c['exit'][2] * 11.25), root=node(c['root']))
                for k, c in routes['chunks'].items()},
        sections=sections,
        entries=[dict(section=e['Section']['name'], weight=e['Weight'], minimum=e['MinimumZ'],
                      maximum=e['MaximumZ'], priority=e['Priority'], unique=bool(e['IsUnique']))
                 for e in root['_entries']],
        intro=root['_introSection']['name'], fallback=root['_fallbackSection']['name'],
        repeatDistance=root['_minimumRepeatDistance'],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, separators=(',', ':'), sort_keys=True) + '\n')


if __name__ == '__main__':
    main()
