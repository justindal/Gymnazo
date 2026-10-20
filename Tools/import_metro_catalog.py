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
        spawner=override.get('name'), pickupType=override.get('_pickupType'),
        spawnGroup=(spawn or {}).get('_spawnGroup', 0),
        disableOnFlyover=bool((spawn or {}).get('_alwaysDisableWhenFlyingOver', False)),
        preserveFlyoverDistance=(spawn or {}).get('_preserveWhenFlyoverDistance', 600),
        colliders=[dict(center=c['center'], size=c['size']) for c in value.get('colliders', [])],
        children=[node(c) for c in value['children']],
    )


def curve(value):
    keys = value['m_Curve']
    if not keys or any(key.get('weightedMode', 0) for key in keys):
        raise ValueError('Only nonempty unweighted AnimationCurves are supported')
    return dict(keys=[{key: value[key] for key in ('time', 'value', 'inSlope', 'outSlope')}
                      for value in keys])


def rules(data):
    objects = {item['name']: item['fields'] for item in data['objects']}
    names = dict(magnet='MagnetPowerConfig', sneakers='SuperSneakersPowerConfig',
                 multiplier='DoubleScorePowerConfig', jetpack='JetpackPowerConfig')
    kinds = dict(Key='key', Magnet='magnet', Sneakers='sneakers', Jetpack='jetpack',
                 MysteryBox='mystery', DoubleScore='multiplier')
    groups = dict(None_=0, Easy=1, Medium=2, Special=3)
    def entry(value):
        return dict(kind=kinds[value['pickupType']], weight=value['weight'],
                    groups=[groups[group] for group in value.get('groups', [])],
                    minimumDistance=value.get('minDistanceToPrevious', 0),
                    requirements=value.get('requirements'))
    spawn = data['tower']['spawnchances']['data']
    normal = spawn['spawnSets']['spawnSet_Default']
    jet = objects['JetpackPowerEffect']
    jet_coins = objects['JetpackCoinSpawner']
    pogo = objects['PogostickPowerConfig']
    pogo_coins = objects['PogostickCoinSpawner']
    pogo_reward = objects['PogostickPowerupSpawner']
    return dict(
        durations={kind: [level['Duration'] for level in objects[name]['Levels']]
                   for kind, name in names.items()},
        sneakersCurve=curve(objects['ability_HighJump']['_jumpCurveOverride']['Value']),
        spawnSpacing=spawn['spawnGroups'][normal['groupId']]['minDistanceToPrevious'],
        spawnEntries=[entry(value) for value in normal['spawnEntries']
                      if value.get('spawnLogic', 'Normal') == 'Normal'],
        pogoEntries=[entry(value) for value in spawn['pogostickSpawnEntries']['default']['entries']],
        jetpack=dict(height=jet['_targetHeight'], speedFactor=jet['_flightSpeedMultiplier'],
                     adjustmentTime=jet['_heightAdjustmentTime'], coinOffset=jet['_coinZStartOffset'],
                     coinSpacing=jet_coins['_distanceBetweenCoins'], coinHeight=jet_coins['_coinYOffset'],
                     coinsPerNode=jet_coins['_coinsBetweenPathNodes'],
                     laneDuration=objects['ability_JetpackLaneChanging']['_laneChangingDuration']['Value']),
        pogo=dict(height=pogo['_jumpHeight'], distance=pogo['_jumpDistance'],
                  curve=curve(pogo['_jumpHeightCurve']), rows=pogo_coins['_coinRows'],
                  coinHeight=pogo_coins['_coinYOffset'], coinStart=pogo_coins['_coinZStartOffset'],
                  coinEnd=pogo_coins['_coinZEndOffset'], rewardHeight=pogo_reward['powerupYOffset'],
                  rewardOffset=pogo_reward['_powerupZOffset']),
        curves={item['name']: dict(curve=curve(item['fields']['_animationCurve']),
                                  zScale=item['fields']['_zScale'], yScale=item['fields']['_yScale'],
                                  fixedOffset=item['fields']['_fixedLengthOffset'])
                for item in data['objects'] if item['class'] == 'Curve'},
        coinSpawners={item['name']: dict(curve=item['fields']['_curve']['name'],
                                       baseCount=item['fields']['_baseNumItems'],
                                       scalesWithSpeed=bool(item['fields']['_scaleWithSpeed']),
                                       pivot=item['fields']['_scaleT'], inAir=bool(item['fields']['_isInAir']))
                      for item in data['objects']
                      if item['class'] == 'CurveSpawner' and item['fields']['_pickupType'] == 1},
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    routes = json.loads((args.source / 'routes.json').read_text())
    schedule = json.loads((args.source / 'route_schedule.json').read_text())
    gameplay = json.loads((args.source / 'gameplay_config.json').read_text())
    root = schedule['schedulers'][schedule['routes'][schedule['default_route']]['_scheduler']['name']]
    sections = {}
    for name in schedule['core_section_names']:
        section = schedule['sections'][name]
        sections[name] = dict(kind=section['class'],
                              chunks=[c['name'] for c in section.get('Chunks', [])],
                              sections=[s['name'] for s in section.get('Sections', [])])
    result = dict(
        schemaVersion=2,
        source='Metro_Run; Subway Surfers iOS 3.70.1 build 96703 configuration export',
        sourceSHA256={f: hashlib.sha256((args.source / f).read_bytes()).hexdigest()
                      for f in ['routes.json', 'route_schedule.json', 'gameplay_config.json', 'train_geometry.json']},
        rules=rules(gameplay),
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
    args.output.write_text(json.dumps(result, separators=(',', ':'), sort_keys=True, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
