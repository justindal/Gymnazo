'use strict';
// Observation only: hooks, getter calls, and reads. No input injection,
// replacement callbacks, return-value changes, or game-state writes.
const CONFIG = /*PROBE_CONFIG*/ null;
const BUILD = {"uuid": "4f9aeb54-055f-3708-89f0-6405a08627f9", "sha256": "90f2c073012f990b53e5cd75f08072c25368c0791ec2bd0c199a4ab600f57b4d", "functions": {"position": {"offset": 47790148, "bytes": "fd7bbfa9fd030091006040f9e00000b4"}, "movement": {"offset": 47810952, "bytes": "ff0307d1ef3b136ded33146deb2b156d"}, "jump": {"offset": 47803236, "bytes": "e923bc6df65701a9f44f02a9fd7b03a9"}, "dive": {"offset": 47805740, "bytes": "f44fbea9fd7b01a9fd43009128008052"}, "lane": {"offset": 47808116, "bytes": "eb2bbb6de923016df65702a9f44f03a9"}}};
if (!CONFIG || !BUILD) throw new Error('Load through capture_original.py with the retained build identity');
if (Process.arch !== 'arm64' || Process.platform !== 'darwin') throw new Error('Expected the analyzed iOS arm64 target');
const module = Process.getModuleByName('UnityFramework');
function uuidOf(base) {
  if (base.readU32() !== 0xfeedfacf) throw new Error('Unexpected Mach-O header');
  const n = base.add(16).readU32();
  if (n > 512) throw new Error('Invalid Mach-O command count');
  let offset = 32;
  for (let i = 0; i < n; i++) {
    const command = base.add(offset), size = command.add(4).readU32();
    if (size < 8 || size > 65536) throw new Error('Invalid Mach-O load command');
    if (command.readU32() === 0x1b) {
      const bytes = new Uint8Array(command.add(8).readByteArray(16));
      const hex = Array.from(bytes, x => x.toString(16).padStart(2, '0')).join('');
      return [hex.slice(0,8),hex.slice(8,12),hex.slice(12,16),hex.slice(16,20),hex.slice(20)].join('-');
    }
    offset += size;
  }
  throw new Error('Mach-O UUID absent');
}
if (uuidOf(module.base) !== BUILD.uuid) throw new Error('Different original-game build; offsets cannot be reused');
for (const [name, spec] of Object.entries(BUILD.functions)) {
  const bytes = new Uint8Array(module.base.add(spec.offset).readByteArray(spec.bytes.length / 2));
  const hex = Array.from(bytes, x => x.toString(16).padStart(2, '0')).join('');
  if (hex !== spec.bytes) throw new Error('Function fingerprint mismatch: ' + name);
}
const address = name => module.base.add(BUILD.functions[name].offset);
const position = new NativeFunction(address('position'), ['float','float','float'], ['pointer','pointer']);
const NIL = ptr(0);
let selected = null, stopped = false, rows = [], timer = null;
const listeners = [];
function flush() { if (rows.length) { send({kind:'batch',rows:rows}); rows = []; } }
function fail(error) {
  if (stopped) return;
  rows.push({kind:'error',message:String(error)});
  stop();
}
function samePlayer(motor) {
  if (motor.isNull() || motor.add(464).readU8() !== 1) return false;
  if (selected === null) selected = motor;
  return selected.equals(motor);
}
function runTime(motor) { return motor.add(240).readFloat(); }
send({kind:'header',schema_version:1,source:'original_game',scenario:CONFIG.scenario,
      target_uuid:BUILD.uuid,target_sha256:BUILD.sha256,frida_version:Frida.version,
      coordinate_convention:'Unity world units; lane centers -20/0/20; placer feet',
      sampling:'CharacterMotor.UpdateMovement onLeave; native elapsed run time',
      world_origin_rebase_compensated:false,probe_runtime_validated:false});
listeners.push(Interceptor.attach(address('movement'), {
  onEnter(args) { this.motor = args[0]; },
  onLeave() {
    try {
      if (stopped || !samePlayer(this.motor)) return;
      const p = position(this.motor, NIL);
      const row = {kind:'sample',t:runTime(this.motor),x:p[0],y:p[1],z:p[2],
                   vy:this.motor.add(472).readFloat(),vz:this.motor.add(476).readFloat(),
                   grounded:!!this.motor.add(208).readU8(),flying:!!this.motor.add(489).readU8()};
      if (![row.t,row.x,row.y,row.z,row.vy,row.vz].every(Number.isFinite)) throw new Error('Nonfinite native state');
      rows.push(row);
    } catch (error) { fail(error); }
  }
}));
function observeAction(name, classify) {
  listeners.push(Interceptor.attach(address(name), {
    onEnter(args) {
      try {
        this.action = null;
        if (!stopped && samePlayer(args[0])) {
          this.action = classify(args);
          this.time = runTime(args[0]);
        }
      } catch (error) { fail(error); }
    },
    onLeave(retval) {
      if (this.action && !stopped) rows.push({kind:'action',t:this.time,action:this.action,
                                            accepted:retval.toInt32() !== 0});
    }
  }));
}
observeAction('jump', args => args[1].toInt32() ? 'jump' : null);
observeAction('dive', () => 'down');
observeAction('lane', args => args[1].toInt32() === -1 ? 'left' : args[1].toInt32() === 1 ? 'right' : null);
timer = setInterval(flush, 250);
function stop() {
  if (stopped) return;
  stopped = true;
  clearInterval(timer);
  listeners.forEach(listener => listener.detach());
  flush();
}
rpc.exports = {stop};
