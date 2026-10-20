#!/usr/bin/env python3
"""Passive observer for one driver-owned verified-wrapper process. It never launches or signals processes."""
from __future__ import annotations
import json, os, pathlib, subprocess, threading, time


def _stat(pid: int) -> tuple[int, int, int] | None:
    try:
        text=pathlib.Path(f'/proc/{pid}/stat').read_text()
        fields=text[text.rfind(')')+2:].split()
        return int(fields[1]), int(fields[2]), int(fields[19])  # PPID, process group, start ticks
    except (FileNotFoundError, IndexError, ValueError): return None

def _all_stats() -> dict[int, tuple[int,int,int]]:
    out={}
    for path in pathlib.Path('/proc').iterdir():
        if path.name.isdecimal():
            value=_stat(int(path.name))
            if value is not None: out[int(path.name)]=value
    return out

def _lineage(root: int, root_start_ticks: int) -> dict[int, tuple[int,int,int]]:
    all_stats=_all_stats()
    # Reject PID reuse before following PPID descendants.
    if root not in all_stats or all_stats[root][2] != root_start_ticks: return {}
    owned={root}
    changed=True
    while changed:
        changed=False
        for pid,(ppid,pgid,start_ticks) in all_stats.items():
            if ppid in owned and pid not in owned: owned.add(pid); changed=True
    return {pid:all_stats[pid] for pid in sorted(owned) if pid in all_stats}

def _exe(pid: int) -> str | None:
    try: return str(pathlib.Path(f'/proc/{pid}/exe').resolve(strict=True))
    except OSError: return None

def _rss_kib(pid: int) -> int | None:
    try:
        for line in pathlib.Path(f'/proc/{pid}/status').read_text().splitlines():
            if line.startswith('VmRSS:'): return int(line.split()[1])
    except FileNotFoundError: pass
    return None

def _host() -> dict:
    memory={}
    for line in pathlib.Path('/proc/meminfo').read_text().splitlines():
        k,v,*_=line.replace(':','').split()
        if k in {'MemAvailable','Cached','SwapFree','SwapTotal'}: memory[k+'_kib']=int(v)
    pressure={}
    for resource in ('cpu','io','memory'):
        try: pressure[resource]=pathlib.Path(f'/proc/pressure/{resource}').read_text().strip()
        except FileNotFoundError: pass
    return {'meminfo':memory,'pressure':pressure}

def _gpu(owned: set[int]) -> dict:
    """Read-only, low-frequency telemetry. Foreign process identities are discarded."""
    out={'method':'nvidia-smi read-only polling; no CUDA event/copy/kernel profiling'}
    try:
        c=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],
                         text=True,capture_output=True,timeout=3,check=False)
        if c.returncode:
            out['error']=c.stderr.strip() or f'returncode {c.returncode}'; return out
        rows=[]
        for line in c.stdout.splitlines():
            pid,sep,mib=line.partition(',')
            if sep and pid.strip().isdigit() and int(pid.strip()) in owned:
                rows.append({'pid':int(pid.strip()),'used_gpu_memory_mib':mib.strip()})
        out['owned_compute_processes']=rows
        # This is shared-device context only; it cannot attribute activity to the wrapper.
        d=subprocess.run(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used,memory.total','--format=csv,noheader,nounits'],
                         text=True,capture_output=True,timeout=3,check=False)
        if d.returncode==0: out['shared_device_envelope_csv']=d.stdout.strip().splitlines()
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc: out['error']=type(exc).__name__
    return out

class WrapperObserver:
    """Attach to an already-running wrapper PID. Driver keeps all launch/wait/cancellation authority."""
    def __init__(self, output: pathlib.Path, *, poll_seconds: float=2.0, gpu_telemetry: bool=True):
        if poll_seconds < 1.0: raise ValueError('poll_seconds must be >= 1.0')
        self.output=pathlib.Path(output); self.poll_seconds=poll_seconds; self.gpu_telemetry=gpu_telemetry
        self._stop=threading.Event(); self._thread=None; self._samples=[]; self._errors=[]; self._lock=threading.Lock(); self._start=None; self._wrapper=None; self._native=None
    def start(self, wrapper_pid: int, native_exe: pathlib.Path) -> None:
        if self._thread is not None: raise RuntimeError('observer already started')
        wrapper_stat=_stat(wrapper_pid)
        if wrapper_pid <= 0 or wrapper_stat is None: raise ValueError('wrapper PID is not alive')
        self._wrapper=wrapper_pid; self._wrapper_start_ticks=wrapper_stat[2]; self._native=str(pathlib.Path(native_exe).resolve(strict=True)); self._start=time.monotonic()
        self._sample(); self._thread=threading.Thread(target=self._loop,daemon=True,name='strata-wrapper-observer'); self._thread.start()
    def _record_error(self, exc: BaseException) -> None:
        with self._lock:
            self._errors.append({'monotonic_seconds': time.monotonic()-self._start, 'type': type(exc).__name__, 'message': str(exc)})

    def _loop(self) -> None:
        try:
            while not self._stop.wait(self.poll_seconds): self._sample()
        except BaseException as exc:
            self._record_error(exc)
    def _sample(self) -> None:
        lineage=_lineage(self._wrapper, self._wrapper_start_ticks); pids=set(lineage); native=[]
        processes=[]
        for pid,(ppid,pgid,start_ticks) in lineage.items():
            exe=_exe(pid); match=(exe==self._native)
            if match: native.append(pid)
            processes.append({'pid':pid,'ppid':ppid,'process_group':pgid,'start_ticks':start_ticks,'rss_kib':_rss_kib(pid),'exe':exe,'exact_native_executable':match})
        row={'monotonic_seconds':time.monotonic()-self._start,'wrapper_pid':self._wrapper,'wrapper_start_ticks':self._wrapper_start_ticks,'wrapper_process_group':(_stat(self._wrapper) or (None,None,None))[1],
             'owned_lineage_processes':processes,'exact_native_pids':native,'host':_host()}
        if self.gpu_telemetry: row['gpu']=_gpu(pids)
        with self._lock:
            self._samples.append(row)
    def finish(self) -> dict:
        if self._thread is None: raise RuntimeError('observer was not started')
        # A worker can be in two bounded nvidia-smi calls (3s each). Do not
        # take a competing final sample: it would race the worker and make
        # teardown part of the wrapper measurement.
        self._stop.set(); self._thread.join(timeout=self.poll_seconds+8)
        if self._thread.is_alive():
            self._record_error(RuntimeError('observer worker did not stop after bounded sampling wait'))
        end=time.monotonic()-self._start
        with self._lock:
            samples=list(self._samples); errors=list(self._errors)
        seen=[x['monotonic_seconds'] for x in samples if x['exact_native_pids']]
        first, last=(seen[0],seen[-1]) if seen else (None,None)
        envelopes={
          'pre_native_polling_envelope': {'start_monotonic_seconds':0.0,'end_monotonic_seconds':first if first is not None else end,
                                          'status':'observed' if first is not None else 'native_unavailable_exact_executable_never_observed'},
          'native_polling_envelope': ({'first_observed_monotonic_seconds':first,'last_observed_monotonic_seconds':last,'status':'observed'} if first is not None else {'status':'unavailable_exact_executable_never_observed'}),
          'post_native_polling_envelope': ({'start_monotonic_seconds':last,'end_monotonic_seconds':end,'status':'observed'} if first is not None else {'status':'unavailable_exact_executable_never_observed'})}
        result={'schema':'strata-wrapper-observer-v1','scope':'observation only; driver retains launch, wait, and cancellation ownership',
                'wrapper_pid':self._wrapper,'admitted_native_executable':self._native,'poll_seconds':self.poll_seconds,'gpu_telemetry':self.gpu_telemetry,
                'limitations':['Polling envelopes bound observation samples, not actual phase boundaries.','No CUDA events, nsys, memcpy, or kernel timing is collected.','Native envelopes include the native executable process lifetime (mmap/init/traces included), not pure kernel or decoding time.','Only wrapper lineage PIDs are retained; foreign process details are not retained.'],
                'envelopes':envelopes,'samples':samples,'observation_errors':errors,
                'observation_complete':not self._thread.is_alive() and not errors}
        self.output.parent.mkdir(parents=True,exist_ok=True); self.output.write_text(json.dumps(result,indent=2)+'\n')
        if errors or self._thread.is_alive():
            raise RuntimeError('observer incomplete; see observation_errors in '+str(self.output))
        return result
