"""Shared four-suite evaluator for the retained JAX expert2k experiments."""
import datetime,json,os,socket,statistics,subprocess,sys,time,fcntl
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
ART=ROOT/'artifacts/jax_2k'; LOG=ROOT/'logs'; ART.mkdir(parents=True,exist_ok=True)
PY=ROOT/'vendor/openpi_official/.venv/bin/python'; SIM=ROOT/'.venv/bin/python'
ENV=os.environ.copy()
SUITES=['libero_goal','libero_spatial','libero_object','libero_10']

def evaluate(checkpoint,label,smoke=False):
    servers=[]; clients=[]; handles=[]
    out=ART/'eval'/label; out.mkdir(parents=True,exist_ok=False)
    try:
        for i,suite in enumerate(SUITES[:1] if smoke else SUITES):
            gpu=4+i;port=8104+i;e=ENV.copy();e['CUDA_VISIBLE_DEVICES']=str(gpu);e['XLA_PYTHON_CLIENT_MEM_FRACTION']='0.55'
            f=(LOG/f'openpi_{label}_{suite}_server.log').open('w');handles.append(f)
            server=subprocess.Popen([str(PY),'scripts/openpi_serve.py','--checkpoint',str(checkpoint),'--port',str(port)],cwd=ROOT,env=e,stdout=f,stderr=subprocess.STDOUT);servers.append(server)
        deadline=time.monotonic()+1200
        for i,server in enumerate(servers):
            while True:
                if server.poll() is not None:raise RuntimeError('Policy server exited; inspect server log')
                try:
                    with socket.create_connection(('127.0.0.1',8104+i),timeout=5) as conn:
                        conn.sendall(b'GET /healthz HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n')
                        if b'200 OK' not in conn.recv(1024):raise OSError('Policy health check not ready')
                        break
                except OSError:
                    if time.monotonic()>deadline:raise TimeoutError('Policy startup timeout')
                    time.sleep(5)
        for i,suite in enumerate(SUITES[:1] if smoke else SUITES):
            e=ENV.copy();e.update(CUDA_VISIBLE_DEVICES=str(4+i),MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',MUJOCO_EGL_DEVICE_ID=str(4+i),LIBERO_CONFIG_PATH=str(ROOT/'configs/openpi_libero'),PYTHONPATH=str(ROOT/'vendor/openpi_official/third_party/libero')+':'+str(ROOT/'vendor/openpi_official/packages/openpi-client/src'))
            args=[str(SIM),'scripts/openpi_eval.py','--suite',suite,'--port',str(8104+i),'--output',str(out/f'{suite}.jsonl')]
            if smoke:args+=['--rollouts','1','--task-ids','0']
            f=(LOG/f'openpi_{label}_{suite}_client.log').open('w');handles.append(f)
            clients.append(subprocess.Popen(args,cwd=ROOT,env=e,stdout=f,stderr=subprocess.STDOUT))
        while any(p.poll() is None for p in clients):
            if any(p.poll() not in (None,0) for p in clients):raise RuntimeError('Evaluation client failed; inspect client logs')
            if any(p.poll() is not None for p in servers):raise RuntimeError('Policy server failed during eval')
            time.sleep(10)
        if any(p.returncode for p in clients):raise RuntimeError('Evaluation failed')
        summary={}
        for suite in SUITES[:1] if smoke else SUITES:
            rows=[json.loads(x) for x in (out/f'{suite}.jsonl').read_text().splitlines()]
            assert len(rows)==(1 if smoke else 500)
            summary[suite]=dict(episodes=len(rows),successes=sum(r['success'] for r in rows),success_rate=sum(r['success'] for r in rows)/len(rows))
        summary['average_success_rate']=statistics.mean(v['success_rate'] for v in summary.values())
        (out/'summary.json').write_text(json.dumps(summary,indent=2))
    finally:
        for p in clients+servers:
            if p.poll() is None:p.terminate()
        for p in clients+servers:
            try:p.wait(timeout=30)
            except subprocess.TimeoutExpired:p.kill();p.wait()
        for f in handles:f.close()
