from libero.libero import benchmark,get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from pathlib import Path
suite=benchmark.get_benchmark_dict()['libero_spatial'](); task=suite.get_task(0)
env=OffScreenRenderEnv(bddl_file_name=str(Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file),camera_heights=128,camera_widths=128)
env.reset(); obs=env.set_init_state(suite.get_task_init_states(0)[0]); print('PASS render',obs['agentview_image'].shape, task.language); env.close()
