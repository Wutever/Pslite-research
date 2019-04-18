import subprocess
import os
server_env = os.environ.copy()
server_env.update({
 "DMLC_ROLE": "server",
 "DMLC_PS_ROOT_URI": "127.0.0.1",
 "DMLC_PS_ROOT_PORT": "9000",
 "DMLC_NUM_SERVER": "1",
 "DMLC_NUM_WORKER": "1",
 "PS_VERBOSE": "2"
 })
subprocess.Popen("python -c 'import mxnet'", shell=True, env=server_env)