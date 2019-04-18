import subprocess
import os
import mxnet
os.environ.update({
 "DMLC_ROLE": "worker",
 "DMLC_PS_ROOT_URI": "127.0.0.1",
 "DMLC_PS_ROOT_PORT": "9000",
 "DMLC_NUM_SERVER": "1",
 "DMLC_NUM_WORKER": "1",
 "PS_VERBOSE": "2"
 })

worker_env = os.environ.copy() 
kv_store = mxnet.kv.create('dist_async')
model.fit(train_iter, eval_iter,
            optimizer_params={
                'learning_rate':0.005, 'momentum': 0.9},
            num_epoch=50,
            eval_metric='mse',
            batch_end_callback
                 = mx.callback.Speedometer(batch_size, 2),
            kvstore=kv_store)