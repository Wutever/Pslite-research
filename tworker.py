import mxnet

kv_store=mxnet.kv.create('dist_async')
model.fit(train_iter, eval_iter,
            optimizer_params={
                'learning_rate':0.005, 'momentum': 0.9},
            num_epoch=50,
            eval_metric='mse',
            batch_end_callback
                 = mx.callback.Speedometer(batch_size, 2),
            kvstore=kv_store)