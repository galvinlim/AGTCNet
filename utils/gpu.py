import tensorflow as tf

def gpu_allocation(gpu_memory_limit):
    gpus = tf.config.list_physical_devices('GPU')
    # tf.config.set_visible_devices(gpus[:1], device_type='GPU')
    # tf.config.experimental.set_memory_growth(gpus[0], True)
    log_dev_conf = tf.config.LogicalDeviceConfiguration(memory_limit=gpu_memory_limit*1024) # GB
    tf.config.set_logical_device_configuration(gpus[0], [log_dev_conf])
    print('GPU Limitation: ', gpu_memory_limit, ' GB')