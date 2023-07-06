import tensorflow as tf

# List available CPUs
cpus = tf.config.list_physical_devices('CPU')
print("Available CPUs:")
for cpu in cpus:
    print(cpu)

# List available GPUs
gpus = tf.config.list_physical_devices('GPU')
print("Available GPUs:")
for gpu in gpus:
    print(gpu)