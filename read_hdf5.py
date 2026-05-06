import tensorflow as tf

file_path = '/home/cl-ment-prigent/LAM/ego4d_dataset-train.tfrecord-00493-of-00512'

# Create a dataset object
raw_dataset = tf.data.TFRecordDataset(file_path)

for raw_record in raw_dataset.take(1):
    # You will need to know the features description to parse this properly
    # example_proto = tf.train.Example.FromString(raw_record.numpy())
    print(raw_record)