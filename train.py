import tensorflow as tf
import datetime
import yaml

params = yaml.safe_load(open('params.yaml'))
epochs = params['epochs']
log_file = params['log_file']

mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

def create_model():
  return tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
  ])

model = create_model()
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

csv_logger = tf.keras.callbacks.CSVLogger(log_file)

model.fit(x=x_train,
          y=y_train,
          epochs=epochs,
          validation_data=(x_test, y_test),
          callbacks=[csv_logger])
          #callbacks=[tensorboard_callback])

