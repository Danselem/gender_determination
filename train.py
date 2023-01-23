import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.preprocessing.image import ImageDataGenerator


lr = 0.001
size = 200
droprate = 0.0
input_size = 224

# Splitting data into train and val sets with batch size of 30
batch_size = 30

def generators(shape, preprocessing): 
    '''Create the training and validation datasets for 
    a given image shape.
    '''
    imgdatagen = ImageDataGenerator(
        preprocessing_function = preprocessing,
        horizontal_flip = True,
        shear_range=10,
        zoom_range=0.1,
        rotation_range=20,
        validation_split = 0.2,
    )

    height, width = shape

    train_dataset = imgdatagen.flow_from_directory(
        'gender_eye/train',
        target_size = (height, width), 
        classes = ('female','male'),
        batch_size = batch_size,
        subset = 'training', 
    )

    val_dataset = imgdatagen.flow_from_directory(
        'gender_eye/train',
        target_size = (height, width), 
        classes = ('female','male'),
        batch_size = batch_size,
        subset = 'validation',
        shuffle=False
    )
    return train_dataset, val_dataset


def make_model(input_size=150, learning_rate=0.001, size_inner=200, droprate=0.5):
    conv_model = resnet50.ResNet50(weights='imagenet', include_top=False, input_shape=(input_size, input_size, 3))
    for layer in conv_model.layers:
        layer.trainable = False
    x = keras.layers.Flatten()(conv_model.output)
    x = keras.layers.Dense(300, activation='relu')(x)
    x = keras.layers.Dense(size_inner, activation='relu')(x)
    x = keras.layers.Dense(50, activation='relu')(x)
    drop = keras.layers.Dropout(droprate)(x)
    predictions = keras.layers.Dense(2, activation='softmax')(drop)
    model = keras.models.Model(inputs=conv_model.input, outputs=predictions)
    model.compile(loss='binary_crossentropy',
                  optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                  metrics=['accuracy', 'Precision','Recall', 'AUC',])
    
    return model


resnet50 = keras.applications.resnet50
train_ds, val_ds = generators((input_size, input_size), preprocessing=resnet50.preprocess_input)

checkpoint = keras.callbacks.ModelCheckpoint(
    'resnet_v1_{epoch:02d}_{val_accuracy:.3f}.h5',
    save_best_only=True,
    monitor='val_accuracy',
    mode='max'
)


model = make_model(input_size=input_size,
                   learning_rate=lr,
                   size_inner=size,
                   droprate=droprate)


history = model.fit(
    train_ds, 
    validation_data = val_ds,
    workers=10,
    epochs=30,
    steps_per_epoch=50,
    validation_steps=5,
    callbacks=[checkpoint])


model_name = 'resnet_v1_02_0.973.h5'
model_best = keras.models.load_model(model_name)


# Convert the model.
converter = tf.lite.TFLiteConverter.from_keras_model(model_best)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Save the model.
with open('eye_model.tflite', 'wb') as f:
    f.write(tflite_model)