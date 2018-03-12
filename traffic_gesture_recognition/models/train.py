# -*- coding: utf-8 -*-
import os
import click
# import glob
from .inception_v4 import create_inception_v4, load_weights, Dense
import tensorflow as tf


def get_label(file_path, labels):
    return labels.index(os.path.basename(os.path.dirname(file_path)))


def load_model(nb_classes, weights=None, freeze_until=None):
    model = create_inception_v4()
    model = tf.keras.models.Model(model.inputs, model.get_layer("logits").output)
    load_weights(model, weights)
    if freeze_until:
        for layer in model.layers[:model.layers.index(model.get_layer(freeze_until))]:
            layer.trainable = False
    out = Dense(units=nb_classes, activation='softmax')(model.output)
    model = tf.keras.models.Model(model.inputs, out)
    return model


def create_data_generator(data_path):
    labels = [
        "0",
        "1"
    ]

    train_generator = tf.keras.preprocessing.image.ImageDataGenerator(samplewise_center=True, rescale=1. / 128)
    train_gen = train_generator.flow_from_directory(
        os.path.join(data_path, "train"),
        target_size=(299, 299),
        classes=labels,
        shuffle=True
    )

    valid_generator = tf.keras.preprocessing.image.ImageDataGenerator(samplewise_center=True, rescale=1. / 128)
    valid_gen = valid_generator.flow_from_directory(
        os.path.join(data_path, "valid"),
        target_size=(299, 299),
        classes=labels,
        shuffle=True,
        batch_size=32
    )
    print(train_gen.class_indices)
    return train_gen, valid_gen


@click.command()
@click.argument("data_path", type=click.Path(exists=True))
@click.argument("save_path", type=click.Path())
@click.option("-e", "--epochs", type=click.INT)
@click.option("-w", "--weight_path", type=click.Path(exists=True))
@click.option("-f", "--freeze_until")
def main(data_path, save_path, weight_path=None, epochs=10, freeze_until=None):
    train_gen, valid_gen = create_data_generator(data_path)

    model = load_model(nb_classes=len(train_gen.class_indices), weights=weight_path, freeze_until=freeze_until)
    model.compile(
        loss="categorical_crossentropy",
        optimizer=tf.keras.optimizers.RMSprop(lr=0.005, decay=0.01),
        metrics=["accuracy"]
    )
    model.fit_generator(
        train_gen,
        validation_data=valid_gen,
        epochs=epochs,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(os.path.join(save_path, "weights.{epoch:02d}-{val_loss:.2f}.hdf5"), monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1),

        ]
    )
    model.save("model.hdf5")


if __name__ == "__main__":
    main()
