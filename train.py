import os
from model import *
from load_dataset import load_data
import keras
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
from schedules import onetenth_20_30_40

def plot_history(history, result_dir):
    plt.plot(history.history['acc'], marker='.')
    plt.plot(history.history['val_acc'], marker='.')
    plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.grid()
    plt.legend(['acc', 'val_acc'], loc='lower right')
    plt.savefig(os.path.join(result_dir, 'model_accuracy.png'))
    plt.close()

    plt.plot(history.history['loss'], marker='.')
    plt.plot(history.history['val_loss'], marker='.')
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid()
    plt.legend(['loss', 'val_loss'], loc='upper right')
    plt.savefig(os.path.join(result_dir, 'model_loss.png'))
    plt.close()


def save_history(history, result_dir):
    loss = history.history['loss']
    acc = history.history['acc']
    val_loss = history.history['val_loss']
    val_acc = history.history['val_acc']
    nb_epoch = len(acc)

    with open(os.path.join(result_dir, 'result.txt'), 'w') as fp:
        fp.write('epoch\tloss\tacc\tval_loss\tval_acc\n')
        for i in range(nb_epoch):
            fp.write('{}\t{}\t{}\t{}\t{}\n'.format(
                i, loss[i], acc[i], val_loss[i], val_acc[i]))


if __name__ == '__main__':
    epochs = 50
    nb_classes = 6
    batch_size = 32
    original_size_image_path = '/home/deep/datasets/kth/images/original/'
    crop_size_image_path = '/home/deep/datasets/kth/images/crop/'
    output_path = 'results/'


    print('training with original size image')
    model_ori,_ = model_original_size()
    x_train, y_train, x_val, y_val = load_data(original_size_image_path,'original')
    y_train = keras.utils.to_categorical(y_train,nb_classes)
    y_val = keras.utils.to_categorical(y_val)

    model_ori.summary()
    lr = 0.005
    sgd = keras.optimizers.SGD(lr=lr, nesterov=True, momentum=0.9)
    model_ori.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=sgd,
                      metrics=['accuracy'])
    history = model_ori.fit(x_train,y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  callbacks=[onetenth_20_30_40(lr=lr)],
                  validation_data=(x_val,y_val),
                  shuffle=True)
    if os.path.exists(output_path):
        if not os.path.exists(output_path+'original/'):
            os.mkdir(output_path+'original')
    else:
        os.mkdir(output_path)
        os.mkdir(output_path+'original')
    save_history(history,output_path+'original/')
    plot_history(history,output_path+'original/')
    model_ori.save_weights(output_path+'original/original_size.h5')


    print('training with crop image')
    model_crop,_ = model_crop_size()
    x_train, y_train, x_val, y_val = load_data(crop_size_image_path,'crop')
    y_train = keras.utils.to_categorical(y_train, nb_classes)
    y_val = keras.utils.to_categorical(y_val)

    model_crop.summary()
    lr = 0.005
    sgd = keras.optimizers.SGD(lr=lr, nesterov=True, momentum=0.9)
    model_crop.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=sgd,
                      metrics=['accuracy'])
    history = model_crop.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  callbacks=[onetenth_20_30_40(lr=lr)],
                  validation_data=(x_val, y_val),
                  shuffle=True)
    if os.path.exists(output_path):
        if not os.path.exists(output_path+'crop/'):
            os.mkdir(output_path+'crop')
    else:
        os.mkdir(output_path)
        os.mkdir(output_path+'crop')
    save_history(history, output_path + 'crop/')
    plot_history(history, output_path + 'crop/')
    model_crop.save_weights(output_path + 'crop/crop_size.h5')