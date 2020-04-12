import keras.backend as K
from keras.optimizers import RMSprop
from keras import callbacks
import matplotlib.pyplot as plt
import os


def trainer_function(model, X_train, Y_train, X_test, Y_test, num_epocs=2000, logs_dir='tmp'):
    def frobenius_norm(y_pred, y_true):
        y = K.square(y_pred - y_true)
        res = K.pow(K.sum(y, -1), 0.5)
        return res

    def leakage(y_pred, y_true):
        pred_sum = K.sum(K.abs(y_true - y_pred), -1)
        actual_sum = K.sum(y_true, -1)
        res = pred_sum / actual_sum
        return res * 100

    #     optimizer = Nadam()
    #     optimizer = Adam()
    optimizer = RMSprop()
    model.compile(loss=frobenius_norm,
                  optimizer=optimizer,
                  metrics=[leakage, 'mse'])
    #               metrics=[ frobenius_norm ])

    os.makedirs(logs_dir, exist_ok=True)
    history = model.fit(X_train, Y_train, epochs=num_epocs, verbose=2,
                        batch_size=32, validation_data=(X_test, Y_test),
                        callbacks=[  # callbacks.TensorBoard(log_dir=logs_dir),
                            callbacks.CSVLogger(logs_dir + '/training.log')
                            #                         callbacks.EarlyStopping(patience=50)
                        ])

    model.save(logs_dir + "/model.h5")
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    #     plt.title('model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper right')
    plt.savefig(logs_dir + "/loss_train_tets")
    plt.clf()

    plt.plot(history.history['val_loss'])
    #     plt.title('model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Test'], loc='upper right')
    plt.savefig(logs_dir + "/loss_tets")
    plt.clf()

    plt.plot(history.history['loss'])
    #     plt.title('model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train'], loc='upper right')
    plt.savefig(logs_dir + "/loss_train")
    plt.clf()
