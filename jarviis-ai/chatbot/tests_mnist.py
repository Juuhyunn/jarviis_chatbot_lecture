import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense


class MnistTest(object):
    def __init__(self):
        pass

    def mnist_execute(self):
        # MNIST 데이터셋 가져오기
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0  # 데이터 정규화

        # tf.data를 사용하여 데이터셋을 섞고 배치 만들기
        ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000)
        train_size = int(len(x_train) * 0.7)  # 학습셋:검증셋 = 7:3
        train_ds = ds.take(train_size).batch(20)
        val_ds = ds.skip(train_size).batch(20)

        # MNIST 분류 모델 구성
        model = Sequential()
        model.add(Flatten(input_shape=(28, 28)))
        model.add(Dense(20, activation='relu'))
        model.add(Dense(20, activation='relu'))
        model.add(Dense(10, activation='softmax'))

        # 모델 생성
        model.compile(loss='sparse_categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
        # model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

        # 모델 학습
        hist = model.fit(train_ds, validation_data=val_ds, epochs=10)

        # 모델 평가
        print('모델 평가')
        model.evaluate(x_test, y_test)

        # 모델 정보 출력
        model.summary()

        # 모델 저장
        model.save('mnist_model.h5')

        # 학습 결과 그래프 그리기
        fig, loss_ax = plt.subplots()
        acc_ax = loss_ax.twinx()
        loss_ax.plot(hist.history['loss'], 'y', label='train loss')
        loss_ax.plot(hist.history['val_loss'], 'r', label='val loss')
        acc_ax.plot(hist.history['accuracy'], 'b', label='train acc')
        acc_ax.plot(hist.history['val_accuracy'], 'g', label='val acc')
        loss_ax.set_xlabel('epoch')
        loss_ax.set_ylabel('loss')
        acc_ax.set_ylabel('accuracy')
        loss_ax.legend(loc='upper left')
        acc_ax.legend(loc='lower left')
        plt.show()

    def hand_writing(self):
        # MNIST 데이터셋 가져오기
        _, (x_test, y_test) = mnist.load_data()
        x_test = x_test / 255.0  # 데이터 정규화

        # 모델 불러오기
        model = load_model('mnist_model.h5')
        model.summary()
        model.evaluate(x_test, y_test, verbose=2)

        # 테스트셋에서 20번째 이미지 출력
        plt.imshow(x_test[20], cmap="gray")
        plt.show()

        # 테스트셋의 20번째 이미지 클래스 분류
        picks = [20]
        predict = model.predict_classes(x_test[picks])
        print("손글씨 이미지 예측값 : ", predict)


if __name__ == '__main__':
    m = MnistTest()
    # m.mnist_execute()
    m.hand_writing()