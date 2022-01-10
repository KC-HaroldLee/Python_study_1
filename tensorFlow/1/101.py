import tensorflow as tf
(mnist_x, mnist_y), _ = tf.keras.datasets.mnist.load_data()
print(mnist_x.shape, mnist_y.shape)

(cifar_x, cifar_y), _ = tf.keras.datasets.cifar10.load_data()
print(cifar_x.shape, cifar_y.shape)

# mnist 출력하기
import matplotlib.pyplot as plt
print(mnist_y[0]) # x의 레이블 값이 될것이다.
plt.imshow(mnist_x[0], cmap='gray') # 채널이 하나니까

# cifar 출력하기
print(cifar_y[0])
plt.imshow(cifar_x[0]) # gray를 써도 되긴한다.

