
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Reshape, Dense, Dropout, UpSampling2D, Flatten
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Activation, BatchNormalization, Dense, Dropout, Flatten, Reshape
from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras import initializers
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from IPython.display import clear_output 


tf.config.list_physical_devices('GPU')


# tf.debugging.set_log_device_placement(True)

#생성자가 생성할 이미지의 세로
img_rows = 256
#생성자가 생성할 이미지의 가로
img_cols = 256
#생성자가 생성할 이미지 채널 (컬러이미지)
channels = 3

# 판별자가 판별할 이미지의 세로 가로 채널
img_shape = (img_rows, img_cols, channels)

# 생성자가 이미지를 생성할 초기 노이즈 개수
z_dim = 256



# z_dim : 256 
def build_generator(z_dim):
    #생성자 객체
    model = Sequential()


    # z_dim(256 개의 난수) 입력 받아서 256 * 64 * 64 번 선형 회귀 
    model.add(Dense(256 * 64 * 64, input_dim=z_dim))
    # 256 * 64 * 64 번 선형회귀 결과를 512, 512, 256 의 3차원 배열로 변환
    model.add(Reshape((64, 64, 256)))

    # Conv2DTranspose 객체를 이용해서 64,64,256 배열을 64,64,512 배열로 변환
    model.add(Conv2DTranspose(512, kernel_size=3, strides=1, padding='same'))

    # Batch normalization
    model.add(BatchNormalization())

    # Leaky ReLU activation
    model.add(LeakyReLU(alpha=0.01))
    
#     # Conv2DTranspose 객체를 이용해서 64,64,512 배열을 64,64,1024 배열로 변환
#     model.add(Conv2DTranspose(1024, kernel_size=3, strides=1, padding='same'))
    
#     model.add(Dropout(0.5))

#     # Batch normalization
#     model.add(BatchNormalization())

#     # Leaky ReLU activation
#     model.add(LeakyReLU(alpha=0.01))

#     #  Conv2DTranspose 객체를 이용해서 64,64,1024 배열을 128,128,1024 
#     model.add(Conv2DTranspose(1024, kernel_size=3, strides=2, padding='same'))
    
#     model.add(Dropout(0.5))

#     # Batch normalization
#     model.add(BatchNormalization())

#     # Leaky ReLU activation
#     model.add(LeakyReLU(alpha=0.01))
    
    #  Conv2DTranspose 객체를 이용해서 128,128,1024 배열을 128,128,512 
    model.add(Conv2DTranspose(512, kernel_size=3, strides=2, padding='same'))
    
#     model.add(Dropout(0.5))

    # Batch normalization
    model.add(BatchNormalization())

    # Leaky ReLU activation
    model.add(LeakyReLU(alpha=0.01))
    
    #  Conv2DTranspose 객체를 이용해서 128,128,512 배열을 128,128,256 
    model.add(Conv2DTranspose(256, kernel_size=3, strides=1, padding='same'))

    # Batch normalization
    model.add(BatchNormalization())

    # Leaky ReLU activation
    model.add(LeakyReLU(alpha=0.01))
    
    #  Conv2DTranspose 객체를 이용해서 128,128,256 배열을 128,128,128 
    model.add(Conv2DTranspose(128, kernel_size=3, strides=1, padding='same'))

    # Batch normalization
    model.add(BatchNormalization())

    # Leaky ReLU activation
    model.add(LeakyReLU(alpha=0.01))

      #  Conv2DTranspose 객체를 이용해서 128,128,128 배열을 128,128,64 로 변환
    model.add(Conv2DTranspose(64, kernel_size=3, strides=1, padding='same'))

    # Batch normalization
    model.add(BatchNormalization())

    # Leaky ReLU activation
    model.add(LeakyReLU(alpha=0.01))

    # Conv2DTranspose 객체를 이용해서 128,128, 64 을 256 ,256, 3 으로 변환
    model.add(Conv2DTranspose(3, kernel_size=3, strides=2, padding='same'))

    # Output layer with tanh activation (-1~1 사이 값 리턴)
    model.add(Activation('tanh'))

    return model

# img_shape : 실제 이미지 인지 (1로 판별) 생성자가 생성한 가짜 이미지 인지 (0으로 판별) 판별 할 이미지의 세로,가로,채널
def build_discriminator(img_shape):
    #판별자 모델
    model = Sequential()

    # CNN 필터 학습 256,256,3 배열을 128x128x32 배열로 변환
    model.add(
        Conv2D(32,
               kernel_size=3,
               strides=1,
               input_shape=img_shape,
               padding='same'))
    # Batch normalization
    model.add(BatchNormalization())

    # Leaky ReLU activation
    model.add(LeakyReLU(alpha=0.01))
    # CNN 필터 학습 128x128x3 배열을 128x128x32 배열로 변환
    model.add(
        Conv2D(32,
               kernel_size=3,
               strides=2,
               input_shape=img_shape,
               padding='same'))

    # Batch normalization
    model.add(BatchNormalization())

    # Leaky ReLU activation
    model.add(LeakyReLU(alpha=0.01))
    # Convolutional layer, from 128x128x32 into 64x64x64 tensor
    model.add(
        Conv2D(64,
               kernel_size=3,
               strides=1,
               input_shape=img_shape,
               padding='same'))
    
    model.add(LeakyReLU(alpha=0.01))
    
    # Convolutional layer, from 128x128x32 into 64x64x64 tensor
    model.add(
        Conv2D(64,
               kernel_size=3,
               strides=2,
               input_shape=img_shape,
               padding='same'))

    # Batch normalization
    model.add(BatchNormalization())

    # Leaky ReLU activation
    model.add(LeakyReLU(alpha=0.01))

    # Convolutional layer, from 64x64x64 tensor into 32x32x128 tensor
    model.add(
        Conv2D(128,
               kernel_size=3,
               strides=1,
               input_shape=img_shape,
               padding='same'))

    # Batch normalization
    model.add(BatchNormalization())

    # Leaky ReLU activation
    model.add(LeakyReLU(alpha=0.01))
    # Convolutional layer, from 64x64x64 tensor into 32x32x128 tensor
    model.add(
        Conv2D(128,
               kernel_size=3,
               strides=2,
               input_shape=img_shape,
               padding='same'))
    
    # Batch normalization
    model.add(BatchNormalization())

    # Leaky ReLU activation
    model.add(LeakyReLU(alpha=0.01))
    
    # Convolutional layer, from 64x64x64 tensor into 32x32x128 tensor
    model.add(
        Conv2D(256,
               kernel_size=3,
               strides=1,
               input_shape=img_shape,
               padding='same'))
    
    # Batch normalization
    model.add(BatchNormalization())

    # Leaky ReLU activation
    model.add(LeakyReLU(alpha=0.01))
    
    # Convolutional layer, from 64x64x64 tensor into 32x32x128 tensor
    model.add(
        Conv2D(512,
               kernel_size=3,
               strides=1,
               input_shape=img_shape,
               padding='same'))

    # Batch normalization
    model.add(BatchNormalization())

    # Leaky ReLU activation
    model.add(LeakyReLU(alpha=0.01))

    # Output layer with sigmoid activation
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))

    return model

# Gan 객체 생성
def build_gan(generator, discriminator):
    #Gan 객체
    model = Sequential()

    # Combined Generator -> Discriminator model
    model.add(generator)
    model.add(discriminator)

    return model



# # Build and compile the Discriminator
# d = build_discriminator(img_shape)

#Discriminator (판별자) 의 학습 설정
# binary_crossentropy : 0 (가짜 이미지), 1 (진짜 이미지 판별)
# d.compile(loss='binary_crossentropy',
#                       optimizer=Adam(learning_rate=1e-4),
#                       metrics=['accuracy'])

# g = build_generator(z_dim)

# # Keep Discriminator’s parameters constant for Generator training
# d.trainable = False

# # Build and compile GAN model with fixed Discriminator to train the Generator
# # d_on_g = build_gan(g, d)
# #생성자가 생성한 이미지를 판별자에게 넘겨 주고 학습을 진행
# d_on_g.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=1e-4))

#cost를 저장할 리스트
losses = []
#판별자의 정확도를 저장할 리스트
accuracies = []

iteration_checkpoints = []

model_save_path = "/tf/notebooks/anogan/anogan with handle data/h5/"

def train(iterations, batch_size, sample_interval, X_train):
    d = build_discriminator(img_shape)
    d.load_weights(model_save_path + "anogan_discriminator.h5")
    d.compile(loss='binary_crossentropy',
                      optimizer=Adam(learning_rate=1e-4),
                      metrics=['accuracy'])
    print("#### discriminator ######")
    d.summary()
    g = build_generator(z_dim)
    g.load_weights(model_save_path + "anogan_generator.h5")
    d.trainable = False
    print("#### generator ######")
    g.summary()
    d_on_g = build_gan(g, d)
    d_on_g.load_weights(model_save_path + "anogan.h5")
    d_on_g.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=1e-4))

    
    y = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))
    
    for iteration in tqdm(range(iterations)):

        idx = np.random.randint(0, X_train.shape[0], batch_size)
        #X_train에서 idx 번째 이미지들 선택해서 images에 저장 (X_train에서 batch)
        X = X_train[idx]
        z = np.random.normal(0, 1, (batch_size, 256))
        generated_images = g.predict(z)

        d_loss_real = d.train_on_batch(X, y)
        d_loss_fake = d.train_on_batch(generated_images, fake)
        d_loss, accuracy = 0.5 * np.add(d_loss_real, d_loss_fake) 
            
        z = np.random.normal(0, 1, (batch_size, 256))        
        generated_images = g.predict(z)
        
        d.trainable = False
        g_loss = d_on_g.train_on_batch(z, y)

        
        if (iteration==0) or ((iteration + 1) % sample_interval == 0):

            # Save losses and accuracies so they can be plotted after training
            losses.append((d_loss, g_loss))
            accuracies.append(100.0 * accuracy)
            iteration_checkpoints.append(iteration + 1)

            # Output training progress
            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" %
                  (iteration + 1, d_loss, 100.0 * accuracy, g_loss))

            # Output a sample of generated image
            sample_images(g, iteration )
            

  
        if ((iteration + 1) % (10 * sample_interval) == 0):
        
            g.save('assets/generator', True)
            d.save('assets/discriminator', True)

            d.save(model_save_path + "anogan_discriminator.h5")

            g.save(model_save_path + "anogan_generator.h5")

            d_on_g.save(model_save_path + "anogan.h5")
            
        if ((iteration + 1) % (2 * sample_interval) == 0):
            
            clear_output(wait = True)
            
    return d, g



generate_save_path = '/tf/notebooks/anogan/anogan with handle data/generate_img/'

def sample_images(g,epoch, image_grid_rows=5, image_grid_columns=8):

    # Sample random noise
    z = np.random.normal(0, 1, (image_grid_rows * image_grid_columns, z_dim))

    # Generate images from random noise
    gen_imgs = g.predict(z)

    # Rescale image pixel values to [0, 1]
    gen_imgs = 0.5 * gen_imgs + 0.5

    # Set image grid
    fig, axs = plt.subplots(image_grid_rows,
                            image_grid_columns,
                            figsize=(50, 30),
                            sharey=True,
                            sharex=True)
    axs.reshape(image_grid_rows, image_grid_columns)

    cnt = 0
    for i in range(image_grid_rows):
        for j in range(image_grid_columns):
            # Output a grid of images
            axs[i][j].imshow(gen_imgs[i+j])
            axs[i][j].axis('off')
            cnt += 1
    #생성 이미지 저장
    plt.savefig(generate_save_path+"anogan_generate{:05n}.jpg".format(epoch))

# def sample_images(g,epoch, image_grid_rows=1, image_grid_columns=3):

#     # Sample random noise
#     z = np.random.normal(0, 1, (image_grid_rows * image_grid_columns, z_dim))

#     # Generate images from random noise
#     gen_imgs = g.predict(z)

#     # Rescale image pixel values to [0, 1]
#     gen_imgs = 0.5 * gen_imgs + 0.5

#     # Set image grid
#     fig, axs = plt.subplots(image_grid_rows,
#                             image_grid_columns,
#                             figsize=(50, 30),
#                             sharey=True,
#                             sharex=True)
#     axs.reshape(image_grid_rows, image_grid_columns)

#     cnt = 0
#     for i in range(image_grid_rows):
#         for j in range(image_grid_columns):
#             # Output a grid of images
#             axs[cnt].imshow(gen_imgs[cnt])
#             axs[cnt].axis('off')
#             cnt += 1
#     #생성 이미지 저장
#     plt.savefig(generate_save_path+"anogan_generate{:05n}.jpg".format(epoch))
# #     plt.show()

anogan_generate_path = '/tf/notebooks/anogan/anogan with handle data/anogan_generate/'

def generate(BATCH_SIZE):
    g = build_generator(z_dim)
    g.load_weights(model_save_path + "anogan_generat.h5")
   
    for batch in tqdm(range(BATCH_SIZE)):
        if batch is not 0:

            sample(g, batch)

    return generated_images

def sample(g,epoch):

    # Sample random noise
    noise = np.random.uniform(0, 1, (epoch, z_dim))

    # Generate images from random noise
    
    generated_images = g.predict(noise)
#     generated_images = generated_images.reshape(256, 256, 1)

    for i in range(len(generated_images)):
        
        generated_images = 0.5 * generated_images + 0.5
    #     plt.figure(figsize=(2, 2))
        plt.imshow(generated_images[i].reshape(256, 256, 1), cmap = 'gray')
        plt.axis('off')


    plt.savefig(anogan_generate_path+"anogan_generate{:0n}.jpg".format(epoch))
    plt.show()


def sum_of_residual(y_true, y_pred):
    return tf.reduce_sum(abs(y_true - y_pred))

def feature_extractor():
    d = build_discriminator(img_shape)
    d.load_weights(model_save_path + "anogan_discriminator.h5")
    intermidiate_model = Model(inputs=d.layers[0].input, outputs=d.layers[-5].output)
    intermidiate_model.compile(loss='binary_crossentropy', optimizer='adam')
    return intermidiate_model

def anomaly_detector():
    g = build_generator(z_dim)
    g.load_weights(model_save_path + "anogan_generator.h5")
    g.trainable = False
    intermidiate_model = feature_extractor()
    intermidiate_model.trainable = False
    
    aInput = Input(shape=(256,))
    gInput = Dense((256))(aInput)
    G_out = g(gInput)
    D_out= intermidiate_model(G_out)    
    model = Model(inputs=aInput, outputs=[G_out, D_out])
    model.compile(loss=sum_of_residual, loss_weights= [0.9, 0.1], optimizer='adam')
    return model

def compute_anomaly_score(model, x):    
    z = np.random.uniform(0, 1, size=(1, 256))
    intermidiate_model = feature_extractor()
    d_x = intermidiate_model.predict(x)
    loss = model.fit(z, [x, d_x], epochs=500, verbose=0)
    similar_data, _ = model.predict(z)
    return loss.history['loss'][-1], similar_data

