from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Activation
from keras.layers import BatchNormalization, Dense, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras import backend as K

def vgg_like(flag):
    img_size = flag.image_size
    lr = flag.initial_learning_rate
    num_classes = flag.num_classes

    inputs = Input((img_size, img_size, 1))
    # Block 1
    x = Conv2D(64, (3, 3), activation=None, padding='same', name='block1_conv1')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(64, (3, 3), activation=None, padding='same', name='block1_conv2')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation=None, padding='same', name='block2_conv1')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(128, (3, 3), activation=None, padding='same', name='block2_conv2')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation=None, padding='same', name='block3_conv1')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(256, (3, 3), activation=None, padding='same', name='block3_conv2')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(256, (3, 3), activation=None, padding='same', name='block3_conv3')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation=None, padding='same', name='block4_conv1')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(512, (3, 3), activation=None, padding='same', name='block4_conv2')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(512, (3, 3), activation=None, padding='same', name='block4_conv3')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    # x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    # x = Conv2D(512, (3, 3), activation=None, padding='same', name='block5_conv1')(x)
    # x = Conv2D(512, (3, 3), activation=None, padding='same', name='block5_conv2')(x)
    # x = Conv2D(512, (3, 3), activation=None, padding='same', name='block5_conv3')(x)
    # x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    x = Conv2D(1024, (3,3), activation=None, padding='same', name='conv6')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(num_classes, activation='softmax', name='predictions')(x)

    # if include_top:
    #     # Classification block
    #     x = Flatten(name='flatten')(x)
    #     x = Dense(4096, activation=None, name='fc1')(x)
    #     x = Dense(4096, activation=None, name='fc2')(x)
    #     x = Dense(classes, activation='softmax', name='predictions')(x)
    # else:
    #     if pooling == 'avg':
    #         x = GlobalAveragePooling2D()(x)
    #     elif pooling == 'max':
    #         x = GlobalMaxPooling2D()(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.

    # Create model.
    model = Model(inputs, x, name='vgg16')

    model.compile(optimizer=Adam(lr=lr, decay=1e-6), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

