from keras.layers import Conv3D,Dense,Flatten,MaxPooling3D,Input,Dropout,BatchNormalization
from keras.models import Model
from keras.regularizers import l2

def model_original_size(input_shape=(112, 112, 16, 3), nb_classes=6, weight_decay = 0.005):
    inputs = Input(input_shape)
    # 112x112x16
    x = Conv3D(32,(3,3,3),strides=(1,1,1),activation='relu',padding='same',
               kernel_regularizer=l2(weight_decay),name='conv1_ori')(inputs)
    x = MaxPooling3D((2,2,1),strides=(2,2,1),padding='same')(x)
    # 56x56x16
    x = Conv3D(64,(3,3,3),strides=(1,1,1),activation='relu',padding='same',
               kernel_regularizer=l2(weight_decay),name='conv2_ori')(x)
    x = MaxPooling3D((2,2,2),strides=(2,2,2),padding='same')(x)
    # 28x28x8
    x = Conv3D(128,(3,3,3),strides=(1,1,1),activation='relu',padding='same',
               kernel_regularizer=l2(weight_decay),name='conv3_ori')(x)
    x = MaxPooling3D((2,2,2),strides=(2,2,2),padding='same')(x)
    # 14x14x4
    x = Conv3D(256, (3, 3, 3), strides=(1, 1, 1), activation='relu',padding='same',
               kernel_regularizer=l2(weight_decay),name='conv4_ori')(x)
    x = MaxPooling3D((2, 2, 2), strides=(2, 2, 2),padding='same')(x)
    # 7x7x2
    x = Conv3D(256, (3, 3, 3), strides=(1, 1, 1), activation='relu',padding='same',
               kernel_regularizer=l2(weight_decay),name='conv5_ori')(x)
    x = MaxPooling3D((2, 2, 2), strides=(2, 2, 2),padding='same')(x)
    # 4x4x1
    x = Flatten()(x)
    y = Dense(1024,activation='relu',kernel_regularizer=l2(weight_decay),
              name='fc1_ori')(x)
    x = Dropout(0.5)(y)
    x = Dense(1024,activation='relu',kernel_regularizer=l2(weight_decay),
              name='fc2_ori')(x)
    x = Dropout(0.5)(x)
    out = Dense(nb_classes,activation='softmax',kernel_regularizer=l2(weight_decay),
                name='fc3_ori')(x)
    model_ori = Model(inputs=inputs, outputs=out)
    model_frature_ori = Model(inputs=inputs, outputs=y)
    return model_ori,model_frature_ori


def model_crop_size(input_shape=(64, 64, 16, 3), nb_classes=6,weight_decay = 0.005):
    inputs = Input(input_shape)
    # 64x64x16
    x = Conv3D(32, (3, 3, 3), strides=(1, 1, 1), activation='relu',padding='same',
               kernel_regularizer=l2(weight_decay),name='conv1_crop')(inputs)
    x = MaxPooling3D((2, 2, 2), strides=(2, 2, 2),padding='same')(x)
    # 32x32x8
    x = Conv3D(64, (3, 3, 3), strides=(1, 1, 1), activation='relu',padding='same',
               kernel_regularizer=l2(weight_decay),name='conv2_crop')(x)
    x = MaxPooling3D((2, 2, 2), strides=(2, 2, 2),padding='same')(x)
    #16x16x4
    x = Conv3D(128, (3, 3, 3), strides=(1, 1, 1), activation='relu',padding='same',
               kernel_regularizer=l2(weight_decay),name='conv3_crop')(x)
    x = MaxPooling3D((2, 2, 2), strides=(2, 2, 2),padding='same')(x)
    #8x8x2
    x = Conv3D(128, (3, 3, 3), strides=(1, 1, 1), activation='relu',padding='same',
               kernel_regularizer=l2(weight_decay),name='conv4_crop')(x)
    x = MaxPooling3D((2, 2, 2), strides=(2, 2, 2),padding='same')(x)
    #4x4x1
    x = Flatten()(x)
    y = Dense(1024, activation='relu',kernel_regularizer=l2(weight_decay),
              name='fc1_crop')(x)
    x = Dropout(0.5)(y)
    x = Dense(1024, activation='relu',kernel_regularizer=l2(weight_decay),
              name='fc2_crop')(x)
    x = Dropout(0.5)(x)
    out = Dense(nb_classes, activation='softmax',kernel_regularizer=l2(weight_decay),
                name='fc3_crop')(x)

    model_crop = Model(inputs=inputs, outputs=out)
    model_feature_crop = Model(inputs=inputs,outputs=y)
    return model_crop,model_feature_crop