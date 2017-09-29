# -*- coding: utf-8 -*-
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from keras.layers import Input, add, concatenate
from keras.models import Model
from keras.optimizers import RMSprop

from losses import binary_crossentropy, dice_loss, bce_dice_loss, dice_coef, weighted_bce_dice_loss

def get_unet_MDCB(input_shape=(1920, 1280, 3), init_nb=44, lr=0.0001, loss=bce_dice_loss):
    
    inputs = Input(input_shape)
    
    down1 = Conv2D(init_nb, (3, 3), activation='relu', padding='same')(inputs)
    down1 = Conv2D(init_nb, (3, 3), activation='relu', padding='same')(down1)
    down1pool = MaxPooling2D((2, 2), strides=(2, 2))(down1)
    
    down2 = Conv2D(init_nb*2, (3, 3), activation='relu', padding='same')(down1pool)
    down2 = Conv2D(init_nb*2, (3, 3), activation='relu', padding='same')(down2)
    down2pool = MaxPooling2D((2, 2), strides=(2, 2))(down2)

    down3 = Conv2D(init_nb*4, (3, 3), activation='relu', padding='same')(down2pool)
    down3 = Conv2D(init_nb*4, (3, 3), activation='relu', padding='same')(down3)
    down3pool = MaxPooling2D((2, 2), strides=(2, 2))(down3)
    
    # stacked dilated convolution
    dilate1 = Conv2D(init_nb*8, (3, 3), activation='relu', padding='same', dilation_rate=1)(down3pool)
    dilate2 = Conv2D(init_nb*8, (3, 3), activation='relu', padding='same', dilation_rate=2)(dilate1)
    dilate3 = Conv2D(init_nb*8, (3, 3), activation='relu', padding='same', dilation_rate=4)(dilate2)
    dilate4 = Conv2D(init_nb*8, (3, 3), activation='relu', padding='same', dilation_rate=8)(dilate3)
    dilate5 = Conv2D(init_nb*8, (3, 3), activation='relu', padding='same', dilation_rate=16)(dilate4)
    dilate6 = Conv2D(init_nb*8, (3, 3), activation='relu', padding='same', dilation_rate=32)(dilate5)
    dilate_all_added = add([dilate1, dilate2, dilate3, dilate4, dilate5, dilate6])
    
    up3 = UpSampling2D((2, 2))(dilate_all_added)
    up3 = Conv2D(init_nb*4, (3, 3), activation='relu', padding='same')(up3)
    up3 = concatenate([down3, up3])
    up3 = Conv2D(init_nb*4, (3, 3), activation='relu', padding='same')(up3)
    up3 = Conv2D(init_nb*4, (3, 3), activation='relu', padding='same')(up3)

    up2 = UpSampling2D((2, 2))(up3)
    up2 = Conv2D(init_nb*2, (3, 3), activation='relu', padding='same')(up2)
    up2 = concatenate([down2, up2])
    up2 = Conv2D(init_nb*2, (3, 3), activation='relu', padding='same')(up2)
    up2 = Conv2D(init_nb*2, (3, 3), activation='relu', padding='same')(up2)
    
    up1 = UpSampling2D((2, 2))(up2)
    up1 = Conv2D(init_nb, (3, 3), activation='relu', padding='same')(up1)
    up1 = concatenate([down1, up1])
    up1 = Conv2D(init_nb, (3, 3), activation='relu', padding='same')(up1)
    up1 = Conv2D(init_nb, (3, 3), activation='relu', padding='same')(up1)
    
    classify = Conv2D(1, (1, 1), activation='sigmoid')(up1)

    model = Model(inputs=inputs, outputs=classify)

    model.compile(optimizer=RMSprop(lr), loss=loss, metrics=[dice_coef])

    return model
