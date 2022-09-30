# Keras Subclassing 형태의 모델 구조 class 정의

import tensorflow as tf

# Implementation using tf.keras.applications (https://www.tensorflow.org/api_docs/python/tf/keras/applications)
## --> tf.keras.applications를 이용해서 미리 pretrained된 Inception V3모델을 가져오게 

# & Keras Functional API (https://www.tensorflow.org/guide/keras/functional)
## --> 이런 형태의 구현을 Keras Functional API라고 함
## --> Keras Functional API를 하게 되면 layer의 특정 input 구조와 output 구조를 우리가 원하는 형태로 지정해서 그거에 맞는 functional API를 만들 수 있음

class YOLOv1(tf.keras.Model):

  ## 생성자 인자로 train.py에서 넘겨주는 resized된 사이즈,...
  def __init__(self, input_height, input_width, cell_size, boxes_per_cell, num_classes):
    super(YOLOv1, self).__init__()
    
    ## tf.keras.application에서 InceptionV3 모델을 가져옴/ include_top=False : 마지막 softmax regression layer는 빼고 가져오겠따.
    ## weights='imagenet' : imagenet에서 pretrained된 것을 가져오겠다.
    ## 원논문처럼 darknet 자체 프레임워크로 GoogleLeNEt과 비슷한 layer를 직접 정의해서 1주일 학습시키고 fine tuning하는 것처럼 저자처럼 할 필요가 굳이 없기 때문에 inception v3(google lenet과 비슷)가져와서 사용
    base_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet', input_shape=(input_height, input_width, 3))
    base_model.trainable = True ## False로 주게 되면 imagenet으로 학습된 파라미터가 freezing(새로운 학습으로 인한 Convlayer 변동사항없고 Conv, feature extractor지나서 softmax regression vector만 갱신하겠다.)된 상태
                                ## True면 앞의 Convlayer까지 학습을 하게 되서 시간이 좀 더 걸림

    x = base_model.output # InceptionV3 FC layer 전의 feature map



    # Global Average Pooling
    ## 거기에 대해서 GAP 적용해서 feature map 을 하나의 scalar vector로 모아주게 됨
    ## 마지막 feature map이 필터 개수만큼 나온 것에 GAP를 이용해서 feature map에 대한 dimension을 다 1차원으로 변경시켜버려서 한번에 fc layer같은 차원 축소를 진행해주고
    ## GAP이 적용된 feature map들에 대해서 yolo shape에 맞는 cell_size*cell_size * num_classes + (5*boxes_per_cell) 같은 fc layer같은 flatten layer를 output에 Dense로 정의
    
    ## GAP : 배운 풀링을 전체 feature map에 대해서 수행 : 하나의 feature map을 평균값으로 하나의 스칼라 값(하나의 특징값)으로 만들어 버림
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    output = tf.keras.layers.Dense(cell_size * cell_size * (num_classes + (boxes_per_cell*5)), activation=None)(x)
    model = tf.keras.Model(inputs=base_model.input, outputs=output) # input : resized된 욜로 인풋 / output : yolo형태에 맞는 flatten된 vector를 prediction하는 모델
    self.model = model
    # print model structure
    self.model.summary()

  def call(self, x):
    return self.model(x)