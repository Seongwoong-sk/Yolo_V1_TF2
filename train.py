# 모델 class를 인스턴스로 선언하고 For-loop을 돌면서 gradient descent를 수행하면서 파라미터를 업데이트 하는 로직

import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import os
import random

## absl library : google에서 만든 파이썬 라이브러리
## tf에서 코드 구현할 때 자주 사용됨
from absl import flags
from absl import app

## loss.py에서 정의한 yolo_loss func
from loss import yolo_loss

## model.py에서 정의한 YOLOv1 Keras subclass 형태로 구성된 모델 클래스
from model import YOLOv1

## dataset.py에서 정의된 func
from dataset import process_each_ground_truth

## utils.py에서 정의된 func
from utils import draw_bounding_box_and_label_info, generate_color, find_max_confidence_bounding_box, yolo_format_to_bounding_box_dict

######################################################################


# set cat 'label dictionary'
## OD 작업에서 빈번하게 사용되는 패턴
## --> 키로 int, value로 label의 string >> DL Model이 예측할 수 있는 것은 int나 float같은 숫자형 자료이기 때문에 이렇게
cat_label_dict = {
  0: "cat" ## 원래 PASCAL VOC 데이터는 20개의 레이블로 구성되어 있지만 이 모델은 고양이 하나만 사용한다.
}
## key와 value의 위치를 바꾸는 reverse dict를 추가적으로 하나 만들어서 label 관련된 처리를 사용하게 됨
cat_class_to_label_dict = {v: k for k, v in cat_label_dict.items()}

#######################################################################

## Terminal에서 변경될 수 있는 가변적인 값들을 지정 
## 
flags.DEFINE_string('checkpoint_path', default='saved_model', help='path to a directory to save model checkpoints during training')
flags.DEFINE_integer('save_checkpoint_steps', default=50, help='period at which checkpoints are saved (defaults to every 50 steps)')
flags.DEFINE_string('tensorboard_log_path', default='tensorboard_log', help='path to a directory to save tensorboard log')
flags.DEFINE_integer('validation_steps', default=50, help='period at which test prediction result and save image')
flags.DEFINE_integer('num_epochs', default=135, help='training epochs') # original paper : 135 epoch
flags.DEFINE_float('init_learning_rate', default=0.0001, help='initial learning rate') # original paper : 0.001 (1epoch) -> 0.01 (75epoch) -> 0.001 (30epoch) -> 0.0001 (30epoch)
flags.DEFINE_float('lr_decay_rate', default=0.5, help='decay rate for the learning rate')
flags.DEFINE_integer('lr_decay_steps', default=2000, help='number of steps after which the learning rate is decayed by decay rate') ## 2,000 steps마다 lr을 1/2만큼 줄여감
flags.DEFINE_integer('num_visualize_image', default=8, help='number of visualize image for validation') ## 중간중간 validation 관련된 것들을 전체 배치에서 몇 개를 test image로 visualizing할건지

FLAGS = flags.FLAGS # 전역적으로 FLAGS로 묶어줌

###################################################################


# set configuration value
## YOLO 설정값

batch_size = 24 # original paper : 64 / batch_size가 크면 gpu사용량이 증가하기 때문에 적절한 사이즈  

## original의 size를 resize하는 부분
input_width = 224 # original paper : 448
input_height = 224 # original paper : 448
cell_size = 7
num_classes = 1 # original paper : 20
boxes_per_cell = 2 ## 하나의 cell에서 몇 개의 bbox를 prediction할 건지

# set color_list for drawing
color_list = generate_color(num_classes) ## class 개수 만큼의 랜덤 컬러값들을 반환받음 --> 클래스별 bbox 색깔 다르게


##############################################################################


# set loss function coefficients
## Loss Function coefficient 지정

coord_scale = 10 # original paper : 5
class_scale = 0.1  # original paper : 1
object_scale = 1 ## obj가 있는 부분에 대한 confidence 앞에 곱해지는 
noobject_scale = 0.5


#########################################################################################
# load pascal voc2007/voc2012 dataset using tfds
## tensorflow_datasets 라이브러리를 이용해서 Pascal VOC cat dataset 불러오기
## reference : https://www.tensorflow.org/datasets/catalog/voc
### This Structure
### train : 2007 test + 2012 train + 2012 val 
### val : 2007 val 
### test : 2007 train 


# notice : voc2007 train data(=2,501 images) for test & voc2007 test data(=4,952 images) for training
## --> 이렇게 적용한 이유 : 많은 양을 train 데이터로 접근
voc2007_test_split_data = tfds.load("voc/2007", split=tfds.Split.TEST, batch_size=1)
voc2012_train_split_data = tfds.load("voc/2012", split=tfds.Split.TRAIN, batch_size=1)
voc2012_validation_split_data = tfds.load("voc/2012", split=tfds.Split.VALIDATION, batch_size=1)

## 위 3개를 불러오고 concatenate를 이용해서 하나의 데이터셋 완성
train_data = voc2007_test_split_data.concatenate(voc2012_train_split_data).concatenate(voc2012_validation_split_data)


# set validation data
## training 중간 중간에 validation 성능을 측정하기 위해 validation set으로 따로 생성
voc2007_validation_split_data = tfds.load("voc/2007", split=tfds.Split.VALIDATION, batch_size=1)
validation_data = voc2007_validation_split_data


# label 7 : cat
## voc의 전체 label 20개 중에서 7(cat) parsing
# Reference : https://stackoverflow.com/questions/55731774/filter-dataset-to-get-just-images-from-specific-class

def predicate(x, allowed_labels=tf.constant([7.0])):
  label = x['objects']['label']
  isallowed = tf.equal(allowed_labels, tf.cast(label, tf.float32))
  reduced = tf.reduce_sum(tf.cast(isallowed, tf.float32))

  ## 7번 레이블이 포함된 데이터셋만 따로 필터링 하기 위한
  return tf.greater(reduced, tf.constant(0.))

## filter func에 predicate 조건을 만들어 준 다음에 7번 레이블이 포함된 데이터셋만 다시 정제해서 가져오고
train_data = train_data.filter(predicate)
## padded_batch func을 이용해서 batch_size로 지정한 개수만큼 cat dataset을 배치 단위로 묶어주는 작업
train_data = train_data.padded_batch(batch_size)


## 마찬가지로 val data에 대해서도 cat데이터만 필터해서 가져오고 padded_batch func을 이용해서 batch 개수로 다시 묶어줌
validation_data = validation_data.filter(predicate)
validation_data = validation_data.padded_batch(batch_size)


#########################################################################################


## 길게 flatten 되있던 vector를 yolo prediction에서 사용하기 편리한 형태인 cell_size * cell_size , numclass, 5*bbox 형태로 변환
def reshape_yolo_preds(preds):
  # flatten vector -> cell_size x cell_size x (num_classes + 5 * boxes_per_cell)
  return tf.reshape(preds, [tf.shape(preds)[0], cell_size, cell_size, num_classes + 5 * boxes_per_cell])


## loss function을 정의하는 부분
def calculate_loss(model, batch_image, batch_bbox, batch_labels):
  total_loss = 0.0
  coord_loss = 0.0
  object_loss = 0.0
  noobject_loss = 0.0
  class_loss = 0.0

  ## 전체 배치를 하나씩 돌아가면서 dataset.py에서 만든 process_each_ground_truth 함수에 해당 batch_index에 대응되는 한 장의 데이터를 얻어서 yolo에서 사용할 수 있는 형태로 resize되고 
  ## 절대좌표로 바뀌고, 실제 obj 개수만큼 파싱된 정보값을 리턴을 받고 
  for batch_index in range(batch_image.shape[0]):
    image, labels, object_num = process_each_ground_truth(batch_image[batch_index], batch_bbox[batch_index], batch_labels[batch_index], input_width, input_height)
    
    ## 이미지가 한장으로 리턴된거기때문에 앞에 더미 디멘션을 만들어주기 위해 차원 늘림
    image = tf.expand_dims(image, axis=0)

    ## 더미 디멘션 만들어준 상태에서 yolo 모델이 predict한 flatten vector 하나 받고
    predict = model(image)
    
    ## grid_cell, grid_cell, bbox, classnum에 대응되는 벡터 형상으로 reshape한 다음에
    predict = reshape_yolo_preds(predict)

    ## 실제 obj가 존재하는 개수만큼 for loop을 돌면서 yolo_loss func에 우리가 predict한 값과 관련된 정답값,coefficient같은 것을 쭉 인자값으로 넣어서 
    ## 여기서 이미지내에서 존재하는 하나의 obj에 대한 total_loss, coordinateloss, obj loss, noobj loss, class loss를 계산을 하고

    for object_num_index in range(object_num):
      each_object_total_loss, each_object_coord_loss, each_object_object_loss, each_object_noobject_loss, each_object_class_loss = yolo_loss(predict[0],
                                   labels,
                                   object_num_index,
                                   num_classes,
                                   boxes_per_cell,
                                   cell_size,
                                   input_width,
                                   input_height,
                                   coord_scale,
                                   object_scale,
                                   noobject_scale,
                                   class_scale
                                   )
## 1. 각각 이미지 내에 존재하는 obj별 loss값을 합산을 하고
## 2. 그걸 다시 전체의 batch에 대한 합산을 하게 되면 
## 하나의 배치단위에서 계산된 전체 loss값들이 
      total_loss = total_loss + each_object_total_loss
      coord_loss = coord_loss + each_object_coord_loss
      object_loss = object_loss + each_object_object_loss
      noobject_loss = noobject_loss + each_object_noobject_loss
      class_loss = class_loss + each_object_class_loss

  return total_loss, coord_loss, object_loss, noobject_loss, class_loss



## 실제 GD가 1 step진행하는 과정이 포함되어 있는 함수
## 정답 관련된 정보들을 받아서 model prediction값과 정답 간의 어떤 loss fucntion을 계산하는 calculate_loss func을 이용해서 개별 loss func에 대한 것들을 실제 계산해서 loss func을 받아오게 됨
def train_step(optimizer, model, batch_image, batch_bbox, batch_labels):
  with tf.GradientTape() as tape:
    total_loss, coord_loss, object_loss, noobject_loss, class_loss = calculate_loss(model, batch_image, batch_bbox, batch_labels)
  
## 전체 Network 파라미터를 갱신할때는 total_loss에 대한 것을 계산한 다음에 total loss에 대한 GD를 
## apply_gradients에 반영해주게 되면 train step func이 한 번 호출될 때마다 yolo 모델의 전체 파라미터가 우리 데이터셋의 obj를 잘 검출할 수 있는 방향성의 GD로 한번씩 갱신이 이루어지는 과정
## total_loss에 기반해서 실제 GD를 1번 수행하는 로직   
  gradients = tape.gradient(total_loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  return total_loss, coord_loss, object_loss, noobject_loss, class_loss


#########################################################################################
# Validation
## --> save_validation_result func 같은 경우는 전체 validation data에 대한 loss funtion값을 텐서보드에 로그로 남기고
## --> num_visualize_image 개수만큼 yolo prediction 결과값과 정답간의 bbox를 그려서 그걸 이제 tf.summary.image로 남겨주는 과정이 작성되어있음 


def save_validation_result(model, ckpt, validation_summary_writer, num_visualize_image):
  ## 전체 validation 를 가져와서
  total_validation_total_loss = 0.0
  total_validation_coord_loss = 0.0
  total_validation_object_loss = 0.0
  total_validation_noobject_loss = 0.0
  total_validation_class_loss = 0.0

  ## 전체 validation data에 대해 학습을 하게 되면 현재 우리 yolo 모델의 파라미터와 loss값을 측정을 하면서 model 성능을 객관적으로(일반화된 상태로 ) 측정 
  ## 전체 validaton data에 대해 순회를 하면서 
  for iter, features in enumerate(validation_data):
    batch_validation_image = features['image']
    batch_validation_bbox = features['objects']['bbox']
    batch_validation_labels = features['objects']['label']

    batch_validation_image = tf.squeeze(batch_validation_image, axis=1)
    batch_validation_bbox = tf.squeeze(batch_validation_bbox, axis=1)
    batch_validation_labels = tf.squeeze(batch_validation_labels, axis=1)

 ## validation data와 우리 yolo 모델이 예측한 validation error값을
    validation_total_loss, validation_coord_loss, validation_object_loss, validation_noobject_loss, validation_class_loss = calculate_loss(model, batch_validation_image, batch_validation_bbox, batch_validation_labels)

 ## 개별적으로 측정
    total_validation_total_loss = total_validation_total_loss + validation_total_loss
    total_validation_coord_loss = total_validation_coord_loss + validation_coord_loss
    total_validation_object_loss = total_validation_object_loss + validation_object_loss
    total_validation_noobject_loss = total_validation_noobject_loss + validation_noobject_loss
    total_validation_class_loss = total_validation_class_loss + validation_class_loss

  # save validation tensorboard log
  ## validation_summary_writer쪽에서 불러온 부분에 total값들을 summary.scalar값으로 tensorboard에 저장
  ## Overfitting에 빠져서 loss가 감소하는 것일수도 있기 때문에 model이 잘 개선되고 있는지 체크하려면 total val loss로 체크 (train에서 보지 않은 이미지에 대해서 loss값이 잘 감소하고 있구나)
  with validation_summary_writer.as_default():
    tf.summary.scalar('total_validation_total_loss', total_validation_total_loss, step=int(ckpt.step))
    tf.summary.scalar('total_validation_coord_loss', total_validation_coord_loss, step=int(ckpt.step))
    tf.summary.scalar('total_validation_object_loss ', total_validation_object_loss, step=int(ckpt.step))
    tf.summary.scalar('total_validation_noobject_loss ', total_validation_noobject_loss, step=int(ckpt.step))
    tf.summary.scalar('total_validation_class_loss ', total_validation_class_loss, step=int(ckpt.step))

  # save validation test image
  ## num_visualize_image 개수만큼 tf.summary api를 이용해서 현재 파라미터에 기반한 prediction 값과 정답값의 차이를 정성적으로 한번에 볼 수 있음
  for validation_image_index in range(num_visualize_image):

    ## 현재 시점의 batch_validation image중에 랜덤하게 하나의 이미지를 선택해서 
    random_idx = random.randint(0, batch_validation_image.shape[0] - 1)
    ## 해당 이미지를 이용해서 process_each_ground_truth func을 이용해서 해당 이미지의 정답과 resize된 이미지 , obj 개수를 받아오고
    image, labels, object_num = process_each_ground_truth(batch_validation_image[random_idx], batch_validation_bbox[random_idx],
                                                          batch_validation_labels[random_idx], input_width, input_height)
 ## 변수에 할당해서 yolo모델이 예측한 결과값을 받아오고
    drawing_image = image

    image = tf.expand_dims(image, axis=0)
    predict = model(image)
    predict = reshape_yolo_preds(predict)


    # parse prediction

    ## flatten vector 30차원 : 20-label 2-confidence 4-position vector 4-position vector 
    predict_boxes = predict[0, :, :, num_classes + boxes_per_cell:]
    ## 4 : xcenter ycenter box width, box height에 대한 position vector 값
    predict_boxes = tf.reshape(predict_boxes, [cell_size, cell_size, boxes_per_cell, 4])


    ## class 이후 2 confidence를 파싱해서 box_per_cell이 confidence값이 되고 
    confidence_boxes = predict[0, :, :, num_classes:num_classes + boxes_per_cell]
    confidence_boxes = tf.reshape(confidence_boxes, [cell_size, cell_size, boxes_per_cell, 1])

    ## 20-label을 파싱해서 class에 대한 prediction 
    class_prediction = predict[0, :, :, 0:num_classes]
    class_prediction = tf.argmax(class_prediction, axis=2)

    # make prediction bounding box list
    ## yolo가 예측한 bbox별 prediction 값 (xcenter, ycenter, box_w, box_h, class )
    bounding_box_info_list = []
    for i in range(cell_size):
      for j in range(cell_size):
        for k in range(boxes_per_cell):
          pred_xcenter = predict_boxes[i][j][k][0]
          pred_ycenter = predict_boxes[i][j][k][1]
          pred_box_w = tf.minimum(input_width * 1.0, tf.maximum(0.0, predict_boxes[i][j][k][2]))
          pred_box_h = tf.minimum(input_height * 1.0, tf.maximum(0.0, predict_boxes[i][j][k][3]))
          
          ## cat_label_dict에서 int값을 cat이라는 string형태로 변환
          pred_class_name = cat_label_dict[class_prediction[i][j].numpy()]
          pred_confidence = confidence_boxes[i][j][k].numpy()[0]

          # add bounding box dict list
          ## utils.py에서 정의된 yolo_format_to_bounding_box_dict를 이용하게 되면 개별 prediction값들이 그 yolo prediction 결과값인 xcenter,ycenter,box_w,box_h를
          ## 다시 xmin,ymin,xmax,ymax로 바꿔주고 bounding box info라는 리스트로 만들어서 넣어줌
          ## 이 리스트는 cell_size, cell_size, boxes_per_cell로 총 98개의 bbox가 ㄷ들어감
          bounding_box_info_list.append(yolo_format_to_bounding_box_dict(pred_xcenter, pred_ycenter, pred_box_w, pred_box_h, pred_class_name, pred_confidence))



    # make ground truth bounding box list
    ## ground truth 데이터셋에 있는 정보들을 labels에서 파싱을 해서 가져온 다음에
    ground_truth_bounding_box_info_list = []
    for each_object_num in range(object_num):
      labels = np.array(labels)
      labels = labels.astype('float32')
      label = labels[each_object_num, :]
      xcenter = label[0]
      ycenter = label[1]
      box_w = label[2]
      box_h = label[3]
      class_label = label[4]

      # label 7 : cat
      # add ground-turth bounding box dict list
      ## 레이블이 맞으면 bbox info format으로 ground truth bbox info list에다가 groud truth bbox에 대한 것들을 리스트 안에 넣기
      if class_label == 7:
        ground_truth_bounding_box_info_list.append(
          yolo_format_to_bounding_box_dict(xcenter, ycenter, box_w, box_h, 'cat', 1.0))


## drawing_image로 따로 빼놓은 resize된 yolo 원본 input image를 가지고 있는 상태에서 
## 우선은 ground truth bbox에 대한 정보를 전부 순회하면서 draw_bounding_box_and_label_info를 이용해서 
## bbox info 형태로 표기된 drawing에 표기된 정보들 / position vector랑 class name이랑 confidence를 넣어서 컬러값을 지정한 상태에서 
## 화면에 ground truth에 대한 정보 : bbox와 class name과 confidence 값을 표현을 한 상태의 이미지를 ground truth drawing 이미지 부분에 할당 

    ground_truth_drawing_image = drawing_image.copy()
    # draw ground-truth image
    for ground_truth_bounding_box_info in ground_truth_bounding_box_info_list:
      draw_bounding_box_and_label_info(
        ground_truth_drawing_image,
        ground_truth_bounding_box_info['left'],
        ground_truth_bounding_box_info['top'],
        ground_truth_bounding_box_info['right'],
        ground_truth_bounding_box_info['bottom'],
        ground_truth_bounding_box_info['class_name'],
        ground_truth_bounding_box_info['confidence'],
        color_list[cat_class_to_label_dict[ground_truth_bounding_box_info['class_name']]]
      )


    # find one max confidence bounding box
    ## yolo prediction 결과를 시각화하는 부분
    ## utils.py에서 정의한 find_max_confidence_bounding_box에 bounding_box_info_list (98개의 bbox) 중에서 confidence가 가장 큰 bbox 하나만 선택을 해주게  됨
    max_confidence_bounding_box = find_max_confidence_bounding_box(bounding_box_info_list)

    # draw prediction
    ## 하나 선택한 bbox를 이용해서 drawing_image부분에 yolo가 prediction한 confidence가 가장 큰 하나의 bbox를 이미지에 표현을 해주게 
    draw_bounding_box_and_label_info(
      drawing_image,
      max_confidence_bounding_box['left'],
      max_confidence_bounding_box['top'],
      max_confidence_bounding_box['right'],
      max_confidence_bounding_box['bottom'],
      max_confidence_bounding_box['class_name'],
      max_confidence_bounding_box['confidence'],
      color_list[cat_class_to_label_dict[max_confidence_bounding_box['class_name']]]
    )

    # left : ground-truth, right : prediction
    ## np.concatenate를 이용해서 이미지의 왼쪽 부분은 ground-truth / 오른쪽 부분은 prediction 결괏값
    drawing_image = np.concatenate((ground_truth_drawing_image, drawing_image), axis=1)
    drawing_image = drawing_image / 255
    drawing_image = tf.expand_dims(drawing_image, axis=0)

    # save tensorboard log
    ## 해당 이미지를 tf.summary.image api를 이용해서 해당 step의 prediction 결괏값을 한 눈에 볼 수 있는 tf.summary.image log를 남기고 
    ## 그것을 253 line에 있는 num_visualize_image 개수 설정한 것만큼 왼쪽에는 ground-truth가 포함된 이미지 / 오른쪽에는 yolo가 예측한 것 중에서 가장 confidence가 큰 bbox를 하나만 뽑아서 drawing한 이미지를 보여주는 이미지 로그를
    ## validation_summary_writer로 지정을 한다.
    with validation_summary_writer.as_default():
      tf.summary.image('validation_image_'+str(validation_image_index), drawing_image, step=int(ckpt.step))

def main(_):
  # set learning rate decay
  ## tf api에서 lr decay를 적용 
  lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    FLAGS.init_learning_rate,
    decay_steps=FLAGS.lr_decay_steps,
    decay_rate=FLAGS.lr_decay_rate,
    staircase=True)

  # set optimizer
  optimizer = tf.optimizers.Adam(lr_schedule)  # original paper (2015) : SGD with momentum 0.9, decay 0.0005

  # check if checkpoint path exists
  ## checkpoint path에 중간 파라미터를 저장
  if not os.path.exists(FLAGS.checkpoint_path):
    os.mkdir(FLAGS.checkpoint_path)

  # create YOLO model
  YOLOv1_model = YOLOv1(input_height, input_width, cell_size, boxes_per_cell, num_classes)

  # set checkpoint manager
  ckpt = tf.train.Checkpoint(step=tf.Variable(0), model=YOLOv1_model)
  ckpt_manager = tf.train.CheckpointManager(ckpt,
                                            directory=FLAGS.checkpoint_path,
                                            max_to_keep=None)
  latest_ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_path)

  # restore latest checkpoint
  if latest_ckpt:
    ckpt.restore(latest_ckpt)
    print('global_step : {}, checkpoint is restored!'.format(int(ckpt.step)))

  # set tensorboard log
  ## tensorboard를 write하기 위한 summar writer를 선언
  train_summary_writer = tf.summary.create_file_writer(FLAGS.tensorboard_log_path +  '/train')
  validation_summary_writer = tf.summary.create_file_writer(FLAGS.tensorboard_log_path +  '/validation')

## 지정 횟수만큼 for loop을 돌면서 
  for epoch in range(FLAGS.num_epochs):
    num_batch = len(list(train_data))

    ## 전체 데이터셋에 있는 개별 batch 단위의 데이터를 다시 batch 단위로 for loop을 돌면서 가져오고
    ## 해당 batch의 features 부분에 tensorflow api의 dict형태로 구성된 정답데이터가 들어가있음 : https://www.tensorflow.org/datasets/catalog/voc
    ## features부분에서 필요한 부분만 key값으로 접근을 해서 가져옴 
    for iter, features in enumerate(train_data):
      batch_image = features['image']
      batch_bbox = features['objects']['bbox']
      batch_labels = features['objects']['label']

    ## tf.squeeze는 tf.expand_dims과 반대 -> 차원을 줄
    ## 불러온 값의 앞에 dummy dimension이 있어서 삭제
      batch_image = tf.squeeze(batch_image, axis=1)
      batch_bbox = tf.squeeze(batch_bbox, axis=1)
      batch_labels = tf.squeeze(batch_labels, axis=1)

      # run optimization and calculate loss
      ## train step 함수에 실제 ground truth데이터를 넣어줌
      ## 실제 yolo모델이 내 데이터(Custom Dataset)에 맞게 파라미터 갱신이 이뤄짐
      total_loss, coord_loss, object_loss, noobject_loss, class_loss = train_step(optimizer, YOLOv1_model, batch_image, batch_bbox, batch_labels)

      # print log
      ## total_loss값은 로그성 프린트문으로 터미널에서 찍어줌
      print("Epoch: %d, Iter: %d/%d, Loss: %f" % ((epoch+1), (iter+1), num_batch, total_loss.numpy()))

      # save tensorboard log
      ## 현재 스텝시점에 yolo 모델이 예측한 값들을 개별적으로 다 tensorboard 로그로 summary.scalar로 남기게 
      with train_summary_writer.as_default():
        tf.summary.scalar('learning_rate ', optimizer.lr(ckpt.step).numpy(), step=int(ckpt.step))
        tf.summary.scalar('total_loss', total_loss, step=int(ckpt.step))
        tf.summary.scalar('coord_loss', coord_loss, step=int(ckpt.step))
        tf.summary.scalar('object_loss ', object_loss, step=int(ckpt.step))
        tf.summary.scalar('noobject_loss ', noobject_loss, step=int(ckpt.step))
        tf.summary.scalar('class_loss ', class_loss, step=int(ckpt.step))

      # save checkpoint
      ## 지정한 ckpt시점에 도달할 때마다  
      if ckpt.step % FLAGS.save_checkpoint_steps == 0:
        # save checkpoint
        ## 현재 yolo 모델의 파라미터를 path경로에다가 ckpt파일로 저장하게 되고 
        ckpt_manager.save(checkpoint_number=ckpt.step)
        print('global_step : {}, checkpoint is saved!'.format(int(ckpt.step)))

      ## 매 반복마다 전역적인 반복 횟수에 대한 ckpt 스텝을 하나씩 +1 
      ckpt.step.assign_add(1)

      # occasionally check validation data and save tensorboard log
      ## 지정 validation step에 도달했을 때 sav~~ func을 호출해서 현재 yolo 모델 파라미터에 기반한 validation을 진행
      if iter % FLAGS.validation_steps == 0:
        save_validation_result(YOLOv1_model, ckpt, validation_summary_writer, FLAGS.num_visualize_image)

## 최초 호출
if __name__ == '__main__':
  app.run(main)