# Evaluate.py/test.py 
# Training된 파라미터를 불러와서 evaluation이나 test/inference를 진행하는 로직

import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import cv2
import os

from absl import flags
from absl import app

from model import YOLOv1
from dataset import process_each_ground_truth
from utils import draw_bounding_box_and_label_info, generate_color, find_max_confidence_bounding_box, yolo_format_to_bounding_box_dict

# set voc label dictionary
cat_label_to_class_dict = {
  0:"cat"
}
cat_class_to_label_dict = {v: k for k, v in cat_label_to_class_dict.items()}

flags.DEFINE_string('checkpoint_path', default='saved_model', help='path to a directory to restore checkpoint file')
flags.DEFINE_string('test_dir', default='test_result', help='directory which test prediction result saved')

FLAGS = flags.FLAGS

# set configuration value
batch_size = 1
input_width = 224 # original paper : 448
input_height = 224 # original paper : 448
cell_size = 7
num_classes = 1 # original paper : 20
boxes_per_cell = 2

# set color_list for drawing
## generate_color를 이용해서 color값들을 불러옴
color_list = generate_color(num_classes)

# load pascal voc 2007 dataset using tfds
# notice : voc2007 train data(=2,501 images) for test & voc2007 test data(=4,952 images) for training
voc2007_train_split_data = tfds.load("voc/2007", split=tfds.Split.TRAIN, batch_size=1)
test_data = voc2007_train_split_data

# label 7 : cat
## label 데이터만 파싱을 해서 
def predicate(x, allowed_labels=tf.constant([7.0])):
  label = x['objects']['label']
  isallowed = tf.equal(allowed_labels, tf.cast(label, tf.float32))
  reduced = tf.reduce_sum(tf.cast(isallowed, tf.float32))

  return tf.greater(reduced, tf.constant(0.))

## 필터링을 해주고 batch_size만큼 만들어줌 
test_data = test_data.filter(predicate)
test_data = test_data.padded_batch(batch_size)


def reshape_yolo_preds(preds):
  # 7x7x(20+5*2) = 1470 -> 7x7x30
  ## flatten vector를 reshape된 형태로 변경
  return tf.reshape(preds, [tf.shape(preds)[0], cell_size, cell_size, num_classes + 5 * boxes_per_cell])


def main(_):
  # check if checkpoint path exists
  ## evaluation은 무조건 ckpt를 restore 해서 진행을 해야 하기 때문에 
  if not os.path.exists(FLAGS.checkpoint_path):
    print('checkpoint file is not exists!')
    exit() #ckpt가 없으면 프로그램 종료

  # create YOLO model
  YOLOv1_model = YOLOv1(input_height, input_width, cell_size, boxes_per_cell, num_classes)

  # set checkpoint manager
  ckpt = tf.train.Checkpoint(step=tf.Variable(0), model=YOLOv1_model)
  latest_ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_path)

  # restore latest checkpoint
  if latest_ckpt:
    ckpt.restore(latest_ckpt)
    print('global_step : {}, checkpoint is restored!'.format(int(ckpt.step)))

  num_images = len(list(test_data))  # batch_size = 1
  print('total test image :', num_images)
  for image_num, features in enumerate(test_data): # 한 장씩(batch_size=1) 테스트 이미지를 순회하면서 
    ## 정답데이터에 대한 값들을 파싱해서 가져오고 
    batch_image = features['image']
    batch_bbox = features['objects']['bbox']
    batch_labels = features['objects']['label']

    ## 불필요한 차원 제거
    batch_image = tf.squeeze(batch_image, axis=1)
    batch_bbox = tf.squeeze(batch_bbox, axis=1)
    batch_labels = tf.squeeze(batch_labels, axis=1)

    ## process_each_ground_truth func를 통해서 한 장의 이미지에 대한 image, labels, object_num 정보를 가져오고
    image, labels, object_num = process_each_ground_truth(batch_image[0], batch_bbox[0], batch_labels[0], input_width, input_height)

    drawing_image = image
    ## tf.expand_dims api를 이용해서 더미 디멘션을 만들어준 상태에서 
    image = tf.expand_dims(image, axis=0)

    ## yolo의 prediction값을 예측하고 
    predict = YOLOv1_model(image)
    ## 그것을 파싱할 수 있는 형태로 reshape 해 준 다음에 
    predict = reshape_yolo_preds(predict)

    ## box들에 관련된 부분들 개별적으로 파싱
    predict_boxes = predict[0, :, :, num_classes + boxes_per_cell:]
    predict_boxes = tf.reshape(predict_boxes, [cell_size, cell_size, boxes_per_cell, 4])

    confidence_boxes = predict[0, :, :, num_classes:num_classes + boxes_per_cell]
    confidence_boxes = tf.reshape(confidence_boxes, [cell_size, cell_size, boxes_per_cell, 1])

    class_prediction = predict[0, :, :, 0:num_classes]
    class_prediction = tf.argmax(class_prediction, axis=2)

    ## bbox list에 전체 98개의 yolo format을 가진 bbox 결괏값들을 변환해서 리스트에 저장
    bounding_box_info_list = []
    for i in range(cell_size):
      for j in range(cell_size):
        for k in range(boxes_per_cell):
          pred_xcenter = predict_boxes[i][j][k][0]
          pred_ycenter = predict_boxes[i][j][k][1]
          pred_box_w = tf.minimum(input_width * 1.0, tf.maximum(0.0, predict_boxes[i][j][k][2]))
          pred_box_h = tf.minimum(input_height * 1.0, tf.maximum(0.0, predict_boxes[i][j][k][3]))

          pred_class_name = cat_label_to_class_dict[class_prediction[i][j].numpy()]
          pred_confidence = confidence_boxes[i][j][k].numpy()

          # add bounding box dict list
          bounding_box_info_list.append(yolo_format_to_bounding_box_dict(pred_xcenter, pred_ycenter, pred_box_w, pred_box_h, pred_class_name, pred_confidence))

    # make ground truth bounding box list
    ## labels쪽에서 파싱해서 
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
      ## cat label에 관련된 정답들을 ground_truth_bbox_info_list에 넣어주고
      if class_label == 7:
        ground_truth_bounding_box_info_list.append(
          yolo_format_to_bounding_box_dict(xcenter, ycenter, box_w, box_h, 'cat', 1.0))

    ## ground_truth_drawing_image 부분에 정답 ground_truth 값을 다 drawing을 해주게 
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
    ## max_confidence_bounding_box에 98개의 bbox prediction 중에서 가장 confidence가 큰 거 하나 뽑아서  
    max_confidence_bounding_box = find_max_confidence_bounding_box(bounding_box_info_list)

    # draw prediction
    ## draw_bounding_box_and_label_info를 이용해서 max_confidence_bounding_box에 기반한 prediction 결과를 그려줌
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
    drawing_image = np.concatenate((ground_truth_drawing_image, drawing_image), axis=1)

    # save test prediction result to png file
    ## 여기선 tensorboard log로 tf.summary.image로 남기는 게 아니라 
    ## 실제 test_dir에 지정된 경로에 폴더를 하나 만든 다음에 그 해당 폴더에 image prediction 결과를 png파일로 한 장씩 image indexnum_result.png로 저장
    ## 그 때, open cv의 imwrite를 이용하고 그 때마다 로그성 문장을 출력
    if not os.path.exists(os.path.join(os.getcwd(), FLAGS.test_dir)):
      os.mkdir(os.path.join(os.getcwd(), FLAGS.test_dir))
    output_image_name = os.path.join(os.getcwd(), FLAGS.test_dir, str(int(image_num)) +'_result.png')
    cv2.imwrite(output_image_name, cv2.cvtColor(drawing_image, cv2.COLOR_BGR2RGB))
    print(output_image_name + ' saved!')

if __name__ == '__main__':
  app.run(main)