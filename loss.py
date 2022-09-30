import tensorflow as tf
import numpy as np
from utils import iou # utils.py에서 정의해놓은 2개의 bbox의 IOU를 계산해주는 함수 

def yolo_loss(predict,
              labels,
              each_object_num,
              num_classes,
              boxes_per_cell,
              cell_size,
              input_width,
              input_height,
              coord_scale,
              object_scale,
              noobject_scale,
              class_scale
              ):
  '''
  Args:
    predict: 3 - D tensor [cell_size, cell_size, num_classes + 5 * boxes_per_cell]
    labels: 2-D list [object_num, 5] (xcenter (Absolute coordinate), ycenter (Absolute coordinate), w (Absolute coordinate), h (Absolute coordinate), class_num)
    each_object_num: each_object number in image  ## obj별로 loss값을 구하기 때문에 여러 obj가 있을 시 어느 obj인지
    num_classes: number of classes
    boxes_per_cell: number of prediction boxes per each cell
    cell_size: each cell size
    input_width : input width of original image
    input_height : input_height of original image

   ## 합산 loss fucntion을 계산할때 앞에 곱해지는 람다 coefficient
    coord_scale : coefficient for coordinate loss
    object_scale : coefficient for object loss
    noobject_scale : coefficient for noobject loss
    class_scale : coefficient for class loss

  Returns:
    total_loss: coord_loss  + object_loss + noobject_loss + class_loss
    coord_loss
    object_loss
    noobject_loss
    class_loss
  '''

  # parse only coordinate vector
  ## predict_boxes : gird cell내에서 하나의 vector : ex) pascal 앞의 20차원 벡터는 클래스 레이블 / 그 다음 2 : confidence 예측(bbox수만큼) / 나머지 8개 벡터가 첫번째 , 두번째 bbox의 x,y,w,h
  predict_boxes = predict[:, :, num_classes + boxes_per_cell:] # 2개의 bbox의 x,y,w,h : position vector / 7 x 7 x 8
  predict_boxes = tf.reshape(predict_boxes, [cell_size, cell_size, boxes_per_cell, 4]) # -> 7x7x2x4


  # prediction : absolute coordinate
  pred_xcenter = predict_boxes[:, :, :, 0] # confidence 2 지나서 x
  pred_ycenter = predict_boxes[:, :, :, 1] # confidence 2 지나서 y
 
  ## 절대좌표로 표현 -> minimum과 maximum으로 제한
  pred_sqrt_w = tf.sqrt(tf.minimum(input_width * 1.0, tf.maximum(0.0, predict_boxes[:, :, :, 2]))) # confidence 2 지나서 w
  pred_sqrt_h = tf.sqrt(tf.minimum(input_height * 1.0, tf.maximum(0.0, predict_boxes[:, :, :, 3]))) # confidence 2 지나서 h
  pred_sqrt_w = tf.cast(pred_sqrt_w, tf.float32)
  pred_sqrt_h = tf.cast(pred_sqrt_h, tf.float32)

  # parse label
  labels = np.array(labels)
  labels = labels.astype('float32')
  label = labels[each_object_num, :] ## each_obj_num이 인자값으로 하나씩 처리하는 형태기 때문에 전체 obj에서 대응되는 obj 정답 벡터값 하나만 가져옴
  xcenter = label[0] ## 정답 센터 x 좌표
  ycenter = label[1] ## 정답 센터 y 좌표
  sqrt_w = tf.sqrt(label[2]) ## 정답 박스 width
  sqrt_h = tf.sqrt(label[3]) ## 정답 박스 height

  # calulate iou between ground-truth and predictions
  ## 전체 box에 대한 IOU를 찾음
  iou_predict_truth = iou(predict_boxes, label[0:4])


  # find best box mask
  I = iou_predict_truth
  max_I = tf.reduce_max(I, 2, keepdims=True)
  best_box_mask = tf.cast((I >= max_I), tf.float32) ## binary mask : 최대면 1 아니면 0


  # set object_loss information
  ## obj가 있는 cell의 confidence 지정
  ## for 수식 라인 3,4
  C = iou_predict_truth ## obj가 있는 cell의 정답은 정답(ground truth)과 prediction 값 간의 IOU
  pred_C = predict[:, :, num_classes:num_classes + boxes_per_cell] ## 전체 pred vector에서 confidence를 나타내는 부분은 30차원 벡터에서 num_class만큼 건너뛰고 confidence값이 저장된 2개의 벡터차원을 슬라이싱


  # set class_loss information
  ## for 수식 라인 5
  P = tf.one_hot(tf.cast(label[4], tf.int32), num_classes, dtype=tf.float32) ## num_classes 개수만큼 바꿈 '정답'
  pred_P = predict[:, :, 0:num_classes] ## 30차원에서 앞에서 num_classes만큼이 yolo가 'predict'한 softmax regression vector 값


  # find object exists cell mask
  ## 전체 cell 사이즈 만큼의 vector값을 가지고 있다가 해당 cell이 obj가 실제로 존재하는 cell이면 1의 마스크맵을 갖는 형태로 바꿈
  object_exists_cell = np.zeros([cell_size, cell_size, 1])
  ## obj가 있는지 판단하는 것은 ycenter, xcenter : ycenter normalized한 값이 0.4라고 하면 4x4 셀크기에서 int(0.4*4) = 1
  object_exists_cell_i, object_exists_cell_j = int(cell_size * ycenter / input_height), int(cell_size * xcenter / input_width)
  object_exists_cell[object_exists_cell_i][object_exists_cell_j] = 1


  # set coord_loss
  ## 현재 pred_xcenter, xcenter, pred_ycenter, ycenter 값등이 절대좌표로 바뀌어 있기 때문에 계산할 때는 xcenter, ycenter에 대해서는 input_width, cell_size로 더 나눠줘서 cell 내의 상대좌표로 normalize를 한 번 더 해줘야 됨
  ## w,h같은 경우도 prediction 값이 전부 절대좌표로 되어 있는데, w,h만큼 나눠서 이미지 전체에 대해 normalize된 값으로 한 번 변경해준 다음에 실제 loss term을 계산해야됨.
  ## object_exists_cell * best_box_mask : 1objij 구현
  ## 4줄이 loss function 수식 맨 위 1,2번째 loss 라인
  coord_loss = (tf.nn.l2_loss(object_exists_cell * best_box_mask * (pred_xcenter - xcenter) / (input_width / cell_size)) + ## grid cell 차원의 normalize
                tf.nn.l2_loss(object_exists_cell * best_box_mask * (pred_ycenter - ycenter) / (input_height / cell_size)) +
                tf.nn.l2_loss(object_exists_cell * best_box_mask * (pred_sqrt_w - sqrt_w)) / input_width + ## 전체 이미지간의 normalize
                tf.nn.l2_loss(object_exists_cell * best_box_mask * (pred_sqrt_h - sqrt_h)) / input_height ) \
               * coord_scale ## ycoord 람다 coefficient의 가중치 (원논문 :5)

  # object_loss
  ## obj가 있는 cell의 confidence 예측
  ## 수식 3번째
  ## object_exists_cell * best_box_mask : 1objij
  object_loss = tf.nn.l2_loss(object_exists_cell * best_box_mask * (pred_C - C)) * object_scale ## object_scale : 우리가 새로 생성한 람다 coefficient -> 자유도를 높이기 위해 

  # noobject_loss
  ## obj가 없는 cell의 confidence 예측
  ## 수식 4번째
  ## object_exists_cell이 obj가 있는 셀은 1, 아닌 셀은 0인 마스크였으니깐 -1을 하면 obj가 없는 셀은 0이 됨
  ## obj가 없는 셀이 더 많을 것이기 때문에 그에 대한 가중치를 낮추기 위해 1에서 작은 값인 0.5를 곱해줌 (coefficient) 
  noobject_loss = tf.nn.l2_loss((1 - object_exists_cell) * (pred_C)) * noobject_scale

  # class loss
  ## 수식 5번째
  ## YOLO 모델이 예측한 class prediciton 값이랑 정답 class prediction 값간의 차이
  ## obj가 존재하는 cell에서만 계산
  class_loss = tf.nn.l2_loss(object_exists_cell * (pred_P - P)) * class_scale ## 여기서 class_scale은 원논문에선 없었지만 자유도를 위해서 만듬

  # sum every loss
  total_loss = coord_loss + object_loss + noobject_loss + class_loss

  return total_loss, coord_loss, object_loss, noobject_loss, class_loss