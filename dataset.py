# 데이터 전처리 및 batch 단위로 묶는 로직

import tensorflow as tf
import numpy as np

# Reference : https://stackoverflow.com/questions/54567986/python-numpy-remove-empty-zeroes-border-of-3d-array
def bounds_per_dimension(ndarray):
  return map(
    lambda e: range(e.min(), e.max() + 1),
    np.where(ndarray != 0)
  )


def zero_trim_ndarray(ndarray):
  return ndarray[np.ix_(*bounds_per_dimension(ndarray))]


# process ground-truth data for YOLO format
## train.py에서 처리하는 데이터를 원하는 형상의 ground truth로 변경해주는 로직 
## train.py에서 이 function을 임포트해서 실제 Gradient Descent 파라미터를 갱신하기 위한 Dataset을 구성하는데 사용됨
def process_each_ground_truth(original_image,
                              bbox,
                              class_labels,
                              input_width,
                              input_height
                              ):
  """
  Reference:

  ## PASCAL VOC 데이터의 리턴형태 설명
    https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/object_detection/voc.py#L115
    bbox return : (ymin / height, xmin / width, ymax / height, xmax / width) --> 이렇게 나눠서 0~1로 상대좌표로 normalization한 벡터값을 순차적으로 리턴 
  
  Args:
    original_image : (original_height, orignal_width, channel) image tensor --> YOLO 논문에 나와있는 resize 이미지 크기 전 오리지널 이미지 size
    bbox : (max_object_num_in_batch, 4) = (ymin / height, xmin / width, ymax / height, xmax / width) --> normalize된 상대좌표 (절대좌표로 바꿔야함)
    class_labels : (max_object_num_in_batch) = class labels without one-hot-encoding
    input_width : yolo input width ## resize에서 넣는 yolo의 이미지
    input_height : yolo input height

  Returns:
    image: (resized_height, resized_width, channel) image ndarray
    labels: 2-D list [object_num, 5] (xcenter (Absolute Coordinate), ycenter (Absolute Coordinate), w (Absolute Coordinate), h (Absolute Coordinate), class_num)
    object_num: total object number in image  --> 해당 obj의 존재하는 실제 obj 개수
  """
  image = original_image.numpy() ## 텐서타입이라 넘파이로 변경
  image = zero_trim_ndarray(image) ## 0인 부분을 잘라주는 지역함수 / 배치 단위의이미지 사이즈가 고정 이미지 사이즈보다 작으면 패딩으로 늘려서 사용하게 되는데, 그때 패딩된 부분은 검은색(0)으로 변하게 됨. 그래서 검은색 부분은 제거하고 이미지를 취하는 방식

  # set original width height
  original_h = image.shape[0]
  original_w = image.shape[1]

  ## resize된 이미지 대비 전체 오리지널 이미지에 대한 비율
  width_rate = input_width * 1.0 / original_w
  height_rate = input_height * 1.0 / original_h

  ## 다양한 사이즈의 이미지를 yolo input으로 넣기 위해 고정사이즈로 변환
  image = tf.image.resize(image, [input_height, input_width])

  object_num = np.count_nonzero(bbox, axis=0)[0] ## 전체 bbox에서 0이 포함되지 않고 실제 obj가 있는 개수를 count / batch 단위로 처ㅣ리하기때문에 생기는 부분을 파싱
  labels = [[0, 0, 0, 0, 0]] * object_num ## 리턴 값에 대한 초기 initialization / 5개의 지표가 들어간 벡터가 obj num만큼 생성 -> 여기선 초기화
  
  ## obj num만큼 돌면서 bbox return 값은 상대좌표로 normalization이 되어있어서 original 값을 곱해서 절대좌표로 변경해서 절대좌표 값을 가져오고 
  for i in range(object_num):
    xmin = bbox[i][1] * original_w
    ymin = bbox[i][0] * original_h
    xmax = bbox[i][3] * original_w
    ymax = bbox[i][2] * original_h

    ## 실제 클래스 레이블 가져오기
    class_num = class_labels[i]

    ## Pascal VOC 정답 데이터셋은 xmin,ymin,xmax,ymax에서 left top , right bottom의 꼭짓점 좌표로 정답이 표현되어 있는데, YOLO모델은 bbox의 center x좌표, center y좌표, box width, box height로 데이터를 표기
    ## 이런 불일치를 막기 위해 xmin,ymin,xmax,ymax 꼭짓점 좌표를 center의 width,height 정보로 변경을 해주는 과정이 필요함
    xcenter = (xmin + xmax) * 1.0 / 2 * width_rate  # xmin, xmax의 1/2 (중앙점)에서 width rate를 곱해서 '절대좌표'로 변경
    ycenter = (ymin + ymax) * 1.0 / 2 * height_rate

    box_w = (xmax - xmin) * width_rate # xmax-xmin이 width이 될거고 거기에 width_rate를 곱해서 절대좌표로 변경된 yolo포맷 형태
    box_h = (ymax - ymin) * height_rate

    ## 절대좌표로 변경된 yolo 포맷형태의 5차원의 벡터를 labels부분에 obj 개수만큼 할당
    labels[i] = [xcenter, ycenter, box_w, box_h, class_num]

  return [image.numpy(), labels, object_num]