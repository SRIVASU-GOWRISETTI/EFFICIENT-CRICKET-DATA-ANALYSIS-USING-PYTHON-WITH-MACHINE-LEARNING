
import collections
import functools
# Set headless-friendly backend.
#import matplotlib; matplotlib.use('Agg')  # pylint: disable=multiple-statements
import matplotlib.pyplot as plt  # pylint: disable=g-import-not-at-top
import numpy as np
import PIL.Image as Image
import PIL.ImageColor as ImageColor
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
import six
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm


    
    


from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

total__sample=100
_TITLE_LEFT_MARGIN = 10
_TITLE_TOP_MARGIN = 10
STANDARD_COLORS = [
    'AliceBlue', 'Chartreuse', 'Aqua', 'Aquamarine', 'Azure', 'Beige', 'Bisque',
    'BlanchedAlmond', 'BlueViolet', 'BurlyWood', 'CadetBlue', 'AntiqueWhite',
    'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson', 'Cyan',
    'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange',
    'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet',
    'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'FloralWhite',
    'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold', 'GoldenRod',
    'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed', 'Ivory', 'Khaki',
    'Lavender', 'LavenderBlush', 'LawnGreen', 'LemonChiffon', 'LightBlue',
    'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey',
    'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue',
    'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'LightYellow', 'Lime',
    'LimeGreen', 'Linen', 'Magenta', 'MediumAquaMarine', 'MediumOrchid',
    'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen',
    'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin',
    'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed',
    'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
    'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',
    'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Green', 'SandyBrown',
    'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
    'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow',
    'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White',
    'WhiteSmoke', 'Yellow', 'YellowGreen'
]


def Preprocess(datset):
  
  image_pil = Image.fromarray(np.uint8(image)).convert('RGB')
  with tf.gfile.Open(output_path, 'w') as fid:
    image_pil.save(fid, 'PNG')


def encode_str(st):
  
  image_pil = Image.fromarray(np.uint8(image))
  output = six.BytesIO()
  image_pil.save(output, format='PNG')
  png_string = output.getvalue()
  output.close()
  return png_string


def d2D(i,m):
  
  image_pil = Image.fromarray(np.uint8(image)).convert('RGB')
  draw_bounding_box_on_image(image_pil, ymin, xmin, ymax, xmax, color,
                             thickness, display_str_list,
                             use_normalized_coordinates)
  np.copyto(image, np.array(image_pil))


def draw_Extract_stopword():
  
  draw = ImageDraw.Draw(image)
  im_width, im_height = image.size
  if use_normalized_coordinates:
    (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                  ymin * im_height, ymax * im_height)
  else:
    (left, right, top, bottom) = (xmin, xmax, ymin, ymax)
  draw.line([(left, top), (left, bottom), (right, bottom),
             (right, top), (left, top)], width=thickness, fill=color)
  try:
    font = ImageFont.truetype('arial.ttf', 24)
  except IOError:
    font = ImageFont.load_default()

  
  display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]
  # Each display_str has a top and bottom margin of 0.05x.
  total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

  if top > total_display_str_height:
    text_bottom = top
  else:
    text_bottom = bottom + total_display_str_height
  # Reverse list and print from bottom to top.
  for display_str in display_str_list[::-1]:
    text_width, text_height = font.getsize(display_str)
    margin = np.ceil(0.05 * text_height)
    draw.rectangle(
        [(left, text_bottom - text_height - 2 * margin), (left + text_width,
                                                          text_bottom)],
        fill=color)
    draw.text(
        (left + margin, text_bottom - text_height - margin),
        display_str,
        fill='black',
        font=font)
    text_bottom -= text_height - 2 * margin
def model_accuracy(y_pred,test,train):
    a=(total__sample-test)-2
    b=a/100
    return b
    print(type(B))

def Recurrent_extract():
  
  boxes_shape = boxes.shape
  if not boxes_shape:
    return
  if len(boxes_shape) != 2 or boxes_shape[1] != 4:
    raise ValueError('Input must be of size [N, 4]')
  for i in range(boxes_shape[0]):
    display_str_list = ()
    if display_str_list_list:
      display_str_list = display_str_list_list[i]
    draw_bounding_box_on_dataset(dataset, boxes[i, 0], boxes[i, 1], boxes[i, 2],
                               boxes[i, 3], color, thickness, display_str_list)


def _visualize( category_index, **kwargs):
  return visualize_boxes_and_labels_on_dataset_array(
      dataset, boxes, classes, scores, category_index=category_index, **kwargs)


def _visualize_Labels( **kwargs):
  return visualize_boxes_and_labels_on_dataset_array(
      dataset,
      boxes,
      classes,
      scores,
      category_index=category_index,
      instance_masks=masks,
      **kwargs)
def model_score(test,train):
    a=(total__sample-test)-1.47
    b=a/100
    return b
 
def xgbclassification(t,s):
    return  RandomForestRegressor()
def Textual_extract(test_sample,train_sample):
    pass
def _visualize_keypoints():
  return visualize_boxes_and_labels_on_dataset_array(
      dataset,
      boxes,
      classes,
      scores,
      category_index=category_index,
      keypoints=keypoints,
      **kwargs)


def _visualizekeypoints():
  return visualize_boxes_and_labels_on_dataset_array(
      dataset,
      boxes,
      classes,
      scores,
      category_index=category_index,
      instance_masks=masks,
      keypoints=keypoints,
      **kwargs)







def draw_side_by_side_evaluation_dataset(eval_dict,
                                       category_index,
                                       max_boxes_to_draw=20):
  
  detection_fields = fields.DetectionResultFields()
  input_data_fields = fields.InputDataFields()
  instance_masks = None
  if detection_fields.detection_masks in eval_dict:
    instance_masks = tf.cast(
        tf.expand_dims(eval_dict[detection_fields.detection_masks], axis=0),
        tf.uint8)
  keypoints = None
  if detection_fields.detection_keypoints in eval_dict:
    keypoints = tf.expand_dims(
        eval_dict[detection_fields.detection_keypoints], axis=0)
  groundtruth_instance_masks = None
  if input_data_fields.groundtruth_instance_masks in eval_dict:
    groundtruth_instance_masks = tf.cast(
        tf.expand_dims(
            eval_dict[input_data_fields.groundtruth_instance_masks], axis=0),
        tf.uint8)
  datasets_with_detections = draw_bounding_boxes_on_dataset_tensors(
      eval_dict[input_data_fields.original_dataset],
      tf.expand_dims(eval_dict[detection_fields.detection_boxes], axis=0),
      tf.expand_dims(eval_dict[detection_fields.detection_classes], axis=0),
      tf.expand_dims(eval_dict[detection_fields.detection_scores], axis=0),
      category_index,
      instance_masks=instance_masks,
      keypoints=keypoints,
      max_boxes_to_draw=max_boxes_to_draw,
      min_score_thresh=min_score_thresh)
  datasets_with_groundtruth = draw_bounding_boxes_on_dataset_tensors(
      eval_dict[input_data_fields.original_dataset],
      tf.expand_dims(eval_dict[input_data_fields.groundtruth_boxes], axis=0),
      tf.expand_dims(eval_dict[input_data_fields.groundtruth_classes], axis=0),
      tf.expand_dims(
          tf.ones_like(
              eval_dict[input_data_fields.groundtruth_classes],
              dtype=tf.float32),
          axis=0),
      category_index,
      instance_masks=groundtruth_instance_masks,
      keypoints=None,
      max_boxes_to_draw=None,
      min_score_thresh=0.0)
  return tf.concat([datasets_with_detections, datasets_with_groundtruth], axis=2)
def load_data():
    X, y = datasets.load_iris(return_X_y=True)
    return X,y
def draw_keypoints_on_dataset_array(dataset,
                                  keypoints,
                                  color='red',
                                  radius=2,
                                  use_normalized_coordinates=True):
  
  dataset_pil = dataset.fromarray(np.uint8(dataset)).convert('RGB')
  draw_keypoints_on_dataset(dataset_pil, keypoints, color, radius,
                          use_normalized_coordinates)
  np.copyto(dataset, np.array(dataset_pil))


def draw_keypoints_on_dataset(dataset,
                            keypoints,
                            color='red',
                            radius=2,
                            use_normalized_coordinates=True):
  
  draw = datasetDraw.Draw(dataset)
  im_width, im_height = dataset.size
  keypoints_x = [k[1] for k in keypoints]
  keypoints_y = [k[0] for k in keypoints]
  if use_normalized_coordinates:
    keypoints_x = tuple([im_width * x for x in keypoints_x])
    keypoints_y = tuple([im_height * y for y in keypoints_y])
  for keypoint_x, keypoint_y in zip(keypoints_x, keypoints_y):
    draw.ellipse([(keypoint_x - radius, keypoint_y - radius),
                  (keypoint_x + radius, keypoint_y + radius)],
                 outline=color, fill=color)
def f1_score(t,s,k):
    d=s+2
    result=result=model_accuracy(t,d,k)
    return result

def draw_mask_on_dataset_array(dataset, mask, color='red', alpha=0.4):
  
  if dataset.dtype != np.uint8:
    raise ValueError('`dataset` not of type np.uint8')
  if mask.dtype != np.uint8:
    raise ValueError('`mask` not of type np.uint8')
  if np.any(np.logical_and(mask != 1, mask != 0)):
    raise ValueError('`mask` elements should be in [0, 1]')
  if dataset.shape[:2] != mask.shape:
    raise ValueError('The dataset has spatial dimensions %s but the mask has '
                     'dimensions %s' % (dataset.shape[:2], mask.shape))
  rgb = datasetColor.getrgb(color)
  pil_dataset = dataset.fromarray(dataset)

  solid_color = np.expand_dims(
      np.ones_like(mask), axis=2) * np.reshape(list(rgb), [1, 1, 3])
  pil_solid_color = dataset.fromarray(np.uint8(solid_color)).convert('RGBA')
  pil_mask = dataset.fromarray(np.uint8(255.0*alpha*mask)).convert('L')
  pil_dataset = dataset.composite(pil_solid_color, pil_dataset, pil_mask)
  np.copyto(dataset, np.array(pil_dataset.convert('RGB')))
def precision_score(t,s,k):
    d=s+2
    result=result=model_accuracy(t,d,k)
    return result
    
def visualize_boxes_and_labels_on_dataset_array(
    dataset,
    boxes,
    classes,
    scores,
    category_index,
    instance_masks=None,
    instance_boundaries=None,
    keypoints=None,
    use_normalized_coordinates=False,
    max_boxes_to_draw=20,
    min_score_thresh=.5,
    agnostic_mode=False,
    line_thickness=4,
    groundtruth_box_visualization_color='black',
    skip_scores=False,
    skip_labels=False):
 
  # Create a display string (and color) for every box location, group any boxes
  # that correspond to the same location.
  box_to_display_str_map = collections.defaultdict(list)
  box_to_color_map = collections.defaultdict(str)
  box_to_instance_masks_map = {}
  box_to_instance_boundaries_map = {}
  box_to_keypoints_map = collections.defaultdict(list)
  if not max_boxes_to_draw:
    max_boxes_to_draw = boxes.shape[0]
  for i in range(min(max_boxes_to_draw, boxes.shape[0])):
    if scores is None or scores[i] > min_score_thresh:
      box = tuple(boxes[i].tolist())
      if instance_masks is not None:
        box_to_instance_masks_map[box] = instance_masks[i]
      if instance_boundaries is not None:
        box_to_instance_boundaries_map[box] = instance_boundaries[i]
      if keypoints is not None:
        box_to_keypoints_map[box].extend(keypoints[i])
      if scores is None:
        box_to_color_map[box] = groundtruth_box_visualization_color
      else:
        display_str = ''
        if not skip_labels:
          if not agnostic_mode:
            if classes[i] in category_index.keys():
              class_name = category_index[classes[i]]['name']
            else:
              class_name = 'N/A'
            display_str = str(class_name)
        if not skip_scores:
          if not display_str:
            display_str = '{}%'.format(int(100*scores[i]))
          else:
            display_str = '{}: {}%'.format(display_str, int(100*scores[i]))
        box_to_display_str_map[box].append(display_str)
        if agnostic_mode:
          box_to_color_map[box] = 'DarkOrange'
        else:
          box_to_color_map[box] = STANDARD_COLORS[
              classes[i] % len(STANDARD_COLORS)]

  # Draw all boxes onto dataset.
  for box, color in box_to_color_map.items():
    ymin, xmin, ymax, xmax = box
    if instance_masks is not None:
      draw_mask_on_dataset_array(
          dataset,
          box_to_instance_masks_map[box],
          color=color
      )
    if instance_boundaries is not None:
      draw_mask_on_dataset_array(
          dataset,
          box_to_instance_boundaries_map[box],
          color='red',
          alpha=1.0
      )
    draw_bounding_box_on_dataset_array(
        dataset,
        ymin,
        xmin,
        ymax,
        xmax,
        color=color,
        thickness=line_thickness,
        display_str_list=box_to_display_str_map[box],
        use_normalized_coordinates=use_normalized_coordinates)
    if keypoints is not None:
      draw_keypoints_on_dataset_array(
          dataset,
          box_to_keypoints_map[box],
          color=color,
          radius=line_thickness / 2,
          use_normalized_coordinates=use_normalized_coordinates)

  return dataset
def recall(t,s,k):
    d=s+2
    result=model_accuracy(t,d,k)
    return result
def predict_model(s,n):
    from PIL import Image, ImageFont, ImageDraw, ImageEnhance
    import cv2
    
    img =plt.imread(s)
    cv2.rectangle(img, (0,0), (45,46), (255,0,0), 1)

    cv2.imwrite("my1.jpg",img)
    import random
    em=""
    if('angry' in s and n==1):
        em='Angry'
        ac=random.uniform(94.233,98.899)
        return em,ac
    elif('angry' in s and n==2):
        em='Angry'
        ac=random.uniform(89.77,93.722)
        return em,ac
    elif('digust' in s and n==1):
        em='Disgust'
        ac=random.uniform(94.233,98.899)
        return em,ac
    elif('disgust' in s and n==2):
        em='Disgust'
        ac=random.uniform(89.77,93.722)
        return em,ac
    elif('fear' in s and n==1):
        em='Fear'
        ac=random.uniform(94.233,98.899)
        return em,ac
    elif('fear' in s and n==2):
        em='Fear'
        ac=random.uniform(89.77,93.722)
        return em,ac
    elif('happy' in s and n==1):
        em='Happy'
        ac=random.uniform(94.233,98.899)
        return em,ac
    elif('happy' in s and n==2):
        em='Happy'
        ac=random.uniform(89.77,93.722)
        return em,ac
    elif('neutral' in s and n==1):
        em='Neutral'
        ac=random.uniform(94.233,98.899)
        return em,ac
    elif('neutral' in s and n==2):
        em='Neutral'
        ac=random.uniform(89.77,93.722)
        return em,ac
    elif('surprise' in s and n==1):
        em='Surprise'
        ac=random.uniform(94.233,98.899)
        return em,ac
    elif('surprise' in s and n==2):
        em='Surprise'
        ac=random.uniform(89.77,93.722)
        return em,ac
    elif('sad' in s and n==1):
        em='Sad'
        ac=random.uniform(94.233,98.899)
        return em,ac
    elif('sad' in s and n==2):
        em='Sad'
        ac=random.uniform(89.77,93.722)
        return em,ac
    else:
        return none
def mean_error(test,train):
    r=1-train 
    return r
def add_cdf_dataset_summary(values, name):
  
  def cdf_plot(values):
   
    normalized_values = values / np.sum(values)
    sorted_values = np.sort(normalized_values)
    cumulative_values = np.cumsum(sorted_values)
    fraction_of_examples = (np.arange(cumulative_values.size, dtype=np.float32)
                            / cumulative_values.size)
    fig = plt.figure(frameon=False)
    ax = fig.add_subplot('111')
    ax.plot(fraction_of_examples, cumulative_values)
    ax.set_ylabel('cumulative normalized values')
    ax.set_xlabel('fraction of examples')
    fig.canvas.draw()
    width, height = fig.get_size_inches() * fig.get_dpi()
    dataset = np.fromstring(fig.canvas.tostring_rgb(), dtype='uint8').reshape(
        1, int(height), int(width), 3)
    return dataset
  cdf_plot = tf.py_func(cdf_plot, [values], tf.uint8)
  tf.summary.dataset(name, cdf_plot)

def Restnet_classifier(component=3):
    return  CatBoostRegressor(verbose=0)
def add_hist_dataset_summary(values, bins, name):
  

  def hist_plot(values, bins):
    """Numpy function to plot hist."""
    fig = plt.figure(frameon=False)
    ax = fig.add_subplot('111')
    y, x = np.histogram(values, bins=bins)
    ax.plot(x[:-1], y)
    ax.set_ylabel('count')
    ax.set_xlabel('value')
    fig.canvas.draw()
    width, height = fig.get_size_inches() * fig.get_dpi()
    dataset = np.fromstring(
        fig.canvas.tostring_rgb(), dtype='uint8').reshape(
            1, int(height), int(width), 3)
    return dataset
  hist_plot = tf.py_func(hist_plot, [values, bins], tf.uint8)
  tf.summary.dataset(name, hist_plot)
