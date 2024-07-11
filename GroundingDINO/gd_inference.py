'''
Ruiyang Zhang, ruiyang.061x@gmail.com, 2024.7.11

GroundingDINO inference on image set of autonomous driving dataset.
'''


from groundingdino.util.inference import load_model, load_image, predict, annotate
import cv2
import glob
from tqdm import tqdm
import pickle
import torch
from torchvision.ops import box_convert


model = load_model("groundingdino/config/GroundingDINO_SwinT_OGC.py", "weights/groundingdino_swint_ogc.pth")
IMAGE_PATH = "data/lyft/MODEST/lyft_kitti_format/training/image_2/*"
TEXT_PROMPT = "car . pedestrian . animal . other_vehicle . bus . motorcycle . truck . emergency_vehicle . bicycle ."
BOX_TRESHOLD = 0.3
TEXT_TRESHOLD = 0.25
OUTPUT_PATH = 'lyft_2Dbox_img/'
LABEL_OUTPUT_PATH = 'lyft_2Dbox_txt/'

image_paths = glob.glob(IMAGE_PATH)
image_paths = sorted(image_paths)
print(f'Image number: {len(image_paths)}.')
results = {}
for image_path in tqdm(image_paths):
    image_source, image = load_image(image_path)
    # GroundingDINO inference on image.
    boxes, logits, phrases = predict(model=model, image=image, caption=TEXT_PROMPT, box_threshold=BOX_TRESHOLD, text_threshold=TEXT_TRESHOLD)
    results[image_path] = {'boxes': boxes, 'logits': logits, 'phrases': phrases}
    # Save visualization images.
    annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
    cv2.imwrite(OUTPUT_PATH + image_path.split('/')[-1], annotated_frame)
# Save result pkl.
pickle.dump(results, open('lyft_2Dbox_pkl/gd_swint_bt_0.3_tt_0.25.pkl', 'wb'))
print('Inference finished.')

for image_path, result in tqdm(results.items()):
    boxes = result['boxes']
    logits = result['logits']
    phrases = result['phrases']
    # Convert format of 2Dbox from 'cxcywh' to 'xyxy'
    h, w = 1024, 1224
    boxes = boxes * torch.Tensor([w, h, w, h])
    xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
    N = xyxy.shape[0]
    # Save 2Dbox into kitti label format
    # FGR generates image-based pseudo labels based on these 2Dboxes
    f = open(LABEL_OUTPUT_PATH + image_path.split('/')[-1].split('.')[0] + '.txt', 'a')
    for i in range(N):
        f.writelines('Dynamic 0 0 0 ' + str(xyxy[i][0]) + ' ' + str(xyxy[i][1]) + ' ' + str(xyxy[i][2]) + ' ' + str(xyxy[i][3]) + ' ' + '0 0 0 0 0 0 0 ' + '\n')
    f.close()
print('Format convert finished.')