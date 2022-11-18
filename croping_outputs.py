import cv2
import json

root_dir = "/data/Object_detection/data/Indoor_objectDetection/Indoor_object_recognition_oldFilename/AllObjects/custom/"
output_dir = "output/inference/coco_instances_results.json"
annotation_file  = "datasets/humanware/annotations/instances_test_custom.json"
save_folder = "/data//Object_detection/data/Indoor_objectDetection/Indoor_object_recognition_oldFilename/AllObjects/crops_trashcan"

f= open(output_dir)
pred_data = json.load(f)
f.close()

f = open(annotation_file)
annotation_data = json.load(f)

print(annotation_data.keys())


images_dict={}
annotations_dict={}
for sample in annotation_data['images']:
    images_dict.update({sample['id']: sample['file_name']})

for sample in annotation_data['annotations']:
    if sample['category_id'] == 3 :
        annotations_dict.update({sample['image_id']:sample['bbox']})
for idx, (img_id, bbox) in enumerate(annotations_dict.items()):
    img = cv2.imread(root_dir + '/' + images_dict[img_id])
    x, y , w, h = [int(x) for x in bbox]
    img_crop = img[y:y+h, x:x+w]
    cv2.imwrite("{}/gt_{}.jpg".format(save_folder, idx) ,img_crop)

idx = 0
for pred_sample in pred_data:
    img = cv2.imread(root_dir + '/' + images_dict[pred_sample['image_id']])
    bbox = pred_sample['bbox']
    x, y , w, h = [int(x) for x in bbox]
    cat = pred_sample['category_id']
    score = pred_sample['score']
    if cat == 3 and score >0.3:
        #bbox xywh
        img_crop = img[y:y+h, x:x+w]
        cv2.imwrite("{}/{}.jpg".format(save_folder, idx) ,img_crop)
        idx+=1

