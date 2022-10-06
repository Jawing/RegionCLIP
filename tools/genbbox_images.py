"""
This script was used to convert individual annotations from labelme into a cleaner annotation format, containing only relevant information.
"""

import json
import os
import argparse
import glob
import shutil
import bbox_visualizer as bbv
from PIL import Image, ImageOps
import numpy as np

#run with args
# "args": [
#     "./HMWR-001-202204/json/{Elevator_door,Key,Trash_can,Wallet,Other_Wall_outlet,US_Wall_outlet}/",
#     "./bbox_JPEGImages/",
#     "./HMWR-001-202204/raw_images/{Elevator_door,Key,Trash_can,Wallet,Other_Wall_outlet,US_Wall_outlet}/",
# ]

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('input_dir', help='input LabelMe annotated directory')
    parser.add_argument('outimg_dir', help='output image directory')
    parser.add_argument('img_dir', help='image directory')
    args = parser.parse_args()

    print("Generating ground truth bbox images in folder " + args.outimg_dir)

    i = 0

    #number of rotated images
    nvar = 0
    man_rot = 0
    
    #count number of classes
    ed = 0
    tc = 0
    kc = 0
    wl = 0
    wo = 0
    ot = 0
    for file_name in os.listdir(args.input_dir):
        if not '.json' in file_name:
            continue

        with open(args.input_dir +  file_name, 'r') as f:
            anno = json.load(f)
        
        anno.pop('lineColor')
        anno.pop('fillColor')
        anno.pop('imageData')
        anno.pop('version')
        anno.pop('flags')

        #renaming images by foldername
        imgname = anno["imagePath"].split('.')[0]
        img_filenames = glob.glob(f'{args.img_dir}/{imgname}.*')
        #make sure only 1 matching image with extension to build new image name
        assert len(img_filenames) == 1
        foldername = args.input_dir.split('/')[-2]
        new_imgname = f"{foldername}_BasicAI_({i})"
        anno["imagePath"] = f"{new_imgname}.jpg"
        
       
        i += 1

        clean_shapes = []
        bbox_labels = []
        bbox_loc = []
        for shape in anno["shapes"]:
            try:
                #shape.pop("flags")
                shape.pop("shape_type")
                shape.pop("line_color")
                shape.pop("fill_color")
                #shape.pop("group_id")
            except:
                pass
            
            #correct label name format
            if shape['label'] == 'elevator_door':
                shape['label'] = 'elevator doors'
                ed+=1
            elif shape['label'] == 'trash_can':
                shape['label'] = 'Trash can'
                tc+=1
            elif shape['label'] == 'key':
                # shape['label'] = 'keychain'
                kc+=1
            elif shape['label'] == 'wallet':
                wl+=1
            elif shape['label'] in ('Wall_outlet_I', 'Wall_outlet_Other', 'Wall_outlet_AB', 'Wall_outlet_G','Wall_outlet_CDEF'):
                shape['label'] = 'Wall_outlet'
                wo+=1
            else:
                print(shape['label'])
                ot+=1

            pt1, pt2 = shape["points"]
            shape.pop("points")
            x_min = min(pt1[0], pt2[0])
            y_min = min(pt1[1], pt2[1])
            x_max = max(pt1[0], pt2[0])
            y_max = max(pt1[1], pt2[1])
            shape["x_min"] = x_min
            shape["y_min"] = y_min
            shape["x_max"] = x_max
            shape["y_max"] = y_max
            
            clean_shapes.append(shape)

            #append labels to draw
            bbox_labels.append(shape['label'])
            bbox_loc.append([int(shape["x_min"]), int(shape["y_min"]), int(shape["x_max"]), int(shape["y_max"])])
    
        anno.pop("shapes")
        anno["bbox"] = clean_shapes

        #redraw image with bboxes
        im = Image.open(img_filenames[0])

        #resave rotated images
        if (im.height != anno['imageHeight']):
            nvar +=1
            #print(f'not equal heights: {anno["imagePath"]}, {nvar}')      
            im = ImageOps.exif_transpose(im)
            #if still not equal then manually rotate 270 degrees
            if (im.height != anno['imageHeight']):
              im = im.rotate(270,expand=True)
              man_rot += 1
        #       im.save(f"{args.outimg_dir}/{anno['imagePath']}", 'JPEG')
        #     else:
        #       im.save(f"{args.outimg_dir}/{anno['imagePath']}", 'JPEG') 
        assert im.height == anno['imageHeight']


        imar = np.asarray(im)
        imar = bbv.draw_multiple_rectangles(imar, bbox_loc)
        imar = bbv.add_multiple_labels(imar, bbox_labels, bbox_loc, top = True)
        bbox_im = Image.fromarray(np.uint8(imar)).convert('RGB')   

        #transfer bbox image to new location
        if not os.path.exists(args.outimg_dir):
            os.makedirs(args.outimg_dir)
        bbox_im.save(f"{args.outimg_dir}/{anno['imagePath']}")


    #print(f"elevator button: {eb}")
    print(f"Total Images: {i}")
    
    print(f"Trash can: {tc}")
    print(f"elevator doors: {ed}")
    print(f"keychain: {kc}")
    print(f"wallet: {wl}")
    print(f"Wall_outlet: {wo}")
    print(f"other: {ot}")
    print(f"number of rotated images: {nvar}")
    print(f"number of man rotated images: {man_rot}")
    

if __name__ == '__main__':
    main()
