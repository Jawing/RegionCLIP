import io
import base64
import pickle
#import gzip
import os
import numpy as np
import pickle
#from jupyter_dash import JupyterDash
from dash import dcc, html, Input, Output, no_update
import plotly.graph_objects as go

from PIL import Image

from sklearn.manifold import TSNE
from dash import Dash, ALL#, ctx
import dash
import dash_bootstrap_components as dbc

#enter paths for dataset

#dataset_name = 'humanware_test_custom'
dataset_name = 'humanware_test_collected'
#dataset_name = 'humanware_test_basic'
#dataset_name = 'humanware_test_full'
#dataset_name = 'humanware_test_awake'

file_pth_gt = os.path.join(f"./{dataset_name}_all_feats_GT_00.pkl")
if dataset_name in ['humanware_test_full', 'humanware_test_awake']:
    file_pth_rpn = os.path.join(f"./{dataset_name}_all_feats_RPN_05.pkl")
else:
    file_pth_rpn = os.path.join(f"./{dataset_name}_all_feats_RPN_03.pkl")

#get label names
class_set = []
name_to_cat= {}
for i, line in enumerate(open("/home/wanjiz/RegionCLIP/labels_5class.txt").readlines()):
    class_name = line.strip()
    class_set.append(class_name)
    name_to_cat[class_name] = i

# print(class_set)
# print(name_to_cat)

# Helper functions
def np_image_to_base64(im_matrix):
    im = Image.fromarray(im_matrix)
    buffer = io.BytesIO()
    im.save(buffer, format="jpeg")
    encoded_image = base64.b64encode(buffer.getvalue()).decode()
    im_url = "data:image/jpeg;base64, " + encoded_image
    return im_url

def fig_gen_rpn(file_pth):
    # load RPN data for tsne visualize
    
    with open(file_pth, 'rb') as fp:
        all_feats = pickle.load(fp)
        # all_feats['boxes'] = []
        # all_feats['classes'] = []
        # #all_feats['probs'] = []
        # all_feats['feats'] = []
        # all_feats['image_name'] = []
        # all_feats['boxes_img'] = []
        # all_feats['scores'] = []
    region_feats = np.array(all_feats['feats'])
    labels = all_feats['classes']
    images = all_feats['boxes_img']
    scores = all_feats['scores']
    scores_idx_9 = [i for i,v in enumerate(scores) if v >= 0.9]
    scores_idx_8 = [i for i,v in enumerate(scores) if (v < 0.9) and (v >= 0.8)]
    scores_idx_7 = [i for i,v in enumerate(scores) if (v < 0.8) and (v >= 0.7)]
    scores_idx_6 = [i for i,v in enumerate(scores) if (v < 0.7) and (v >= 0.6)]
    scores_idx_5 = [i for i,v in enumerate(scores) if (v < 0.6) and (v >= 0.5)]
    # scores_idx_4 = [i for i,v in enumerate(scores) if (v < 0.5)]


    img_loc = all_feats['image_name'] #change to image_name
    box_loc = all_feats['boxes']
    box_loc = np.array(box_loc).astype(int).tolist()
    print(f'displaying RPN: {len(images)} points')

    gt_label_rpn = ['None']*len(scores)
    
    # t-SNE Outputs a 3 dimensional point for each image
    tsne = TSNE(
        learning_rate= 'auto',
        init = 'pca',
        random_state = 123,
        n_components=3,
        verbose=0,
        perplexity=40,
        n_iter=len(all_feats['classes'])*3) \
        .fit_transform(region_feats)
    
    # Color for each class
    color_map = {
        0: "#0040FF",
        1: "#978BE3",
        2: "#DCC649",
        3: "#25A18E",
        4: "#D81159",
        # 5: "#915C83",
        # 6: "#008000",
        # 7: "#7FFFD4",
        # 8: "#E9D66B",
        # 9: "#007FFF",
    }
    colors = [color_map[l] for l in labels]

    sizes = np.array([4] * len(colors))
    sizes_l = [13] * len(scores_idx_9)
    sizes[scores_idx_9] = sizes_l
    sizes_l = [10] * len(scores_idx_8)
    sizes[scores_idx_8] = sizes_l
    sizes_l = [8] * len(scores_idx_7)
    sizes[scores_idx_7] = sizes_l
    sizes_l = [6] * len(scores_idx_6)
    sizes[scores_idx_6] = sizes_l
    sizes_l = [5] * len(scores_idx_5)
    sizes[scores_idx_5] = sizes_l
    # sizes_l = [4] * len(scores_idx_4)
    # sizes[scores_idx_4] = sizes_l


    fig = go.Figure(data=[go.Scatter3d(
        x=tsne[:, 0],
        y=tsne[:, 1],
        z=tsne[:, 2],
        mode='markers',
        marker=dict(
            size=sizes,
            color=colors,
        )
    )])
    
    fig.update_traces(
        hoverinfo="none",
        hovertemplate=None,
    )
    
    buffer2 = io.StringIO()
    fig.write_html(buffer2)
    html_bytes = buffer2.getvalue().encode()
    encoded = base64.b64encode(html_bytes).decode()

    return sizes,region_feats, fig, encoded, images, labels, gt_label_rpn, img_loc,colors,scores,box_loc

#generate GT figure and values
def fig_gen_gt(file_pth):
    # load GT data for tsne visualize

    with open(file_pth, 'rb') as fp:
        all_feats = pickle.load(fp)
        # all_feats['boxes'] = []
        # all_feats['classes'] = []
        # #all_feats['probs'] = []
        # all_feats['feats'] = []
        # all_feats['image_name'] = []
        # all_feats['boxes_img'] = []
        # all_feats['scores'] = []
    region_feats = np.array(all_feats['feats'])
    labels = all_feats['classes']
    gt_labels = all_feats['gt_classes']
    images = all_feats['boxes_img']
    scores = all_feats['scores']
    scores_idx_9 = [i for i,v in enumerate(scores) if v >= 0.9]
    scores_idx_8 = [i for i,v in enumerate(scores) if (v < 0.9) and (v >= 0.8)]
    scores_idx_7 = [i for i,v in enumerate(scores) if (v < 0.8) and (v >= 0.7)]
    scores_idx_6 = [i for i,v in enumerate(scores) if (v < 0.7) and (v >= 0.6)]
    scores_idx_5 = [i for i,v in enumerate(scores) if (v < 0.6) and (v >= 0.5)]
    # scores_idx_4 = [i for i,v in enumerate(scores) if (v < 0.5)]

    img_loc = all_feats['image_name'] #change to image_name
    box_loc = all_feats['boxes']
    box_loc = np.array(box_loc).astype(int).tolist()

    diff_idx=[]
    diff=0
    for i, (a, b) in enumerate(zip(gt_labels, labels)):
        if a != b:
            diff_idx.append(i)
            diff += 1
    print("number of diff class: ", diff)
    print(f'displaying GT: {len(images)} points')

    # t-SNE Outputs a 3 dimensional point for each image
    tsne = TSNE(
        learning_rate= 'auto',
        init = 'pca',
        random_state = 123,
        n_components=3,
        verbose=0,
        perplexity=40,
        n_iter=len(all_feats['classes'])*3) \
        .fit_transform(region_feats)

    # Color for each class
    color_map = {
        0: "#558cff",
        1: "#BCB4ED",
        2: "#E4D374",
        3: "#40D3BC",
        4: "#F04583",
        # 5: "#915C83",
        # 6: "#008000",
        # 7: "#7FFFD4",
        # 8: "#E9D66B",
        # 9: "#007FFF",
    }
    color_diff = '#000000'
    colors = [color_map[l] for l in labels]
    sizes = np.array([4] * len(colors))
    sizes_l = [13] * len(scores_idx_9)
    sizes[scores_idx_9] = sizes_l
    sizes_l = [10] * len(scores_idx_8)
    sizes[scores_idx_8] = sizes_l
    sizes_l = [8] * len(scores_idx_7)
    sizes[scores_idx_7] = sizes_l
    sizes_l = [6] * len(scores_idx_6)
    sizes[scores_idx_6] = sizes_l
    sizes_l = [5] * len(scores_idx_5)
    sizes[scores_idx_5] = sizes_l
    # sizes_l = [4] * len(scores_idx_4)
    # sizes[scores_idx_4] = sizes_l
    
    # sizes[diff_idx] = 4
    for idx in diff_idx:
        colors[idx] = color_diff

    fig = go.Figure(data=[go.Scatter3d(
        x=tsne[:, 0],
        y=tsne[:, 1],
        z=tsne[:, 2],
        mode='markers',
        marker=dict(
            size=sizes,
            color=colors,
        )
    )])

    fig.update_traces(
        hoverinfo="none",
        hovertemplate=None,
    )


    buffer2 = io.StringIO()
    fig.write_html(buffer2)
    html_bytes = buffer2.getvalue().encode()
    encoded = base64.b64encode(html_bytes).decode()

    return sizes,region_feats, fig, encoded, images, labels, gt_labels, img_loc,colors,scores,box_loc

def save_tsne(saved_path_pkl):
    #get all data values
    sizes_gt,region_feats_gt, fig_gt, encoded_gt, images_gt, labels_gt, gt_labels_gt, img_loc_gt,colors_gt,scores_gt,box_loc_gt = fig_gen_gt(file_pth_gt)
    sizes_rpn,region_feats_rpn, fig_rpn, encoded_rpn, images_rpn, labels_rpn, gt_labels_rpn, img_loc_rpn,colors_rpn,scores_rpn,box_loc_rpn = fig_gen_rpn(file_pth_rpn)
        # with open(saved_path_pkl, 'rb') as fp:
        #     all_feats = pickle.load(fp)

    #generate combined data figure/values
    sizes_c=sizes_gt.tolist()+sizes_rpn.tolist()
    region_feats_c = np.concatenate((region_feats_gt,region_feats_rpn))
    images_c = images_gt + images_rpn
    labels_c = labels_gt + labels_rpn
    gt_labels_c = gt_labels_gt + gt_labels_rpn
    img_loc_c = img_loc_gt + img_loc_rpn
    colors_c = colors_gt + colors_rpn
    scores_c = scores_gt + scores_rpn
    box_loc_c = box_loc_gt + box_loc_rpn
    tsne_c = TSNE(
        learning_rate= 'auto',
        init = 'pca',
        random_state = 123,
        n_components=3,
        verbose=0,
        perplexity=40,
        n_iter=len(labels_c)*3) \
        .fit_transform(region_feats_c)


    #generate all fig
    fig_c = go.Figure(data=[go.Scatter3d(
        x=tsne_c[:, 0],
        y=tsne_c[:, 1],
        z=tsne_c[:, 2],
        mode='markers',
        marker=dict(
            size=sizes_c,
            color=colors_c,
        )
    )])
    fig_c.update_traces(
        hoverinfo="none",
        hovertemplate=None,
    )
    buffer_c = io.StringIO()
    fig_c.write_html(buffer_c)
    html_bytes = buffer_c.getvalue().encode()
    encoded_c = base64.b64encode(html_bytes).decode()



    #save all precomputed data
    save_tsne = (sizes_gt,region_feats_gt, fig_gt, 
                encoded_gt, images_gt, labels_gt, gt_labels_gt, 
                img_loc_gt,colors_gt,scores_gt,box_loc_gt,
                sizes_rpn,region_feats_rpn, fig_rpn, encoded_rpn, 
                images_rpn, labels_rpn, gt_labels_rpn, img_loc_rpn,
                colors_rpn,scores_rpn,box_loc_rpn,
                sizes_c,tsne_c, fig_c, encoded_c, 
                images_c, labels_c, gt_labels_c, 
                img_loc_c,colors_c,scores_c,box_loc_c,
                )
    with open(saved_path_pkl, "wb") as f:
        pickle.dump(save_tsne, f)


saved_path_pkl = os.path.join(f"./vis_regions/{dataset_name}_tsne.pkl")
#save all precomputed data
save_tsne(saved_path_pkl)