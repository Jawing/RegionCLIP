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



saved_path_pkl = os.path.join(f"./vis_regions/{dataset_name}_tsne.pkl")

#load all precomputed data
with open(saved_path_pkl, 'rb') as fp:
    (sizes_gt,region_feats_gt, fig_gt, 
            encoded_gt, images_gt, labels_gt, gt_labels_gt, 
            img_loc_gt,colors_gt,scores_gt,box_loc_gt,
            sizes_rpn,region_feats_rpn, fig_rpn, encoded_rpn, 
            images_rpn, labels_rpn, gt_labels_rpn, img_loc_rpn,
            colors_rpn,scores_rpn,box_loc_rpn,
            sizes_c,tsne_c, fig_c, encoded_c, 
            images_c, labels_c, gt_labels_c, 
            img_loc_c,colors_c,scores_c,box_loc_c,
            ) = pickle.load(fp)

gt_len = len(labels_gt)
#generate gt only fig
fig_c_gt = go.Figure(data=[go.Scatter3d(
    x=tsne_c[:gt_len, 0],
    y=tsne_c[:gt_len, 1],
    z=tsne_c[:gt_len, 2],
    mode='markers',
    marker=dict(
        size=sizes_c[:gt_len],
        color=colors_c[:gt_len],
    )
)])
fig_c_gt.update_traces(
    hoverinfo="none",
    hovertemplate=None,
)
buffer_c_gt = io.StringIO()
fig_c_gt.write_html(buffer_c_gt)
html_bytes = buffer_c_gt.getvalue().encode()
encoded_c_gt = base64.b64encode(html_bytes).decode()

#generate rpn only fig
fig_c_rpn = go.Figure(data=[go.Scatter3d(
    x=tsne_c[gt_len:, 0],
    y=tsne_c[gt_len:, 1],
    z=tsne_c[gt_len:, 2],
    mode='markers',
    marker=dict(
        size=sizes_c[gt_len:],
        color=colors_c[gt_len:],
    )
)])
fig_c_rpn.update_traces(
    hoverinfo="none",
    hovertemplate=None,
)
buffer_c_rpn = io.StringIO()
fig_c_rpn.write_html(buffer_c_rpn)
html_bytes = buffer_c_rpn.getvalue().encode()
encoded_c_rpn = base64.b64encode(html_bytes).decode()


# main dash app
app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])
row_content = [
        dbc.Col([dbc.Button('GT', id='btn-nclicks-1', n_clicks=0, color="primary"),
        html.Div([dbc.Button('Both', id='btn-c', n_clicks=0, color="secondary")], id='div-c',style= {'display': 'none'})
        ],width="2"),
        dbc.Col(html.Div(id='graph-5-cont',
        className="container",
        children=dcc.Graph(id="graph-5", figure=fig_rpn, clear_on_unhover=True, style={'width': '95vh', 'height': '95vh'}),
        ),width="auto"),
        dbc.Col(html.A(
        dbc.Button("Download as HTML", id='html-btn',color="info", className="me-1"), 
        id="download",
        href="data:text/html;base64," + encoded_rpn,
        download="tsne_graph.html",),width="2")
    ]
app.layout = html.Div([
    dbc.Row(row_content,
    align="center",
    justify="center"),
    dcc.Tooltip(id="graph-tooltip-5", direction='right')
]
)

#callback for switching RPN, GT, ALL tsne displays
@app.callback(
    Output("btn-c", "children"),
    Output("btn-nclicks-1", "children"),
    Output("graph-5", "figure"),
    Output('download','href'),
    Output("div-c", "style"),
    Input('btn-nclicks-1', 'n_clicks'),
    Input('btn-c', 'n_clicks'),
)
def displayClick(n_clicks,n_clicks_c):
    style = {'display': 'none'}
    msg_c = no_update
    if n_clicks % 3 == 1:
        msg = "GT"
        figure = fig_gt
        href = "data:text/html;base64," + encoded_gt
    elif n_clicks % 3 == 2:
        msg = "ALL"
        style = {'display': 'block'}
        if n_clicks_c % 3 == 1:
            msg_c = "GT"
            figure = fig_c_gt
            href = "data:text/html;base64," + encoded_c_gt
        elif n_clicks_c % 3 == 2:
            msg_c = "RPN"
            figure = fig_c_rpn
            href = "data:text/html;base64," + encoded_c_rpn
        else:
            msg_c = "Both"
            figure = fig_c
            href = "data:text/html;base64," + encoded_c
    else:
        msg = "RPN"
        figure = fig_rpn
        href = "data:text/html;base64," + encoded_rpn
    return msg_c,msg, figure, href, style

@app.callback(
    Output("graph-tooltip-5", "show"),
    Output("graph-tooltip-5", "bbox"),
    Output("graph-tooltip-5", "children"),
    Input("graph-5", "hoverData"),
    Input("btn-nclicks-1", "children"),
    Input("btn-c", "children"),
)
def display_hover(hoverData,Button,Button_c):
    if hoverData is None:
        return False, no_update, no_update

    # demo only shows the first point, but other points may also be available
    hover_data = hoverData["points"][0]
    bbox = hover_data["bbox"]
    num = hover_data["pointNumber"]
    #print('button:', Button)

    if Button == 'RPN':
        im_matrix = images_rpn[num]
        im_url = np_image_to_base64(im_matrix)
        children = [
            html.Div([
                html.Img(
                    src=im_url,
                    style={"width": "80%", 'display': 'block', 'margin': '0 auto'},
                ),
                html.P(class_set[labels_rpn[num]], style={'color':f'{colors_rpn[num]}','font-weight':'bold','font-size': '0.8em','font-family':'Sans-serif',
                'text-align': 'center', "width": "100%", 'display': 'block', 'margin': '0 auto'}),
                html.P([f"Score: {scores_rpn[num]:.03f}",
                html.Br(),f"\n{img_loc_rpn[num]}",html.Br(),f"\nXYWH: {box_loc_rpn[num]}"] , style={'font-size': '0.8em','font-family':'Sans-serif',
                'text-align': 'center', "width": "100%", 'display': 'block', 'margin': '0 auto'})
            ])
        ]
    elif Button == 'GT':
        im_matrix = images_gt[num]
        im_url = np_image_to_base64(im_matrix)
        children = [
            html.Div([
                html.Img(
                    src=im_url,
                    style={"width": "80%", 'display': 'block', 'margin': '0 auto'},
                ),
                html.P(class_set[labels_gt[num]], style={'color':f'{colors_gt[num]}','font-weight':'bold','font-size': '0.8em','font-family':'Sans-serif',
                'text-align': 'center', "width": "100%", 'display': 'block', 'margin': '0 auto'}),
                html.P([f'GT: {class_set[gt_labels_gt[num]]}',html.Br(),f"\nScore: {scores_gt[num]:.03f}",
                html.Br(),f"\n{img_loc_gt[num]}",html.Br(),f"\nXYWH: {box_loc_gt[num]}"] , style={'font-size': '0.8em','font-family':'Sans-serif',
                'text-align': 'center', "width": "100%", 'display': 'block', 'margin': '0 auto'})
            ])
        ]
    elif Button == 'ALL':
        if Button_c == "RPN":
            gt_text = "None"
            im_matrix = images_rpn[num]
            im_url = np_image_to_base64(im_matrix)
            children = [
                html.Div([
                    html.Img(
                        src=im_url,
                        style={"width": "80%", 'display': 'block', 'margin': '0 auto'},
                    ),
                    html.P(class_set[labels_rpn[num]], style={'color':f'{colors_rpn[num]}','font-weight':'bold','font-size': '0.8em','font-family':'Sans-serif',
                    'text-align': 'center', "width": "100%", 'display': 'block', 'margin': '0 auto'}),
                    html.P([f'GT: {gt_text}',html.Br(),f"\nScore: {scores_rpn[num]:.03f}",
                    html.Br(),f"\n{img_loc_rpn[num]}",html.Br(),f"\nXYWH: {box_loc_rpn[num]}"] , style={'font-size': '0.8em','font-family':'Sans-serif',
                    'text-align': 'center', "width": "100%", 'display': 'block', 'margin': '0 auto'})
                ])
            ]
        else:
            im_matrix = images_c[num]
            im_url = np_image_to_base64(im_matrix)
            if gt_labels_c[num] == "None":
                gt_text = "None"
            else:
                gt_text = class_set[gt_labels_c[num]]
        
            children = [
                html.Div([
                    html.Img(
                        src=im_url,
                        style={"width": "80%", 'display': 'block', 'margin': '0 auto'},
                    ),
                    html.P(class_set[labels_c[num]], style={'color':f'{colors_c[num]}','font-weight':'bold','font-size': '0.8em','font-family':'Sans-serif',
                    'text-align': 'center', "width": "100%", 'display': 'block', 'margin': '0 auto'}),
                    html.P([f'GT: {gt_text}',html.Br(),f"\nScore: {scores_c[num]:.03f}",
                    html.Br(),f"\n{img_loc_c[num]}",html.Br(),f"\nXYWH: {box_loc_c[num]}"] , style={'font-size': '0.8em','font-family':'Sans-serif',
                    'text-align': 'center', "width": "100%", 'display': 'block', 'margin': '0 auto'})
                ])
            ]
    return True, bbox, children

if __name__ == "__main__":
    app.run_server(debug=True,port=8050)

