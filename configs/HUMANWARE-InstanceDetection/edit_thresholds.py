import yaml

model_config_file = "./server_config.yaml"

def set_thresholds(config_file,iou_threshold=0.2, conf_threshold=0.6):
    with open(config_file, "r") as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    cfg['MODEL']['ROI_HEADS']['NMS_THRESH_TEST'] = iou_threshold
    cfg['MODEL']['ROI_HEADS']['SCORE_THRESH_TEST'] = conf_threshold
    with open(config_file, "w") as ymlfile:
        yaml.safe_dump(cfg,ymlfile,default_flow_style=False)

set_thresholds(model_config_file)