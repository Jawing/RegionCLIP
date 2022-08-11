import time
from subprocess import call
inference_script = './server_inference.sh'
inst_file = './output/inference/lvis_instances_results.json'
import json

#get classes
classes = []
cls_file = './classes.txt'
with open(cls_file, 'r') as f:
    for line in f:
        classes.append(line.strip())
#add names to inst_file
def add_category_name(annos):
    for anno in annos:
        anno["category_name"] = classes[anno["category_id"]-1]
    return annos

start_inf = time.time()
#get inference (can be slow)
call(['bash',inference_script])


with open(inst_file, 'r') as f:
    annos = json.load(f)

#compute prediction and output real json in response
jsonResponse = add_category_name(annos)

print(json.dumps(jsonResponse))

end_inf = time.time()
print(f'Inference time: {end_inf - start_inf}')