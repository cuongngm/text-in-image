import mlflow
import yaml
import torch
from ultocr.utils.utils_function import create_module
from mlflow.models.signature import infer_signature

def convert_image(image_path):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    img = cv2.imread(image_path)
    h_origin, w_origin = img.shape[:2]
    tmp_img = test_preprocess(img, new_size=736, pad=False) 
    tmp_img = tmp_img.to(device)
    return tmp_img
    
    
def parse_args():
    parser = argparse.ArgumentParser(description='Hyper_parameter')
    parser.add_argument('--path', type=str, default='assets/2.jpg', help='choose gpu device')
    args = parser.parse_args()
    return args
    

with open('config/master.yaml', 'r') as stream:
    cfg = yaml.safe_load(stream)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
recog_model = create_module(cfg['model']['function'])(cfg)
state_dict = torch.load('saved/master_pretrain.pth', map_location=device)['model_state_dict']
recog_model.load_state_dict(state_dict)
print('recog model', recog_model)
recog_model.eval()
recog_model.to(device)

artifact_path = 'master'
# mlflow.set_tracking_uri('sqlite:///mlruns.db')
with mlflow.start_run() as run:
    run_num = run.info.run_id
model_uri = "runs:/{run_id}/{artifact_path}".format(run_id=run_num, artifact_path=artifact_path)
print('uri', model_uri)
mlflow.pytorch.log_model(recog_model, artifact_path)
mlflow.pytorch.save_model(recog_model, artifact_path)
mlflow.register_model(model_uri=model_uri, name=artifact_path)
    
"""
import mlflow.pyfunc
model_name = "dbnet"
model = mlflow.pytorch.load_model(model_uri=f"models:/{model_name}/Production")
opt = parse_args()
image_path = opt.path
test_img = convert_image(image_path)
out = model(test_img)
print(out)

data = {"inputs": test_img.tolist()}
data = json.dumps(data)
response = requests.post(
url="http://172.26.33.199:2514/invocations",
data=data,
headers={"Content-Type": "application/json"},
)

if response.status_code != 200:
raise Exception(
    "Status Code {status_code}. {text}".format(
	status_code=response.status_code, text=response.text
    )
)
print("Prediction: ", response.text)
"""

