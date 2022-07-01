import mlflow
import yaml
import torch
from ultocr.utils.utils_function import create_module
from mlflow.models.signature import infer_signature


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
