import torch
from torch.utils.data import Dataset, DataLoader
from ultocr.utils.utils_function import create_module
from ultocr.loader.recognition.translate import LabelTransformer
from ultocr.model.recognition.master import greedy_decode_with_probability


class MasterReg:
    def __init__(self, cfg):
        self.img_w = cfg['dataset']['img_w']
        self.img_h = cfg['dataset']['img_h']
        self.transform = cfg['dataset']['transformer']
        self.batch = 16
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load('../checkpoint/master_best.pth', map_location=self.device)
        model = create_module(cfg['functional']['master'])(cfg)
        self.model = model.to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

    def recognition(self, list_img):
        text_dataset = TextInference(list_img, transform=self.transform)
        text_loader = DataLoader(text_dataset, batch_size=self.batch, shuffle=False, num_workers=1, drop_last=False)
        pred_results = []
        for step_idx, data_item in enumerate(text_loader):
            images = data_item
            with torch.no_grad():
                images = images.to(self.device)
                outputs, probs = greedy_decode_with_probability(self.model, images, LabelTransformer.max_length,
                                                                LabelTransformer.SOS, LabelTransformer.EOS,
                                                                LabelTransformer.PAD, result_device=self.device,
                                                                is_padding=True)
            for idx, (pred, prob) in enumerate(zip(outputs[:, 1:], probs)):
                pred_text = ''
                pred_score_list = []
                for i in range(len(pred)):
                    if pred[i] == LabelTransformer.EOS:
                        pred_score_list.append(prob[i])
                        break
                    if pred[i] == LabelTransformer.UNK:
                        continue
                    decoded_char = LabelTransformer.decode(pred[i])
                    pred_text += decoded_char
                    pred_score_list.append(prob[i])
                pred_score = sum(pred_score_list) / len(pred_score_list)
                if pred_score > 0.85:
                    pred_results.append(pred_text)
                else:
                    pred_results.append('')
        return pred_results


class TextInference(Dataset):
    def __init__(self, all_img, transform=None):
        self.all_img = all_img
        self.transform = transform

    def __getitem__(self, idx):
        img = self.all_img[idx]
        if self.transform is not None:
            img, width_ratio = self.transform(img)
            return img

    def __len__(self):
        return len(self.all_img)
