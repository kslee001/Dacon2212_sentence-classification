import os
import re
import random
import pandas as pd
import numpy as np
import warnings
from typing import Optional
from tqdm import tqdm as tq
from sklearn import preprocessing
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
from transformers import logging

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="1"
warnings.filterwarnings(action='ignore')
logging.set_verbosity_error()


today = "1222_new"


# Configuration
CFG = {
    'SEED':1203,
    'EPOCHS':50,
    'LEARNING_RATE':0.05,
    'BATCH_SIZE':512,
    'PLM':"klue/roberta-large",
    'MAX_LEN':128,
    'TIME_LEN':16, 
    
    'DIM_FEEDFORWARD':256,
    'DROPOUT':0.35,
    
    'WEIGHTS':[
    "/home/gyuseonglee/workspace/outputs/full_SentenceClassifier_33_120_dict.pt",
    
    "/home/gyuseonglee/workspace/outputs/full_SentenceClassifier_35_100_dict.pt",
    
    "/home/gyuseonglee/workspace/outputs/full_SentenceClassifier_42_100_dict.pt",
    
    "/home/gyuseonglee/workspace/outputs/full_SentenceClassifier_97_120_dict.pt",
    
    "/home/gyuseonglee/workspace/outputs/full_SentenceClassifier_98_120_dict.pt",
    
    "/home/gyuseonglee/workspace/outputs/full_SentenceClassifier_364_120_dict.pt",
    
    "/home/gyuseonglee/workspace/outputs/SentenceClassifier_317_150_dict.pt",
    
    "/home/gyuseonglee/workspace/outputs/SentenceClassifier_910_146_dict.pt",
    
    "/home/gyuseonglee/workspace/outputs/SentenceClassifier_1203_149_dict.pt",
    
    "/home/gyuseonglee/workspace/outputs/full_SentenceClassifier_1_68_dict.pt",
    
    "/home/gyuseonglee/workspace/outputs/full_SentenceClassifier_2_70_dict.pt",
    
    "/home/gyuseonglee/workspace/outputs/full_SentenceClassifier_3_69_dict.pt",
    
    "/home/gyuseonglee/workspace/outputs/full_SentenceClassifier_4_70_dict.pt",
    
    
],
    'OUTPUT_FOLDER':"/home/gyuseonglee/workspace"
    
}

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
seed_everything(CFG['SEED'])
folder = os.getcwd() + '/open'
test = 'test.csv'
submit = 'sample_submission.csv'
test = pd.read_csv(folder + '/' + test)
submit = pd.read_csv(folder + '/' + submit)

# preprocessing (same as train data)
test['문장'] = test['문장'].apply(lambda x: re.sub('[^a-zA-Zㄱ-ㅎ가-힣]', ' ',x))


# dataset, data loader
class CustomDataset(Dataset):
    def __init__(self, data, max_len, plm=CFG["PLM"], infer=False):
        self.text      = data['문장'].tolist()
        self.tokenizer = AutoTokenizer.from_pretrained(plm)
        self.max_len   = max_len
        self.infer     = infer
        
        if self.infer == False:
            type1 = data['유형']=='사실형'
            type2 = data['유형']=='추론형'
            type3 = data['유형']=='예측형'
            type4 = data['유형']=='대화형'
            
            sent1 = data['극성']=='긍정'
            sent2 = data['극성']=='부정'
            sent3 = data['극성']=='미정'
            
            tense1 = data['시제']=='현재'
            tense2 = data['시제']=='과거'
            tense3 = data['시제']=='미래'
            
            certainty1 = data['확실성']=='확실'
            certainty2 = data['확실성']=='불확실'
            
            self.y1 = np.where(type1, 0, 
                    np.where(type2, 1, 
                    np.where(type3, 2, 3))).tolist()
            self.y2 = np.where(sent1, 0,
                    np.where(sent2, 1, 2)).tolist()
            self.y3 = np.where(tense1, 0, 
                    np.where(tense2, 1, 2)).tolist()
            self.y4 = np.where(certainty1, 0, 1).tolist()
                        
    def __len__(self):
        return len(self.text)
    
    def __getitem__(self, idx):
        text = self.text[idx]
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding = 'max_length',
            truncation = True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        x_ids, x_attn = [encoding['input_ids'].flatten(), encoding['attention_mask'].flatten()]
        if self.infer == False:
            y1 = self.y1[idx]
            y2 = self.y2[idx]
            y3 = self.y3[idx]
            y4 = self.y4[idx]
            
            ys = torch.tensor([y1,y2,y3,y4]) 
            return x_ids, x_attn, ys        
        else:
            return x_ids, x_attn
            
def get_dataloader(data, batch_size, max_len=CFG["MAX_LEN"], plm=CFG["PLM"], infer=False, shuffle=True):
    return DataLoader(
        dataset=CustomDataset(data=data, max_len=max_len, plm=plm, infer=infer),
        num_workers=4,
        batch_size=batch_size,
        shuffle=shuffle
    )
    
# model
class SentenceClassifier(torch.nn.Module):
    seed_everything(CFG['SEED']) 
    def __init__(
        self,
        plm,
        time_length = 8,
        max_length = 128, 
        num_layers = 4,
        dim_model = 1024,
        num_heads = 6,
        dim_feedforward = 768,
        dropout = 0.08,
        ):
        super().__init__()
        # base settings
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.activate = torch.nn.SiLU()
        self.dim_model = dim_model
        self.num_heads = num_heads
        self.dim_feedforward = dim_feedforward
        self.dropout = torch.nn.Dropout(p=0.1)
        self.time_length = time_length
        self.max_length = max_length
        self.plm = plm
        
        self.linear_dim = dim_model
        
        self.feature_extractor = AutoModel.from_pretrained(self.plm)
        modules = [self.feature_extractor.embeddings, *self.feature_extractor.encoder.layer[:-5]] #Replace 5 by what you want
        for module in modules:
            for param in module.parameters():
                param.requires_grad = False

        self.linears = torch.nn.ModuleList(
            [self.get_cls(self.linear_dim, 4), self.get_cls(self.linear_dim, 3), self.get_cls(self.linear_dim, 3), self.get_cls(self.linear_dim, 2)]
        )
        
    
    def forward(self, x_ids, x_attn):
        x = self.feature_extractor(x_ids, x_attn).pooler_output
        
        y0 = self.linears[0](x)
        y1 = self.linears[1](x)
        y2 = self.linears[2](x)
        y3 = self.linears[3](x)
        
        
        return y0, y1, y2, y3


    def get_cls(self, input_dim=1024*4, output_dim=1024):
        return torch.nn.Sequential(
            torch.nn.Linear(input_dim, input_dim//4),
            torch.nn.LayerNorm(input_dim//4),
            torch.nn.Dropout(p=0.1),
            self.activate,
            torch.nn.Linear(input_dim//4, output_dim),
        )
       
       
# inference 
device = "cuda" if torch.cuda.is_available() else 'cpu'
test_loader = get_dataloader(
    data=test,
    batch_size=CFG['BATCH_SIZE'],
    max_len=CFG['MAX_LEN'],
    plm = CFG['PLM'],
    infer=True,
    shuffle=False
    )
model = SentenceClassifier(        
    time_length = CFG["TIME_LEN"],
    max_length = CFG["MAX_LEN"], 
    num_layers = 2,
    dim_model = 1024,
    num_heads = 8,
    dim_feedforward = CFG['DIM_FEEDFORWARD'],
    dropout = CFG['DROPOUT'],
    plm = CFG['PLM']
    )

# weights = [
#     "/home/n7/gyuseong/workspace/current/sentence/outputs/SentenceClassifier_317_49_dict.pt",
#     "/home/n7/gyuseong/workspace/current/sentence/outputs/SentenceClassifier_910_49_dict.pt",
#     "/home/n7/gyuseong/workspace/current/sentence/outputs/SentenceClassifier_1203_49_dict.pt",
# ]
weights = CFG['WEIGHTS']


def inference(model, test_loader, device):
    def step(batch, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        x_ids, x_attn = batch
        x_ids, x_attn = x_ids.to(device), x_attn.to(device)
        yhat0, yhat1, yhat2, yhat3 = model(x_ids, x_attn)
        yhat0, yhat1, yhat2, yhat3 = yhat0.float(), yhat1.float(), yhat2.float(), yhat3.float()
        return yhat0, yhat1, yhat2, yhat3
    
    model.to(device)
    model.eval()
    type_probs, polarity_probs, tense_probs, certainty_probs = [], [], [], []
    type_preds, polarity_preds, tense_preds, certainty_preds = [], [], [], []
    with torch.no_grad():
        for batch in tq(test_loader):
            type_logit, polarity_logit, tense_logit, certainty_logit = step(batch, model, device)                        
            type_probs += type_logit.detach().cpu().numpy().tolist()
            polarity_probs += type_logit.detach().cpu().numpy().tolist()
            tense_probs += type_logit.detach().cpu().numpy().tolist()
            certainty_probs += type_logit.detach().cpu().numpy().tolist()
            
            # type_preds += type_logit.argmax(1).detach().cpu().numpy().tolist()
            # polarity_preds += polarity_logit.argmax(1).detach().cpu().numpy().tolist()
            # tense_preds += tense_logit.argmax(1).detach().cpu().numpy().tolist()
            # certainty_preds += certainty_logit.argmax(1).detach().cpu().numpy().tolist()
    return np.array(type_probs), np.array(polarity_probs), np.array(tense_probs), np.array(certainty_probs)

def __inference(model, test_loader, device):
    def step(batch, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        x_ids, x_attn = batch
        x_ids, x_attn = x_ids.to(device), x_attn.to(device)
        yhat0, yhat1, yhat2, yhat3 = model(x_ids, x_attn)
        yhat0, yhat1, yhat2, yhat3 = yhat0.float(), yhat1.float(), yhat2.float(), yhat3.float()
        return yhat0, yhat1, yhat2, yhat3
    
    model.to(device)
    model.eval()
    
    type_probs, polarity_probs, tense_probs, certainty_probs = [], [], [], []
    type_preds, polarity_preds, tense_preds, certainty_preds = [], [], [], []

    with torch.no_grad():
        for batch in tq(iter(test_loader)):
            type_logit, polarity_logit, tense_logit, certainty_logit = step(batch, model, device)                        
            
            type_probs = type_logit.detach().cpu().numpy().tolist()
            polarity_probs = polarity_logit.detach().cpu().numpy().tolist()
            tense_probs = tense_logit.detach().cpu().numpy().tolist()
            certainty_probs = certainty_logit.detach().cpu().numpy().tolist()
            
            type_preds.extend(type_probs)
            polarity_preds.extend(polarity_probs)
            tense_preds.extend(tense_probs)
            certainty_preds.extend(certainty_probs)

    return np.array(type_preds), np.array(polarity_preds), np.array(tense_preds), np.array(certainty_preds
                                                                                           )

type_preds = []
polarity_preds = []
tense_preds = []
certainty_preds = []
for idx in tq(range(len(weights))):
    model.load_state_dict(torch.load(weights[idx]))
    model = model.to(device)
    model.eval()
    type_probs, polarity_probs, tense_probs, certainty_probs = __inference(model, test_loader, device)
    torch.cuda.empty_cache()
    type_preds.append(type_probs)
    polarity_preds.append(polarity_probs)
    tense_preds.append(tense_probs)
    certainty_preds.append(certainty_probs)
    
res1 = np.zeros_like(type_probs)
res2 = np.zeros_like(polarity_probs)
res3 = np.zeros_like(tense_probs)
res4 = np.zeros_like(certainty_probs)

for i in range(len(weights)):
    res1 += type_preds[i]
    res2 += polarity_preds[i]
    res3 += tense_preds[i]
    res4 += certainty_preds[i]
    
res1 = res1.argmax(1)
res2 = res2.argmax(1)
res3 = res3.argmax(1)
res4 = res4.argmax(1)
    
result = pd.DataFrame([res1, res2, res3, res4]).T
result.to_csv("확인1.csv", encoding="cp949")

result.columns = ['유형', '극성', '시제', '확실성']
result['유형'] = result['유형'].astype(int)
result['극성'] = result['극성'].astype(int)
result['시제'] = result['시제'].astype(int)
result['확실성'] = result['확실성'].astype(int)

result['유형'] = np.where(
    result['유형'] == 0, '사실형', np.where(
        result['유형'] == 1, '추론형', np.where(
            result['유형'] == 2, '예측형', '대화형'
    )))
result['극성'] = np.where(
    result['극성'] == 0, '긍정', np.where(
        result['극성'] == 1, '부정', '미정'
    ))
result['시제'] = np.where(
    result['시제'] == 0, '현재', np.where(
        result['시제'] == 1, '과거', '미래'
    ))
result['확실성'] = np.where(
    result['확실성'] == 0, '확실', '불확실'
)
result['label'] = result['유형'] + '-' + result['극성'] + '-' + result['시제'] + '-' + result['확실성']
submit['label'] = result['label']
submit.to_csv(f"submission_{today}_voting.csv", index=False)
