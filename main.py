import os
import re
import random
import pandas as pd
import numpy as np
import warnings
from typing import Optional
# from dm import dm as 
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
# os.environ["CUDA_VISIBLE_DEVICES"]="0"
warnings.filterwarnings(action='ignore')
logging.set_verbosity_error()


# Configuration
CFG = {
    'SEED':42,
    'EPOCHS':61,
    'LEARNING_RATE':0.05,
    'BATCH_SIZE':256,
    'PLM':"klue/roberta-large",
    'MAX_LEN':128,
    'TIME_LEN':16, 
    
    'DIM_FEEDFORWARD':256,
    'DROPOUT':0.35,
    
    
    'OUTPUT_FOLDER':"/home/gyuseonglee/workspace"
}

# Seed 고정
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    

if __name__ == "__main__":

    print("\nfull42_con\n")

    seed_everything(CFG['SEED']) 
    # load data
    folder = os.getcwd() + '/open'
    train = 'train.csv'
    test  = 'test.csv'
    submit = 'sample_submission.csv'
    train = pd.read_csv(folder + '/' + train)


    # preprocessing
    train = pd.concat([train, train], 0) # duplicates
    train['문장'] = train['문장'].apply(lambda x: re.sub('[^a-zA-Zㄱ-ㅎ가-힣]', ' ',x))
    train['문장'] = train['문장'].str.replace("  ", " ")

    # train-valid split
    # train, valid = train_test_split(train, test_size=0.15, random_state=CFG['SEED'])
    test = pd.read_csv(folder + '/' + test)
    submit = pd.read_csv(folder + '/' + submit)

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
        

    # Focal Loss
    def label_to_one_hot_label(
        labels: torch.Tensor,
        num_classes: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        eps: float = 1e-6,
        ignore_index=255,
    ) -> torch.Tensor:
        shape = labels.shape
        one_hot = torch.zeros((shape[0], ignore_index+1) + shape[1:], device=device, dtype=dtype)
        one_hot = one_hot.scatter_(1, labels.unsqueeze(1), 1.0) + eps
        ret = torch.split(one_hot, [num_classes, ignore_index+1-num_classes], dim=1)[0]
        
        return ret

    # https://github.com/zhezh/focalloss/blob/master/focalloss.py
    def focal_loss(input, target, alpha, gamma, reduction, eps, ignore_index):
        if not isinstance(input, torch.Tensor):
            raise TypeError(f"Input type is not a torch.Tensor. Got {type(input)}")

        if not len(input.shape) >= 2:
            raise ValueError(f"Invalid input shape, we expect BxCx*. Got: {input.shape}")

        if input.size(0) != target.size(0):
            raise ValueError(f'Expected input batch_size ({input.size(0)}) to match target batch_size ({target.size(0)}).')
        
        n = input.size(0) # B
        out_size = (n,) + input.size()[2:]
        if target.size()[1:] != input.size()[2:]:
            raise ValueError(f'Expected target size {out_size}, got {target.size()}')
        if not input.device == target.device:
            raise ValueError(f"input and target must be in the same device. Got: {input.device} and {target.device}")
        if isinstance(alpha, float):
            pass
        elif isinstance(alpha, np.ndarray):
            alpha = torch.from_numpy(alpha)
            alpha = alpha.view(-1, len(alpha), 1, 1).expand_as(input)
        elif isinstance(alpha, torch.Tensor):
            alpha = alpha.view(-1, len(alpha), 1, 1).expand_as(input)       

        input_soft = F.softmax(input, dim=1) + eps
        target_one_hot = label_to_one_hot_label(target.long(), num_classes=input.shape[1], device=input.device, dtype=input.dtype, ignore_index=ignore_index)
        weight = torch.pow(1.0 - input_soft, gamma)
        focal = -alpha * weight * torch.log(input_soft)
        loss_tmp = torch.sum(target_one_hot * focal, dim=1)

        if reduction == 'none':
            loss = loss_tmp
        elif reduction == 'mean':
            loss = torch.mean(loss_tmp)
        elif reduction == 'sum':
            loss = torch.sum(loss_tmp)
        else:
            raise NotImplementedError(f"Invalid reduction mode: {reduction}")
        return loss

    class FocalLoss(nn.Module):
        def __init__(self, alpha =0.01 , gamma = 2.0, reduction = 'mean', eps = 1e-8, ignore_index=30):
            super().__init__()
            self.alpha = alpha
            self.gamma = gamma
            self.reduction = reduction
            self.eps = eps
            self.ignore_index = ignore_index

        def forward(self, input, target):
            return focal_loss(input, target, self.alpha, self.gamma, self.reduction, self.eps, self.ignore_index)


    # Optimizer
    class SAM(torch.optim.Optimizer):
        def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
            assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

            defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
            super(SAM, self).__init__(params, defaults)

            self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
            self.param_groups = self.base_optimizer.param_groups
            self.defaults.update(self.base_optimizer.defaults)

        @torch.no_grad()
        def first_step(self, zero_grad=False):
            grad_norm = self._grad_norm()
            for group in self.param_groups:
                scale = group["rho"] / (grad_norm + 1e-12)

                for p in group["params"]:
                    if p.grad is None: continue
                    self.state[p]["old_p"] = p.data.clone()
                    e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                    p.add_(e_w)  # climb to the local maximum "w + e(w)"

            if zero_grad: self.zero_grad()

        @torch.no_grad()
        def second_step(self, zero_grad=False):
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is None: continue
                    p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

            self.base_optimizer.step()  # do the actual "sharpness-aware" update

            if zero_grad: self.zero_grad()

        @torch.no_grad()
        def step(self, closure=None):
            assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
            closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

            self.first_step(zero_grad=True)
            closure()
            self.second_step()

        def _grad_norm(self):
            shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
            norm = torch.norm(
                        torch.stack([
                            ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                            for group in self.param_groups for p in group["params"]
                            if p.grad is not None
                        ]),
                        p=2
                )
            return norm

        def load_state_dict(self, state_dict):
            super().load_state_dict(state_dict)
            self.base_optimizer.param_groups = self.param_groups
            
            


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
            
    # train
    device = "cuda" if torch.cuda.is_available() else 'cpu'
    train_loader = get_dataloader(data=train,batch_size=CFG['BATCH_SIZE'], infer=False)
    valid_loader = get_dataloader(data=train,batch_size=CFG['BATCH_SIZE'], infer=False)
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

    model = model.to(device)
    model.load_state_dict(torch.load("/home/gyuseonglee/workspace/outputs/full_SentenceClassifier_42_39_dict.pt"))
    base_optimizer = torch.optim.SGD
    optimizer = SAM(model.parameters(), base_optimizer, lr=CFG['LEARNING_RATE'], momentum=0.9)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1,threshold_mode='abs',min_lr=1e-8, verbose=False)
    criterions = [FocalLoss().to(device) for _ in range(4)]

    # parallel
    # model = torch.nn.parallel.DataParallel(model)

    def train(model, optimizer, criterions, train_loader, valid_loader, scheduler, device):    
        best_f1_score_mean = 0
        best_model = model
        best_epoch = 0
        
        def step(batch, model, optimizer, criterions, device='cuda' if torch.cuda.is_available() else 'cpu'):
            x_ids, x_attn, y = batch
            optimizer.zero_grad()
            x_ids, x_attn, y = x_ids.to(device), x_attn.to(device), y.to(device).float()
            yhat0, yhat1, yhat2, yhat3 = model(x_ids, x_attn)
            yhat0, yhat1, yhat2, yhat3 = yhat0.float(), yhat1.float(), yhat2.float(), yhat3.float()

            loss1 = criterions[0](yhat0, y[:, 0])
            loss2 = criterions[1](yhat1, y[:, 1])
            loss3 = criterions[2](yhat2, y[:, 2])
            loss4 = criterions[3](yhat3, y[:, 3])
            loss = loss1+loss2+loss3+loss4
            
            return loss, yhat0, yhat1, yhat2, yhat3
            
            
        for epoch in range(1, CFG['EPOCHS']+1):
            model.train()
            train_loss=[]
            
            # train
            for batch in (train_loader):
                #first step
                loss, _, _, _, _ = step(batch, model, optimizer, criterions, device='cuda' if torch.cuda.is_available() else 'cpu')
                loss.backward()
                optimizer.first_step(zero_grad=True)
                
                # second step
                loss, _, _, _, _ = step(batch, model, optimizer, criterions, device='cuda' if torch.cuda.is_available() else 'cpu')
                loss.backward()
                optimizer.second_step(zero_grad=True)
                
                # optimizer.step()
                train_loss.append(loss.item())

            # valid
            val_loss = []
            type_preds, polarity_preds, tense_preds, certainty_preds = [], [], [], []
            type_labels, polarity_labels, tense_labels, certainty_labels = [], [], [], []
            
            with torch.no_grad():
                for batch in (valid_loader):
                    # optimizer.zero_grad()
                    y = batch[2]
                    loss, yhat0, yhat1, yhat2, yhat3 = step(batch, model, optimizer, criterions, device='cuda' if torch.cuda.is_available() else 'cpu')
                    val_loss.append(loss.item())
                    
                    type_preds += yhat0.argmax(1).detach().cpu().numpy().tolist()
                    type_labels += y[:, 0].detach().cpu().numpy().tolist()
                    
                    polarity_preds += yhat1.argmax(1).detach().cpu().numpy().tolist()
                    polarity_labels += y[:, 1].detach().cpu().numpy().tolist()
                    
                    tense_preds += yhat2.argmax(1).detach().cpu().numpy().tolist()
                    tense_labels += y[:, 2].detach().cpu().numpy().tolist()
                    
                    certainty_preds += yhat3.argmax(1).detach().cpu().numpy().tolist()
                    certainty_labels += y[:, 3].detach().cpu().numpy().tolist()
            
            type_f1 = f1_score(type_labels, type_preds, average='weighted')
            polarity_f1 = f1_score(polarity_labels, polarity_preds, average='weighted')
            tense_f1 = f1_score(tense_labels, tense_preds, average='weighted')
            certainty_f1 = f1_score(certainty_labels, certainty_preds, average='weighted')
            
            f1_score_mean = np.mean([type_f1, polarity_f1, tense_f1, certainty_f1])
            print(f1_score_mean)
            if f1_score_mean >= best_f1_score_mean:
                best_f1_score_mean = f1_score_mean
                best_model = model
                best_epoch = epoch
            
            
            print(f"-- EPOCH {epoch} --")
            print("train_loss : ", round(np.mean(val_loss), 4))
            print("type_f1  : ", round(type_f1, 4))
            print("pol_f1   : ", round(polarity_f1, 4)) 
            print("tense_f1 : ", round(tense_f1, 4))
            print("cert_f1  : ", round(certainty_f1, 4))
            
        return best_model, best_epoch
    
    
    def save_model(best_model, best_epoch, output_folder, model_name):
        model_states_name = "full_" + model_name + "_" + str(CFG['SEED']) + "_" + str(best_epoch) + "_dict.pt"
        model_name        = "full_" + model_name + "_" + str(CFG['SEED']) + "_" + str(best_epoch) + ".pt"
        torch.save(
            best_model, 
            output_folder + "/" + model_name
        )
        torch.save(
            best_model.state_dict(),
            output_folder + "/" + model_states_name
        )
        
    best_model, best_epoch = train(model, optimizer, criterions, train_loader, valid_loader, scheduler, device)
    output_folder = CFG['OUTPUT_FOLDER']
    save_model(best_model, 39+61, output_folder, 'SentenceClassifier') 