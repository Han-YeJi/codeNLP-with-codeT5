import os
import torch
import time
import logging
import utils
import numpy as np
import pandas as pd
from tqdm import tqdm
from config import getConfig
from transformers import AutoTokenizer
from torch.cuda.amp import autocast, grad_scaler
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from network import *

import warnings
warnings.filterwarnings('ignore')

class Trainer():
    def __init__(self, args, save_path):
        super(Trainer, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Logging
        log_file = os.path.join(save_path, 'log.log')
        self.logger = utils.get_root_logger(logger_name='IR', log_level=logging.INFO, log_file=log_file)
        self.logger.info(args)
        self.logger.info(args.tag)        

        # Train, Valid Set load
        train_data = pd.read_csv(f'../data/df_train_{args.fold}fold.csv')
        valid_data = pd.read_csv(f'../data/df_valid_{args.fold}fold.csv')

        # Tokenizing & Encoding
        tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)
        train_encoded = utils.create_encoding(train_data, tokenizer)
        val_encoded = utils.create_encoding(valid_data, tokenizer)

        # Target Data
        train_vals = train_data['similar'].tolist()
        train_vals = torch.Tensor(train_vals)

        valid_vals = valid_data['similar'].tolist()
        valid_vals = torch.Tensor(valid_vals)

        # DataLoader
        train_dataset = TensorDataset(train_encoded['input_ids'], train_encoded['token_type_ids'], train_vals)
        valid_dataset = TensorDataset(val_encoded['input_ids'], val_encoded['token_type_ids'], valid_vals)

        self.train_loader = DataLoader(train_dataset, sampler = RandomSampler(train_dataset), batch_size = args.batch_size)
        self.val_loader = DataLoader(valid_dataset, sampler = SequentialSampler(valid_dataset), batch_size = args.batch_size)
        
        # Network
        self.model = CodeSimModel(args).to(self.device)

        # Loss
        self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer & Scheduler
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr = args.init_lr, weight_decay = args.weight_decay)
        
        iter_per_epoch = len(self.train_loader)
        self.warmup_scheduler = utils.WarmUpLR(self.optimizer, iter_per_epoch * args.warm_epoch)

        if args.scheduler == 'step':
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=args.milestone, gamma=args.lr_factor, verbose=True)
        elif args.scheduler == 'cos':
            tmax = args.tmax # half-cycle 
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max = tmax, eta_min=args.min_lr, verbose=True)
        elif args.scheduler == 'cycle':
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=args.max_lr, steps_per_epoch=iter_per_epoch, epochs=args.epochs)

        load_epoch=0
        if args.re_training_exp is not None:
            pth_files = torch.load(f'./results/{args.re_training_exp}/best_model.pth')
            load_epoch = pth_files['epoch']
            self.model.load_state_dict(pth_files['state_dict'])
            self.optimizer.load_state_dict(pth_files['optimizer'])

            sch_dict = pth_files['scheduler']
            sch_dict['total_steps'] = sch_dict['total_steps'] + args.epochs * iter_per_epoch
            self.scheduler.load_state_dict(sch_dict)

            print(f'Start {load_epoch+1} Epoch Re-training')
            for i in range(args.warm_epoch+1, load_epoch+1):
                self.scheduler.step()

        if args.multi_gpu:
            self.model = nn.DataParallel(self.model).to(self.device)

        # Train / Validate
        best_loss = np.inf
        best_acc = 0
        best_epoch = 0
        early_stopping = 0
        start = time.time()
        for epoch in range(load_epoch+1, args.epochs+1):
            self.epoch = epoch

            if args.scheduler == 'cos':
                if epoch > args.warm_epoch:
                    self.scheduler.step()

            # Training
            self.training(args)

            # Model weight in Multi_GPU or Single GPU
            state_dict = self.model.state_dict()

            # Validation
            val_loss, val_acc = self.validate(args, phase='val')

            # Save models
            if val_loss < best_loss:
                early_stopping = 0
                best_epoch = epoch
                best_loss = val_loss
                best_acc = val_acc
               

                torch.save({'epoch':epoch,
                            'state_dict':state_dict,
                            'optimizer': self.optimizer.state_dict(),
                            'scheduler': self.scheduler.state_dict(),
                    }, os.path.join(save_path, 'best_model.pth'))
                self.logger.info(f'-----------------SAVE:{best_epoch}epoch----------------')
            else:
                early_stopping += 1

            # Early Stopping
            if early_stopping == args.patience:
                break

            if epoch == args.epochs:
                torch.save({'epoch':epoch,
                            'state_dict':state_dict,
                            'optimizer': self.optimizer.state_dict(),
                            'scheduler': self.scheduler.state_dict(),
                    }, os.path.join(save_path, 'last_model.pth'))
                self.logger.info('-----------------SAVE: last epoch----------------')

        self.logger.info(f'\nBest Val Epoch:{best_epoch} | Val Loss:{best_loss:.4f} | Val Acc:{best_acc:.4f}')
        end = time.time()
        self.logger.info(f'Total Process time:{(end - start) / 60:.3f}Minute')

    # Training
    def training(self, args):
        self.model.train()
        total_train_loss = utils.AvgMeter()
        train_acc = 0

        scaler = grad_scaler.GradScaler()
        for i, batch in enumerate(tqdm(self.train_loader)):
            b_input_ids = batch[0].to(self.device)
            b_labels = batch[2].long().to(self.device)
            
            if self.epoch <= args.warm_epoch:
                self.warmup_scheduler.step()

            self.model.zero_grad(set_to_none=True)
            if args.amp:
                with autocast():
                    preds = self.model(b_input_ids)
                    loss = self.criterion(preds, b_labels)
                scaler.scale(loss).backward()

                # Gradient Clipping
                if args.clipping is not None:
                    scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.clipping)

                scaler.step(self.optimizer)
                scaler.update()

            else:
                preds = self.model(b_input_ids)
                loss = self.criterion(preds, b_labels)
                loss.backward()
                
                nn.utils.clip_grad_norm_(self.model.parameters(), args.clipping)
                self.optimizer.step()

            if args.scheduler == 'cycle':
                if self.epoch > args.warm_epoch:
                    self.scheduler.step()

            # Metric
            label_ids = b_labels.cpu().numpy()
            preds = preds.detach().cpu().numpy()
            train_acc += utils.flat_accuracy(preds, label_ids)
            total_train_loss.update(loss.item(), n=len(self.train_loader))

        train_acc /= len(self.train_loader.dataset)
    
        self.logger.info(f'Epoch:[{self.epoch:03d}/{args.epochs:03d}]')
        self.logger.info(f'Train Loss:{total_train_loss.avg:.3f} | Acc:{train_acc:.4f}')
            
    # Validation or Dev
    def validate(self, args, phase='val'):
        self.model.eval()
        with torch.no_grad():
            total_eval_loss = utils.AvgMeter()
            val_acc = 0

            for i, batch in enumerate(self.val_loader):
                b_input_ids = batch[0].to(self.device)
                b_labels = batch[2].long().to(self.device)

                preds = self.model(b_input_ids)
                loss = self.criterion(preds, b_labels)

                # Metric
                total_eval_loss.update(loss.item(), n=len(self.train_loader))
                preds = preds.detach().cpu().numpy()
                label_ids = b_labels.detach().cpu().numpy()
                val_acc += utils.flat_accuracy(preds, label_ids)

            val_acc /= len(self.val_loader.dataset)

            self.logger.info(f'{phase} Loss:{total_eval_loss.avg:.3f} | Acc:{val_acc:.4f}')
            
        return total_eval_loss.avg, val_acc

