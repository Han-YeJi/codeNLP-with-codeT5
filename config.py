import os
import argparse

def getConfig():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_num', default='1', type=str)
    parser.add_argument('--tag', default='Default', type=str, help='tag')
    
    parser.add_argument('--fold', default='0', type=str)
    parser.add_argument('--model_path', type=str, default='results/')
    parser.add_argument('--pretrained_model', type=str, default='microsoft/graphcodebert-base')
    parser.add_argument('--drop_path_rate', type=float, default=0.2)
    
    # SSL
    parser.add_argument('--use_ssl_df', type=str, default=None, help='SSL DataFrame File Name')

    # Training parameter settings
    ## Base Parameter
    parser.add_argument('--batch_size', default=64, type=int)  
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--init_lr', type=float, default=1.04e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-3) 
    
    ## Scheduler
    parser.add_argument('--scheduler', type=str, default='cos')
    parser.add_argument('--warm_epoch', type=int, default=5)  # WarmUp Scheduler
    parser.add_argument('--freeze_epoch', type=int, default=0)
    ### Cosine Annealing
    parser.add_argument('--min_lr', type=float, default=5e-6)
    parser.add_argument('--tmax', type=int, default = 145)
    ### MultiStepLR
    parser.add_argument('--milestone', type=int, nargs='*', default=[50])
    parser.add_argument('--lr_factor', type=float, default=0.1)
	### OnecycleLR
    parser.add_argument('--max_lr', type=float, default=1e-3)

    ## etc.
    parser.add_argument('--patience', type=int, default=5, help='Early Stopping')
    parser.add_argument('--clipping', type=float, default=None, help='Gradient clipping')
    parser.add_argument('--re_training_exp', type=str, default=None)
    parser.add_argument('--use_weight_norm', type=bool, default=None, help='Weight Normalization')

    # Hardware settings
    parser.add_argument('--amp', default=True)
    parser.add_argument('--multi_gpu', type=bool, default=False)
    parser.add_argument('--logging', type=bool, default=False)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    return args    
    
if __name__ == '__main__':
    args = getConfig()
    args = vars(args)
    print(args)