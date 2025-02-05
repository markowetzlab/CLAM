from __future__ import print_function

import argparse
import torch
import os
import pandas as pd
import wandb
from utils.utils import *
from dataset_modules.dataset_generic import Generic_MIL_Dataset
from utils.eval_utils import *
from sklearn.metrics import confusion_matrix

# Training settings
parser = argparse.ArgumentParser(description='CLAM Evaluation Script')
parser.add_argument('--data_root_dir', type=str, default=None,
                    help='data directory')
parser.add_argument('--results_dir', type=str, default='./results',
                    help='relative path to results folder, i.e. '+
                    'the directory containing models_exp_code relative to project root (default: ./results)')
parser.add_argument('--save_exp_code', type=str, default=None,
                    help='experiment code to save eval results')
parser.add_argument('--models_exp_code', type=str, default=None,
                    help='experiment code to load trained models (directory under results_dir containing model checkpoints')
parser.add_argument('--splits_dir', type=str, default=None,
                    help='splits directory, if using custom splits other than what matches the task (default: None)')
parser.add_argument('--model_size', type=str, choices=['small', 'big'], default='small', 
                    help='size of model (default: small)')
parser.add_argument('--model_type', type=str, choices=['abmil', 'clam_sb', 'clam_mb', 'mil', 'dgcn', 'mi_fcn', 'dsmil', 'trans_mil'], default='clam_sb', 
                    help='type of model (default: clam_sb)')
parser.add_argument('--k', type=int, default=10, help='number of folds (default: 10)')
parser.add_argument('--k_start', type=int, default=-1, help='start fold (default: -1, last fold)')
parser.add_argument('--k_end', type=int, default=-1, help='end fold (default: -1, first fold)')
parser.add_argument('--fold', type=int, default=-1, help='single fold to evaluate')
parser.add_argument('--micro_average', action='store_true', default=False, 
                    help='use micro_average instead of macro_avearge for multiclass AUC')
parser.add_argument('--split', type=str, choices=['train', 'val', 'test', 'all'], default='test')
parser.add_argument('--log_data', action='store_true', default=False, help='log data using wandb')
parser.add_argument('--project', type=str, default='wsi_mil', help='wandb project name')
parser.add_argument('--task', type=str, choices=['task_1_tumor_vs_normal',  'task_2_tumor_subtyping', 'he_tff3_delta', 'he_tff3_best4_pilot', 'he_tff3_best4_surveillance'])
parser.add_argument('--drop_out', type=float, default=0.25, help='dropout')
parser.add_argument('--embed_dim', type=int, default=1024)
args = parser.parse_args()

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

args.save_dir = os.path.join('./eval_results', 'EVAL_' + str(args.save_exp_code))
args.models_dir = os.path.join(args.results_dir, str(args.models_exp_code))

os.makedirs(args.save_dir, exist_ok=True)

if args.splits_dir is None:
    args.splits_dir = args.models_dir

assert os.path.isdir(args.models_dir)
assert os.path.isdir(args.splits_dir)

settings = {'task': args.task,
            'split': args.split,
            'save_dir': args.save_dir,
            'models_dir': args.models_dir,
            'model_type': args.model_type,
            'drop_out': args.drop_out,
            'model_size': args.model_size}

with open(args.save_dir + '/eval_experiment_{}.txt'.format(args.save_exp_code), 'w') as f:
    print(settings, file=f)
f.close()

print(settings)
if args.task == 'task_1_tumor_vs_normal':
    args.n_classes=2
    dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/tumor_vs_normal_dummy_clean.csv',
                            data_dir= os.path.join(args.data_root_dir, 'tumor_vs_normal_resnet_features'),
                            shuffle = False,
                            print_info = True,
                            label_dict = {'normal_tissue':0, 'tumor_tissue':1},
                            patient_strat=False,
                            ignore=[])
elif args.task == 'task_2_tumor_subtyping':
    args.n_classes=3
    dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/tumor_subtyping_dummy_clean.csv',
                            data_dir= os.path.join(args.data_root_dir, 'tumor_subtyping_resnet_features'),
                            shuffle = False,
                            print_info = True,
                            label_dict = {'subtype_1':0, 'subtype_2':1, 'subtype_3':2},
                            patient_strat= False,
                            ignore=[])
elif args.task == 'he_tff3_delta':
    args.n_classes=2
    dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/delta/he_tff3_adequate.csv',
                            data_dir= args.data_root_dir,
                            shuffle = False,
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'N':0, 'Y':1},
                            patient_strat=False,
                            ignore=[])
elif args.task == 'he_tff3_best4_pilot':
    args.n_classes=2
    dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/best4/pilot/he_tff3_slides.csv',
                            data_dir= args.data_root_dir,
                            shuffle = False,
                            print_info = True,
                            label_dict = {'N':0, 'E':0, 'Y':1},
                            patient_strat=False,
                            ignore=[])
elif args.task == 'he_tff3_best4_surveillance':
    args.n_classes=2
    dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/best4/surveillance/he_tff3_slides.csv',
                            data_dir= args.data_root_dir,
                            shuffle = False, 
                            print_info = True,
                            label_dict = {'Negative':0, 'Not Provided':0, 'Positive':1},
                            patient_strat=False,
                            ignore=[])
else:
    raise NotImplementedError

if args.k_start == -1:
    start = 0
else:
    start = args.k_start
if args.k_end == -1:
    end = args.k
else:
    end = args.k_end

if args.fold == -1:
    folds = range(start, end)
else:
    folds = range(args.fold, args.fold+1)
ckpt_paths = [os.path.join(args.models_dir, 's_{}_checkpoint.pt'.format(fold)) for fold in folds]
datasets_id = {'train': 0, 'val': 1, 'test': 2, 'all': -1}

if __name__ == "__main__":
    all_results = []
    all_auc = []
    all_acc = []
    for ckpt_idx in range(len(ckpt_paths)):
        if args.log_data:
            mode = 'online'
        else:
            mode = 'disabled'

        wandb.init(
            project=args.project,
            name=f"{args.save_exp_code}_{args.model_type}_{ckpt_idx}",
            config={"dataset": args.task, "model": args.model_type},
            group=f"{args.save_exp_code}_{args.model_type}",
            mode=mode,
            resume='allow'
            )
        
        if datasets_id[args.split] < 0:
            split_dataset = dataset
        else:
            csv_path = '{}/splits_{}.csv'.format(args.splits_dir, folds[ckpt_idx])
            datasets = dataset.return_splits(from_id=False, csv_path=csv_path)
            split_dataset = datasets[datasets_id[args.split]]
            writer = None

        model, patient_results, test_error, auc, df  = eval(split_dataset, args, ckpt_paths[ckpt_idx])
        all_results.append(all_results)
        all_auc.append(auc)
        all_acc.append(1-test_error)
        df.to_csv(os.path.join(args.save_dir, 'fold_{}.csv'.format(folds[ckpt_idx])), index=False)
        
        ground_truth_ids = df['Y'].tolist()
        pred_ids = df['Y_hat'].tolist()

        tn, fp, fn, tp = confusion_matrix(ground_truth_ids, pred_ids).ravel()
        sensitivity = tp/(tp+fn)
        specificity = tn/(tn+fp)

        # Log metrics to wandb
        wandb.log({
            f'{args.task}': folds[ckpt_idx],
            f'{args.task}/auc': auc,
            f'{args.task}/acc': 1-test_error,
            f'{args.task}/sensitivity': sensitivity,
            f'{args.task}/specificity': specificity
        })
        wandb.log({f"{args.task}/conf_matrix" : wandb.plot.confusion_matrix( 
            preds=pred_ids, y_true=ground_truth_ids)})

        wandb.finish()

    final_df = pd.DataFrame({'folds': folds, 'test_auc': all_auc, 'test_acc': all_acc})
    if len(folds) != args.k:
        save_name = 'summary_partial_{}_{}.csv'.format(folds[0], folds[-1])
    else:
        save_name = 'summary.csv'
    final_df.to_csv(os.path.join(args.save_dir, save_name), index=False)
