#!/bin/bash

#! Give your job a name
#SBATCH -J extract_features
#! How many cores per task?
#SBATCH --cpus-per-task=12
#! How much memory do you need?
#SBATCH --mem=128G
#! How much wallclock time will be required?
#SBATCH --time=24:00:00
#! Request a GPU for this job
#SBATCH --gres=gpu:1
#! What types of email messages do you wish to receive?
#SBATCH --mail-type=END
#! Specify your email address here otherwise you wont receive emails!
#SBATCH --mail-user=rehan.zuberi@cruk.cam.ac.uk
#! Uncomment this to prevent the job from being requeued (e.g. if
#! interrupted by node failure or system downtime):
##SBATCH --no-requeue
#SBATCH -p cuda
#SBATCH -o /scratchc/fmlab/zuberi01/phd/saved_patches/features.out
#SBATCH -e /scratchc/fmlab/zuberi01/phd/saved_patches/features.error

CUDA_VISIBLE_DEVICES=0 python3 extract_features_fp.py   --data_h5_dir /scratchc/fmlab/zuberi01/phd/saved_patches   --data_slide_dir /mnt/scratchc/fmlab/datasets/imaging/SWGCohort/unannotated/   --csv_path /scratchc/fmlab/zuberi01/phd/saved_patches/process_list_full.csv   --feat_dir /scratchc/fmlab/zuberi01/phd/saved_patches/features_univ2   --model_name uni_v2  --batch_size 256   --target_patch_size 448   --silent   --no_auto_skip --slide_ext .ndpi
