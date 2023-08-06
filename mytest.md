## Data Preparation


python data/cndict_novel/prepare.py 


## Train


python train.py config/train_cndict_novel_xxxl.py --device='cuda:5' 
python train.py config/train_cndict_novel_xxxl.py --device='cuda:5' --init_from='resume'  

### multi-gpu training
torchrun --standalone --nproc_per_node=4 train.py config/train_cndict_novel.py 

### define GPU 2 AND 3 to run 
CUDA_VISIBLE_DEVICES=2,3 torchrun --standalone --nproc_per_node=2 train.py config/train_cndict_novel_highlayer.py

## inference

python sample.py --out_dir=out-cndict_novel_xl/80x80x1024/ --temperature=1.0 --device='cuda:4' --seed=800 --start="一个天文爱好者去峨眉山的山顶观测，发现了一个诡异的现象" 

python sample.py --out_dir=out-cndict_novel_xl/80x80x1024/ --temperature=1.0 --device='cuda:4' --seed=800 --start="白日依山尽，黄河入海流"

python sample.py --out_dir=out-cndict_novel_xl/80x80x1024/ --seed=100 --start="如何在一个彗星上生存,当第一次踏上这颗彗星的时候,这个问题就一直在脑海中浮现,彗星围绕着太阳旋转,会飞到太阳系的边缘" --temperature=1.0 --device='cuda:4' --num_samples=1 --max_new_tokens=500


## finetune
### Prepare the train/val dataset
It is important to use the dict of the model to generate the train/val dataset. 
(1)Make a sub-folder in /data   
(2)copy the train txt file into this sub-folder
(3)modify the prepare.py based on the prepare.py in cn-dict_novel
      （3.1） Read the data.pkl file in the model
      


2.Prepare the config file
