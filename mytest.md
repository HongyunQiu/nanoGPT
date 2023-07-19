## Data Preparation


python data/cndict_novel/prepare.py 


## Train


python train.py config/train_cndict_novel_xxxl.py --device='cuda:5' 
python train.py config/train_cndict_novel_xxxl.py --device='cuda:5' --init_from='resume'  


torchrun --standalone --nproc_per_node=4 train.py config/train_cndict_novel.py 



## inference

python sample.py --out_dir=out-cndict_novel_xl/80x80x1024/ --temperature=1.0 --device='cuda:4' --seed=800 --start="一个天文爱好者去峨眉山的山顶观测，发现了一个诡异的现象" 

python sample.py --out_dir=out-cndict_novel_xl/80x80x1024/ --temperature=1.0 --device='cuda:4' --seed=800 --start="白日依山尽，黄河入海流"

python sample.py --out_dir=out-cndict_novel_xl/80x80x1024/ --temperature=1.0 --device='cuda:4' --seed=800 --start="如何炼制硫酸" 