import time

out_dir = 'out-cndict_novel/32x32x2048'
eval_interval = 500
eval_iters = 500
wandb_log = True # feel free to turn 
wandb_project = 'cn-dict_novel_finetune'
wandb_run_name = 'ft-' + str(time.time())

dataset = 'lora'
init_from = 'resume' # this is the largest GPT-2 model

# only save checkpoints if the validation loss improves
always_save_checkpoint = False

# the number of examples per iter:
# 1 batch_size * 32 grad_accum * 1024 tokens = 32,768 tokens/iter
# shakespeare has 301,966 tokens, so 1 epoch ~= 9.2 iters
batch_size = 1
gradient_accumulation_steps = 32
max_iters = 25000
lr_decay_iters = 25000 # make equal to max_iters usually
block_size = 256 # context of up to 256 previous characters
# finetune at constant LR
learning_rate = 3e-5
decay_lr = False
