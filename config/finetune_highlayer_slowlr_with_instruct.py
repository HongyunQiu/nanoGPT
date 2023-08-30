import time

out_dir = 'out-cndict_novel_xl/hihglayer_slowlr'
eval_interval = 250
eval_iters = 200
wandb_log = False # feel free to turn on
wandb_project = 'highlayer-slowlr-finetune'
wandb_run_name = 'ft-' + str(time.time())

dataset = 'cndict_novel_instruct'  # this is the folder name with the finetune dataset
init_from = 'out-cndict_novel_xl/hihglayer_slowlr' # this is the model name

# only save checkpoints if the validation loss improves
always_save_checkpoint = True

# the number of examples per iter:
# 1 batch_size * 32 grad_accum * 1024 tokens = 32,768 tokens/iter
# shakespeare has 301,966 tokens, so 1 epoch ~= 9.2 iters
batch_size = 1
gradient_accumulation_steps = 2
max_iters = 305000

# finetune at constant LR
learning_rate = 1e-5
decay_lr = False
