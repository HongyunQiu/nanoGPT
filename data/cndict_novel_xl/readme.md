
# The dataset for a bigger dataset based on about 6000 Chinese novels. 


Automaticly split the train and valid can not keep the style of the novels, For bigger dataset,In order to keep multiple style of the different sytle novels both in the train set and test set, we hope manually prepare the train and test dataset. 

During the data preparing I take a small percent of the novel as the test set, and the rest as the train set for each style. We will have the train.txt and val.txt



After running `prepare.py`:

- train.bin has   tokens
- val.bin has     tokens
