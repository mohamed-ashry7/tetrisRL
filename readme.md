NN+Dellacherie+Schwenker
BatchNorm1D
https://medium.com/syncedreview/batchnorm-dropout-dnn-success-eed740e1ca13
https://arxiv.org/pdf/1502.03167.pdf
BatchNorm+Dropout
https://stackoverflow.com/questions/39691902/ordering-of-batch-normalization-and-dropout
https://arxiv.org/pdf/1801.05134.pdf
https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf
Large Batch Size 
https://openai.com/blog/science-of-ai/



https://discuss.pytorch.org/t/batch-normalization-of-linear-layers/20989
https://towardsdatascience.com/designing-your-neural-networks-a5e4617027ed


exp1

When state space -> whole board 
reward -> not scaled and no time is used 



exp2 
reward = -5*avgh - qu - 16*holes + 10 cleared_lines - wells + time_stamp 
state space-> five rows 
reward scaled 


exp3
TD0 
reward -> Schwenker
reward scaled 

V1 BatchSize 640,LR->1e-5 ->

V2 BatchSize 120 lr->1e-4

V3 BatchSize 120 lr->1e-4 no time in the reward function no penalty at losing

V4 BatchSize lr 1e-4 timestamp penalty clippedStateEvaluation only between -15,15

V5 -10,10 HuberLoss 

V6 SYNC_TARGET_FRAMES = 10000 EPSILON_DECAY_LAST_FRAME = 50000 death penalty =-1 

V7 SYNC_TARGET_FRAMES = 10000 EPSILON_DECAY_LAST_FRAME = 200 000 death penalty =-1 Fixed some state space calculation 

23 lines cleared on average in 1000 games. 

