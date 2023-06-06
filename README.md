# AllaertCNN
A Pytorch implementation for Allaert CNN(2D only) in https://arxiv.org/pdf/1904.11592.pdf, for facial expression classification.
<img width="709" alt="image" src="https://github.com/shibbit/AllaertCNN/assets/98523803/c1caebb1-9683-477b-96d4-2e100b2497f9">

# Note
The AllaertCNN is designed for facial expression optical flows classification task. And this project was composed for CASME dataset, you can apply them here:http://casme.psych.ac.cn/casme/. Or you can train the network on your own dataset.

# Demo
The pretrained pt file on CASME II is provided. With the accuracy of predictions on original testset(5166 images): 79.07%。
You can run *my_test_which_I_don't_know_if_it's_solid.py*, and load the *Test_model_5-14_backup.t7* from Modelsave directory.
