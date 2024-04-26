注意用13000个结构的数据集，pooling用tsfm，没有用主动学习，没有用data augmentation，参数如文件所示。

只测试了FcN_BC_Homo，测试结果的模型和输出的txt已上传，最好的test mae为epoch 283的0.152

由于sklearn版本问题，我把dataset.py的OneHotEncoder(sparse 改成了OneHotEncoder(sparse_output

##########################################################################################

把epoch增加至400，test mae最好是0.149

##########################################################################################

把epoch增加至500，test mae最好是0.147
