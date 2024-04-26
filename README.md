1.线性层和卷积层的每层维度可以在main_stream.py中指定

2.修改net.py中的CONT2E_Net类，分别用Transformer, mean, max, add池化之后cat再过线性层

3.由于sklearn版本问题，我把dataset.py的OneHotEncoder(sparse 改成了OneHotEncoder(sparse_output

4.main_stream.py中注释了if __name__ == "__main__":，并加入了一些#%%，这是为了在vscode中分段执行

############################################################################################

linear_dim_list=[],conv_dim_list=[[256,256],[256,256],[256,256],[256,256],]，300个epoch，最好test mae达到了0.145，在epoch 278
