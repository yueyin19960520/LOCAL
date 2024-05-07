1.与old_active_learning的不同在于，不是直接选pool中表现差的，而是选与train中表现差的相似的pool中的结构，这也与我们最终的目标吻合

####################################################################################################################################

1.数据集不shuffle（因为之前也都没shuffle，保持一致方便比较）；只使用与train set中表现不好的相似的pool中结构为了完全不泄露test的信息；允许同一个结构被添加多次（比如train中表现不好的两个结构a,b都与pool中的c相似，就
把c添加2次）以达到强调的作用；pool中结构被添加到train set后不删除因为如果它下次会再被选中就说明它蕴含的信息没学好，再添加一次以强调；每次选训练集中最差的0.03*len(pos2e_dataset)个结构，每个结构在pool里找3个，这里没有使用selection_ratio是为了想让每次加入的数量固定（0.09），最终加到大约是总数据集的90%；。4个iteration分别是0.184，0.172，0.167，0.161，最终test mae是0.161eV。
