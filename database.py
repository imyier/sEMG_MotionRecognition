import scipy.io as sio
import numpy as np



#===============================================================================
"""
    Sapsanis是UCI(University of California Irvine)提供的手部动作肌电图信号数据集。
此数据是在两篇论文中使用的，也是分两次采集的，因此分成了两个数据库，即包中的两个文件夹。

第一次采集：
    雇佣了3女2男实验，每个人采集的数据保存到一个文件中，每个文件中包含某个人在做6种手部
抓取动作时采集到的信号
    采集仪器是双通道的
    每次采集的序列长度为 3000。
    每个动作重复 30遍
        a) Spherical: 握球              spher_ch1,spher_ch2
        b) Tip:捏                       tip_ch1,
        c) Palmar: 握笔                 palm_ch1,
        d) Lateral: 拿薄的平的东西       lat_ch1,
        e) Cylindrical: 拿圆柱形的东西   cyl_ch1,
        f) Hook: 承受重压                hook_ch1,
"""
'''
参数说明：

返回值说明：
'''
def read_sapsanis():
    sapsanis_dataset ={}
    persons = ["female_1","female_2","female_3","male_1","male_2"]
    for person in persons:
        mat = sio.loadmat("./data_corpora/sapsanis/"+person+".mat")
        sapsanis_dataset[person]=mat
        
    #print(sapsanis_dataset)
    return sapsanis_dataset
#===============================================================================
"""
    CSL_HDEMG数据库是由德国不莱梅大学提供的手指动作肌电图信号数据集
"""
def read_csl_hdemg():
    pass


def main():
    data = read_sapsanis()
    print(data.keys)

if __name__ == '__main__':
    main()