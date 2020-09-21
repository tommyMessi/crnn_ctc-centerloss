# crnn_ctc-centerloss

### 本项目用自己想法实现阿里云栖大会中，阿里团队提到的ctc+centerloss来解决相近字的问题 pdf百度网盘链接: https://pan.baidu.com/s/13370jLcBblmqvwfprHPYXw 提取码: mejj 

## 大概介绍
### 此项目是个人对pdf的一句话理解，外加尝试的结果。给大家一个解决相近字，或者ctc+centerloss的一个crnn实现方案。另外本项目适用于有debug代码能力的同学哦

## 环境(Requirements)
```pip install -r requirements.txt```

## 预训练模型(Model)
- 链接: https://pan.baidu.com/s/1H8YyRVN9keOQuQ-v3nwArg 提取码: vs3g

## 训练(train)
- 1 先用https://github.com/Sanster/tf_crnn 的crnn训练
- 2.对原始crnn训练到val acc 到95% loss 0.1左右，或者直至有满意的效果。
- 3.用gen_CR_data.py，用上面训练好的模型文件进行新的label生成
- 4.修改 crnn.py 文件 109行 centerloss 的权重为0。00001进行crnn 的训练 ```python train.py```
- 5.训练到val acc 95% 或者到自己对效果满意


## 测试(test)
```python test.py```

## 效果
### 测试图片为./data_example/test_data/xingjin

['成一一威，风一一凤，干一一千，土一一士，元一一无，他一一地', '素一一索，朱一一末，都一一郡，汨一一汩，李一一季，直一一真，', '阴一一阻，史一一更，思一一恩，孟一一盂', '侯一一候，竞一一竟，宵一一霄，毫一一毫', '令：怜，伶，邻， 冷，领，龄，铃，岭，玲，拎', '逢：缝，蓬，篷，峰，锋，逢，蜂', '1 79.00 30.02 30.02', 'A1200622287g4811330009']

- ![1](https://github.com/tommyMessi/crnn_ctc-centerloss/blob/master/data_example/test_data/xingjin/gg1.jpg)
- 'A1200622287g4811330009'
- ![2](https://github.com/tommyMessi/crnn_ctc-centerloss/blob/master/data_example/test_data/xingjin/gg2.jpg)
- '1 79.00 30.02 30.02'
- ![3](https://github.com/tommyMessi/crnn_ctc-centerloss/blob/master/data_example/test_data/xingjin/ling.png)
- '令：怜，伶，邻， 冷，领，龄，铃，岭，玲，拎'

## 其他
训练数据与预训练模型 关注微信公众账号 hulugeAI 留言：ctc 获取 线下wx交流群入门券


## 声明

copy right huluge
