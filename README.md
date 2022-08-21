# pytorch-lightning-learn

## 简介

这部分是知乎上**PyTorch Lightning入门教程**的代码，代码分为三部分，分别对应知乎中的三个例子。

这里使用的python包的版本参考文件`requirements.txt`，这里采用anaconda导出的环境，安装命令如下：

```
conda install --yes --file requirements.txt
```

其中比较重要的几个环境的版本如下：
```
python3.7
pytorch=1.7.1
pytorch-lightning=1.5.10
transformers =4.21.1
```

## 运行方式

针对ResNet-50运行代码：

```
python main.py
```

针对BERT-base运行代码：

```
python main.py
```

针对T5-base运行代码：

```
python main.py
```


这里参考了如下的一些代码和内容：
https://www.pytorchlightning.ai/
https://github.com/renmada/t5-pegasus-pytorch