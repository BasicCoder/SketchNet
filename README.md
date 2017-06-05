# SketchNet
## 1. Environment configuration
   > - Ubuntu 16.04 (16GB)
   > - python3.5
   > - Tensorflow 1.1

## 2. Code:
DataGenTFRecord.py : 从原始数据生成 TFRecord
	
三个网络分支:<br/>
SketchNet.py : sketch 分支<br/>
ImageNetNeg.py : Image Negative 分支<br/>
ImageNetPos.py : Image Positive 分支<br/>
	
ReadData.py : 从 TFRecord 读取数据
TrainModel.py : 训练模型

## 3. Data:
DownLoad Link:( [here](http://www.eecs.qmul.ac.uk/~qian/Qian's%20Materials/sbir_cvpr2016.tar) )

## 4. References:
- [Sketch Me That Shoe](http://www.eecs.qmul.ac.uk/~qian/Project_cvpr16.html)<br/>

| Sketch | Image |
| --- | --- |
| ![大銭湯と百の階段の街　線画](https://raw.githubusercontent.com/BasicCoder/SketchNet/master/9623213_p0.jpg) | ![大銭湯と百の階段の街](https://raw.githubusercontent.com/BasicCoder/SketchNet/master/9956255_p0.jpg) | 

[Street_Sketch]: https://raw.githubusercontent.com/BasicCoder/SketchNet/master/9623213_p0.jpg
[Street_Image]: https://raw.githubusercontent.com/BasicCoder/SketchNet/master/9956255_p0.jpg