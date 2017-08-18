# Keras classification

Image classification with keras

## Requirements

- python 2.7
- OpenCV 3.2.0
- [Keras 2.0.5](https://github.com/fchollet/keras)
- [TensorFlow 1.2.0](https://github.com/tensorflow/tensorflow)

## Usage  

Input data(only for evaluation)

    └── data
        └── validation
            └── cat
                └── xxx.png (name doesn't matter)
            └── dog    
                └── xxx.png (name doesn't matter)

The dataset directory structure is followed to use the Keras DataGen Framework.

checkpoint files,
    
    └── checkpoint
        └── (ckpt_name)
            ├── model.json 
            ├── weight.xx.h5
            └── ...

You can download ~~CHECKPOINT~~ files in project

To test a model

    $ python main.py --mode eval --ckpt_name weight --data_path ./data --batch_size 20

### Result

![result](./result/result.png)

