# Implementation of "Globally and Locally Consistent Image Completion" with pytorch

```
  "Globally and Locally Consistent Image Completion"
  Satoshi Iizuka, Edgar Simo-Serra, and Hiroshi Ishikawa
  ACM Transaction on Graphics (Proc. of SIGGRAPH 2017), 2017
```

This is an implementation of the image completion model proposed in the paper
([Globally and Locally Consistent Image Completion](
http://hi.cs.waseda.ac.jp/%7Eiizuka/projects/completion/data/completion_sig2017.pdf))
with Pytorch 0.4.


# Requirements

- Python 3
- Pytorch 0.4
- TensorbardX
- argparser
- etc (PIL, tqdm...)
# Result
![](results/208_compare_gl.jpg)

![](results/474_compare_gl.jpg)

![](results/487_compare_gl.jpg)
# Usage

## I. Prepare the training data
 This step is pre-pocessing of the image (make random mask)
 and transform image to torch tensor.
```
$ cd src_gl
$ python TensorData_Loader.py
```

## II. Train model

Train the "GL" model with pre-processed tensor data in step I.
```
$ cd src_gl
$ python train.py
```
## III. Evaluate model
```
$ cd src_gl
$ python eval.py
```

