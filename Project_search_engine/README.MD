# Simple Image Search Engine

This Python program is compatible on Python 3.7.4

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install cv2 and numpy.

```bash
pip install cv2
pip install numpy
```

## Usage

```python
# from the code repository, issue the following commands:
# creates the pickle file
python index.py --dataset images --index index.pickle
# shows the results
python search.py --dataset images --index index.pickle --query query/qi1.jpg
```