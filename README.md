# nlp_ml
[![Build Status](https://travis-ci.org/pySRURGS/nlp_ml.svg?branch=master)](https://travis-ci.org/pySRURGS/nlp_ml)

Natural language processing and machine learning of a CSV file with text data and labels.
CSV file must have the columns 'text' and 'class'.

## Installation

```
git clone https://github.com/sohrabtowfighi/nlp_ml.git
cd nlp_ml
pip install -r requirements.txt --user
```

## Usage

```
python nlp_ml.py -h
```

Should print 

```
usage: nlp_ml.py [-h] train test iters path_to_db

positional arguments:
  train       absolute or relative file path to the csv file housing the
              training data
  test        absolute or relative file path to the csv file housing the
              testing data
  iters       the number of classifiers to be attempted in this run
  path_to_db  absolute or relative file path to the output sqlite database

optional arguments:
  -h, --help  show this help message and exit
```

### An example

```
python nlp_ml.py ./spam_data/SMSSpamCollection_train.csv ./spam_data/SMSSpamCollection_test.csv 4 ./spam_data.db
Using TensorFlow backend.

0.9727907536720443 0.9820770930518046 0.963159162051529
0.9949434143992295 0.9928074802205706 0.9971105225138455
0.9257163496267758 0.9900221729490022 0.8601011317120154
0.9920539369130749 0.9882915173237754 0.995906573561281

  _train_accuracy    _train_precision    _train_recall    _test_accuracy    _test_precision    _test_recall
-----------------  ------------------  ---------------  ----------------  -----------------  --------------
         0.994943            0.992807         0.997111          0.970323           0.877358        0.902913
         0.992054            0.988292         0.995907          0.974194           0.88785         0.92233
         0.972791            0.982077         0.963159          0.965161           0.872549        0.864078
         0.925716            0.990022         0.860101          0.944516           0.894737        0.660194

```

## Author
Sohrab Towfighi


## License
This project is licensed under the GPL 3.0 License - see the [LICENSE](LICENSE) file for details
