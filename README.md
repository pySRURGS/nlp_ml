# nlp_ml
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
python nlp_ml.py ./spam_data/SMSSpamCollection_train.csv ./spam_data/SMSSpamCollection_test.csv 10 ./spam_data.db
```

## Author
Sohrab Towfighi

## License
GPL Version 3.0
