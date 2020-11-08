# Condolence Models

## Intro
`condolence-models` is a package used to detect condolence and distress
expressions, as well as empathetic comments. It is released with the
EMNLP 2020 paper `Condolence and Empathy in Online Commmunities`. 


## Install 

### Use pip
If `pip` is installed, question-intimacy could be installed directly from it:

    pip3 install condolence-models

### Dependencies
    python>=3.6.0
    torch>=1.6.0
    pytorch-transformers
    markdown
    beautifulsoup4
    numpy
    tqdm
    simpletransformers
    pandas
    numpy
    
## Usage and Example

See `example.py` for an example of how to use the classifiers.

> Note: The first time you run the code, the model parameters will need to be
> downloaded, which could take up significant space. The condolence and
> distress classifiers are about 500MB each, and the empathy classifier is
> about 1GB.

The interface for condolence and distress are the same. The interface for
empathy is slightly different, to align with the simpletransformers interface
more closely.

### Classifying condolence or distress.

```py
from condolence_models.condolence_classifier import CondolenceClassifier

cc = CondolenceClassifier()

# single string gets turned into a length-1 list
# outputs probabilities
print("I like ice cream")
print(cc.predict("I like ice cream"))
# [0.11919236]

# multiple strings
print(["I'm so sorry for your loss.", "F", "Tuesday is a good day of the week."])
print(cc.predict(["I'm so sorry for your loss.", "F", "Tuesday is a good day of the week."]))
# [0.9999901  0.8716224  0.20647633]
```

### Classifying empathy.

```py
from condolence_models.empathy_classifier import EmpathyClassifier
ec = EmpathyClassifier(use_cuda=True, cuda_device=2)

# list of lists
# first item is target, second is observer
# regression output on scale of 1 to 5
print([["", "Yes, but wouldn't that block the screen?"]])
print(ec.predict([["", "Yes, but wouldn't that block the screen?"]]))
# [1.098]
```

## Contact
Naitian Zhou (naitian@umich.edu)
