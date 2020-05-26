# news-topic-understanding
Understand news articles' topics and entity names using neural network

## How to run

### 1) Perceptron model

Run these commands in Terminal. They will apply Perceptron model to the Test data and output prediction to `result` folder.

- Newsgroup data:

    `python perceptron.py newsgroups`

- Propername data:

    `python perceptron.py propernames`

### 2) Maximum Entropy model
Run these commands in Terminal. They will apply Maximum Entropy model to the Test data and output prediction to `result` folder.

- Newsgroup data:

    `python3 maximum_entropy.py newsgroups`

- Propername data:

    `python3 maximum_entropy.py propernames`

### 3) Multi Layer Perception model

Run these commands in Terminal. They will apply Multi Layer Perceptron model to the Test data and output prediction to `result` folder. By default this will run for 1000 epochs. 

- Newsgroup data: 

    `python multilayer_perceptron.py newsgroups`

- Propername data:

    `python multilayer_perceptron.py propernames`
