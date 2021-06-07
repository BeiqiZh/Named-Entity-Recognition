# Named Entity Recognition (NER) using Keras LSTM & Spacy
How can we get useful information from massive unstructured documents? This question has been around for a long time before the named entity recognition (NER) model came out. This method can help people to extract key information from many different industries. This article will introduce and explain the methods used to solve the NER problem and shows the coding to build and train a bi-directional LSTM with Keras. On top of that, we will also demonstrate a NER model using Spacy.

## What is Name Recognition? 
NER seeks to extract and classify words into predefined categories such as person names, organizations, locations, medical codes, time expressions, quantities, monetary values, etc. NER can be used in natural language processing (NLP) to help answer real-world problems. This can be applied to recognize and parse important information from resumes, search for specific products mentioned in complaints or reviews, look for a company name in a news article, and many other uses. Apart from being used as an information extraction tool, it is also a preprocessing step for many NLP applications like machine translation, question answering, and text summarization. Now, let’s take a look a the different deep learning approaches to solve the NER problem.

## Keras Bidirectional-LSTM
Long Short Term Memory networks are a special kind of RNN capable of learning long-term dependencies. The network is designed to avoid the vanishing gradient problem. An LSTM unit is composed of a forget gate, an input gate, and an output gate. These gates can learn which data in a sequence is important to keep or throw away. As information travels through each chunk of neural network A, LSTM first decides what information we want to throw away using the forget gate. The next step is to decide what new information we are going to store in the cell state, using the input gate to update values. Finally, we decide what we are going to output based on the filtered information. By doing that, it can pass relevant information down the long chain of sequences to make predictions.

A bidirectional LSTM is a combination of two LSTMs — one runs forwards from right to left and one runs backward from left to right. This can prevent making predictions taking only the past information into account to improve model performance on sequence classification problems. We will be using bidirectional LSTM with Keras to solve the NER problem.

## Spacy
A simpler approach to solve the NER problem is to used Spacy, an open-source library for NLP. It provides features such as Tokenization, Parts-of-Speech (PoS) Tagging, Text Classification, and Named Entity Recognition. We will be using Spacy Pre-trained Model to show important

## Visit our Medium Article for Further Details: 
https://zhoubeiqi.medium.com/named-entity-recognition-ner-using-keras-lstm-spacy-da3ea63d24c5
