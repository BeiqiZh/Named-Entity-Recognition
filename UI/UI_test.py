import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import plotly.express as px
import tqdm
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import LSTM, Embedding, Dense
from tensorflow.keras.layers import InputLayer, TimeDistributed, SpatialDropout1D, Bidirectional
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from livelossplot.tf_keras import PlotLossesCallback
import time
import SessionState
import random

np.random.seed(0)
plt.style.use("ggplot")

ss = SessionState.get(df=pd.DataFrame(), xt=pd.DataFrame(), yt=pd.DataFrame(), model = 0, button1=False, button2=False)

#st.image('https://www.aacsb.edu/-/media/centennial/images/2017-innovations-that-inspire/mcgill%20u_2.ashx?h=106&w=350&la=en&hash=73814AEED3FB43CFD24426DF2A5B83211FA40E7F')
st.title('Named Entity Recognition (NER) using LSTMs with Keras')
st.write('\n')

st.sidebar.image('https://www.aacsb.edu/-/media/centennial/images/2017-innovations-that-inspire/mcgill%20u_2.ashx?h=106&w=350&la=en&hash=73814AEED3FB43CFD24426DF2A5B83211FA40E7F')

st.sidebar.markdown('**Please use this navigation panel to go through different steps:**')
selection = st.sidebar.radio('',('1. Import data', 
                     '2. Tags in the dataset', 
                     '3. Sentence length in the dataset',
                     '4. Training NER model using Keras',
                     '5. Model performance evaluation'))


if selection == '1. Import data':
    
    st.header('Step 1: Import the dataset used for training our NER model')
    
    st.write('This dashborad help you go through the steps of building an NER model using Keras.')
    st.write('Detailed explanation of the LSTM and NER model can be found in the original article. The full article on Medium that this dashboard refers to can be read by clicking the link below:')
    st.markdown("""<a href="https://zhoubeiqi.medium.com/named-entity-recognition-ner-using-keras-lstm-spacy-da3ea63d24c5">Link to the article</a>""", unsafe_allow_html=True,)
    
    st.write('The dataset is from Kaggle, a highly cited dataset used to train NER projects. It is extracted from the Groningen Meaning Bank (GMB), comprises thousands of sentences and words tagged, annotated, and built specifically to train the classifier to predict named entities. The dataset contains sentences in English and also an annotation for each word.')
    
    st.markdown("""<a href="https://www.kaggle.com/abhinavwalia95/entity-annotated-corpus">Link to the dataset on Kaggle</a>""", unsafe_allow_html=True,)
    
    st.write('The first step is to import the dataset. By clicking the "Import Data" button below, the dataset will be automatically loaded.')
    
    if st.button('Import Data'):

        #upload = st.file_uploader("Please select the dataset and upload it (.csv only)")
            
        data = pd.read_csv('ner_dataset.csv', encoding="latin1")
        
        
        #if upload is not None:
            
        if data is not None:
            
            
            #data = pd.read_csv(upload, encoding="latin1")
            data = data.fillna(method="ffill")
            
            st.write("Now let's take a preview of the dataset. The preview shows us the first 20 entries of the dataset.")
            st.write('Preview:')
            
            col1, col2, col3 = st.beta_columns([1,6,1])
            
            with col1:
                st.write("")
        
            with col2:
                st.write(data.head(20))
            
            with col3:
                st.write("")
            
            st.write('The columns of the dataset include the tags of the words and corresponding sentences that the words are in. ')
            
            ss.df = data
        
        st.write("Before moving to the next step, let's take a look at how many unique words and tags are in this dataset:")
        
        st.write("Unique words in corpus:", data['Word'].nunique())
        st.write("Unique tags in corpus:", data['Tag'].nunique())
        
        words = list(set(data["Word"].values))
        words.append("ENDPAD")
        num_words = len(words)
        
        tags = list(set(data["Tag"].values))
        num_tags = len(tags)
        
        st.write('Now we can move to the next step. Please use the navigation panel on the left-hand side to move through different steps.')

elif selection == '2. Tags in the dataset':

    st.header('Step 2: Explore different tags in the dataset')
    
    st.write('In this step, we will first examine the meanings of different tags and see how they are distributed among the dataset.')
    
    st.write('The meanings of the tags seen in this dataset in the table below:')
    
    col1, col2, col3 = st.beta_columns([7,6,6])
    
    with col1:
        st.write("")
    
    with col2:
        st.image('https://miro.medium.com/max/419/1*vwA3NYdjRVWBsLaheVRgiw.png')
    
    with col3:
        st.write("")
    
        
    data = ss.df
    
    st.write('By clicking the "View the Plot" button below, a plot showing the distribution of different tags within the dataset will be generated.')
    
    if st.button('View the Plot'):
        
        st.write('The plot below shows the number of words within each tag group:')
    
        fig = px.histogram(data[~data.Tag.str.contains("O")], x="Tag",color="Tag", title='Tag Distribution')
        fig.update_layout(title_x=0.5)
        st.plotly_chart(fig)
        
        st.write('We now have a brief understanding of the tags in the dataset, and we can move to the next step. Please use the navigation panel on the left-hand side to move through different steps.')

elif selection == '3. Sentence length in the dataset':
    
    st.header('Step 3: Explore the length of sentences in the dataset')

    st.write('Using Keras, we need to pad each sentence to the same length before feeding it to the model. Thus, it is important to see the sentence length distribution within the dataset in order to determine the padding size.')
    
    st.write('By clicking the "View the Plot" button below, a plot showing the distribution of different tags within the dataset will be generated.')
    
    data = ss.df
    
    def sentence_integrate(data):
      agg_func = lambda s: [(w, p, t) for w, p, t in zip(s["Word"].values.tolist(),
                                                               s["POS"].values.tolist(),
                                                               s["Tag"].values.tolist())]
      return data.groupby('Sentence #').apply(agg_func).tolist()
    
    sentences=sentence_integrate(data)
    
    if st.button('View the Plot'):
        
        st.write('The plot below shows the distribution of the length of sentences contained in the dataset:')
    
        fig = px.histogram(pd.DataFrame([len(s) for s in sentences],columns=['length']),x="length",marginal='box', title='Sentence Length Distribution')
        fig.update_layout(title_x=0.5)
        st.plotly_chart(fig)
        
        st.write('From the sentence length distribution, we can see that the mean value is around 20 words per sentence. Thus, we choose the padding to be 50, so most of the values do not need to be padded.')
        
        st.write("As we have determined the padding size, now let's move to the next step. Please use the navigation panel on the left-hand side to move through different steps.")

elif selection == '4. Training NER model using Keras':
    
    st.header('Step 4: Building and Training the NER model using LSTMs with Keras')
    
    st.write('First, we split the dataset into training and test subsets. Then, we build our model using Tensorflow Keras. We start with an input layer of shape 50, as defined previously. Then we add a layer for embeddings, and apply a spatial drop out that will drop the entire 1D feature map across all the channels. Finally, we create our bidirectional LSTM.')
        
    data = ss.df
    
    st.write('Click the "Build the Model" button below and the model building process will begin.')
    
    button1=st.empty()
    #ss.button1 = False
    #if st.button('Build the Model'):
    if button1.button('Build the Model') or ss.button1 == True:
        ss.button1 = True
        
    if ss.button1:
    
        def sentence_integrate(data):
          agg_func = lambda s: [(w, p, t) for w, p, t in zip(s["Word"].values.tolist(),
                                                                   s["POS"].values.tolist(),
                                                                   s["Tag"].values.tolist())]
          return data.groupby('Sentence #').apply(agg_func).tolist()
        
        sentences=sentence_integrate(data)
        
        
        st.write('The model summary plot is shown below. In total, we have 1,879,750 parameters.')
        
        words = list(set(data["Word"].values))
        words.append("ENDPAD")
        num_words = len(words)
        
        tags = list(set(data["Tag"].values))
        num_tags = len(tags)
        
        word2idx = {w: i + 1 for i, w in enumerate(words)}
        tag2idx = {t: i for i, t in enumerate(tags)}
        
        max_len = 50
        
        X = [[word2idx[w[0]] for w in s] for s in sentences]
        X = pad_sequences(maxlen=max_len, sequences=X, padding="post", value=num_words-1)
        
        y = [[tag2idx[w[2]] for w in s] for s in sentences]
        y = pad_sequences(maxlen=max_len, sequences=y, padding="post", value=tag2idx["O"])
        
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
        
        ss.xt = x_test
        ss.yt = y_test
        
        
        model = keras.Sequential()
        model.add(InputLayer((max_len)))
        model.add(Embedding(input_dim=num_words, output_dim=max_len, input_length=max_len))
        model.add(SpatialDropout1D(0.1))
        model.add( Bidirectional(LSTM(units=100, return_sequences=True, recurrent_dropout=0.1)))
        
        
        tf.keras.utils.plot_model(
            model, to_file='model.png', show_shapes=True, show_dtype=False,
            show_layer_names=True, rankdir='LR', expand_nested=True, dpi=300,
        )
        
        st.image('model.png')
        
        st.write('We then compile our model using an adam optimizer, sparse categorical cross-entropy loss, and an accuracy metric. Adam optimizer is usually used for stochastic gradient descent for training deep learning models. Categorical cross-entropy is usually used for multi-class classification problems.')
            
        model.compile(optimizer="adam",
                      loss="sparse_categorical_crossentropy",
                      metrics=["accuracy"])
        
        st.write('Now we have finished building our model, and the training progress can start. Click the "Start Training" button below to start the training process (it may take some time)')
        
        button2=st.empty()
        
        #if st.button('Start Training'):
        if button2.button('Start Training'):
            
            st.write('Model training has started...')
            
            logdir="log/"
            tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
            
            chkpt = ModelCheckpoint("model_weights.h5", monitor='val_loss',verbose=1, save_best_only=True, save_weights_only=True, mode='min')
            
            early_stopping = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=1, verbose=0, mode='max', baseline=None, restore_best_weights=False)
            
            callbacks = [PlotLossesCallback(), chkpt, early_stopping,tensorboard_callback]
            
            st.write('Model training in progress...')
            
            history = model.fit(
                x=x_train,
                y=y_train,
                validation_data=(x_test,y_test),
                batch_size=32, 
                epochs=3,
                callbacks=callbacks,
                verbose=1
                
            )
            
            st.write('Model training is finished!')
            
            ss.model = model
            
            accuracy = {'training accuracy': history.history['accuracy'],
                    'validation accuracy': history.history['val_accuracy']}
            
            loss = {'training loss': history.history['loss'],
                'validation loss': history.history['val_loss']}
            
            
            df_a = pd.DataFrame(columns=['epoch','accuracy','type'])
            
            df_a['index'] = range(len(accuracy['training accuracy'])+len(accuracy['validation accuracy']))
            
            for i in range(len(accuracy['training accuracy'])):
                df_a['accuracy'][i] = accuracy['training accuracy'][i]
                df_a['type'][i] = 'training'
                df_a['epoch'][i] = i
            
            for i in range(len(accuracy['training accuracy']),len(accuracy['training accuracy'])+len(accuracy['validation accuracy'])):
                df_a['accuracy'][i] = accuracy['validation accuracy'][i-len(accuracy['training accuracy'])]
                df_a['type'][i] = 'validation'
                df_a['epoch'][i] = i-len(accuracy['training accuracy'])
                
            df_l = pd.DataFrame(columns=['epoch','loss','type'])
            
            df_l['index'] = range(len(loss['training loss'])+len(loss['validation loss']))
            
            for i in range(len(loss['training loss'])):
                df_l['loss'][i] = loss['training loss'][i]
                df_l['type'][i] = 'training'
                df_l['epoch'][i] = i
            
            for i in range(len(loss['training loss']),len(loss['training loss'])+len(loss['validation loss'])):
                df_l['loss'][i] = loss['validation loss'][i-len(loss['training loss'])]
                df_l['type'][i] = 'validation'
                df_l['epoch'][i] = i-len(loss['training loss'])
                
            st.write('The training and validation accuracy for each epoch can be seen in the plot below:')
        
            fig = px.line(df_a, x="epoch", y="accuracy", color='type',title='Model Accuracy')
            fig.update_layout(title_x=0.5)
            st.plotly_chart(fig)
            
            st.write('The training and validation loss for each epoch can be seen in the plot below:')
        
            fig = px.line(df_l, x="epoch", y="loss", color='type',title='Model Loss')
            fig.update_layout(title_x=0.5)
            st.plotly_chart(fig)
            
            st.write('We have finished training our model now. We could moove to the final step to test our model performance. Please use the navigation panel on the left-hand side to move through different steps.')
    
elif selection == '5. Model performance evaluation':
    
    st.header('Step 5: Evaluating the NER model performance')
    
    st.write('In this final step, we will use the model we have trained to predict the tag of words in the test dataset and compare with the actual tags to evaluate the model performance. ')
    st.write('Click the "Evaluate Model" button below to see how well the model performs when predicting the word tags in the test dataset')
    
    data=ss.df
    x_test=ss.xt
    y_test=ss.yt
    model=ss.model
    
    max_len = 50
    
    words = list(set(data["Word"].values))
    words.append("ENDPAD")
    num_words = len(words)
    
    tags = list(set(data["Tag"].values))
    num_tags = len(tags)
    
    button1=st.empty()
    
    if button1.button('Evaluate Model') or ss.button2 == True:
        ss.button2 = True
    
    if ss.button2:
    
        st.markdown('__Performance Evaluation:__')
        st.write("Evaluation on test data is in progress...")
        results = model.evaluate(x_test, y_test, batch_size=128)
        st.write("Evaluation on test data is finished!")
        st.write('\n')
        st.markdown("__Test Accuracy__: {} ".format(results[1]))
        st.markdown("__Test Loss__: {} ".format(results[0]))
        st.write('\n')
        st.write('\n')
        
        st.write("Now let's examine the NER model performance in detail for a specific sentence in the dataset.")
        st.write('Click the button "Pick a Sentence" below will randomly choose a sentence from the dataset and display the ture and predicted tags of each word within it. This allows us to have a detailed view of how the model performs.')
        
        button2=st.empty()
        
        if button2.button('Pick a Sentence'):
            
            i = random.randint(0, x_test.shape[0])
            st.write("This sentence picked is:","Sentence No.",i)
            p = model.predict(np.array([x_test[i]]))
            p = np.argmax(p, axis=-1)
            
            test = pd.DataFrame(columns=['Index','Word','True','Pred'])
            
            test['Index']=range(max_len)
            
            index = 0
            
            for w, true, pred in zip(x_test[i], y_test[i], p[0]):
                
                test['Word'][index] = words[w-1]
                test['True'][index] = tags[true]
                test['Pred'][index] = tags[pred]
                
                index += 1
            
            test = test.drop(columns = ['Index'])
            
            st.write(test)
            
            st.write('You can click the "Pick a Sentence" button again to randomly choose a new sentence and see how the model performs on it.')
            
            st.header('Thanks for using!')

    


















    
