from flask import *
import plotly.express as px
import plotly.io as pio
from werkzeug.utils import secure_filename
#import python classes and packages
from keras.utils.np_utils import to_categorical
from keras.models import Sequential,load_model
#loading CNN3D classes
from keras.layers import Conv3D, MaxPool3D, Flatten, Dense, InputLayer, BatchNormalization, Dropout, GlobalAveragePooling3D, MaxPooling2D
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
import pandas as pd
from keras import layers
import numpy as np
import keras
import tensorflow as tf
import os
from keras.layers import Convolution2D
import pickle
from keras.layers import Bidirectional, GRU, Conv1D, MaxPooling1D, RepeatVector#loading GRU, bidirectional, and CNN
import seaborn as sns
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from auth_utils import *


#defining class labels
labels = ['Walking', 'Upstairs', 'Downstairs', 'Sitting', 'Standing', 'Laying']
#define global variables to calculate and store accuracy and other metrics
precision = []
recall = []
fscore = []
accuracy = []


# Constants for file uploads
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'txt'}
MAX_UPLOAD_SIZE_MB = 512  # Maximum upload size in megabytes
app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_UPLOAD_SIZE_MB * 1024 * 1024  # Set max content length

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS





@app.route('/register')
def register():
    return render_template('register.html')

@app.route('/login')
def login():
    return render_template('login.html')


@app.route("/signup")
def signup_route():
    return signup()


@app.route("/signin")
def signin_route():
    return signin()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/home')
def home():
    return render_template('home.html')


@app.route('/load')
def load_dataset():
    global X,Y
    #loading UCI Har dataset captured activities from smart phones
    X = pd.read_csv("Dataset/X_train.txt", header=None, delim_whitespace=True)
    Y = pd.read_csv("Dataset/y_train.txt", header=None, delim_whitespace=True)
    content = "Dataset loaded successfully!!"
    
    # #visualizing class labels count found in dataset
    # names, count = np.unique(Y, return_counts = True)
    # height = count
    # bars = labels
    # y_pos = np.arange(len(bars))
    # plt.figure(figsize = (4, 3)) 
    # plt.bar(y_pos, height)
    # plt.xticks(y_pos, bars)
    # plt.xlabel("Dataset Class Label Graph")
    # plt.ylabel("Count")
    # plt.xticks(rotation=90)
    # plt.show()
    
    
    
    return render_template('load_dataset.html',content=content)

@app.route('/preprocess')
def preprocess_dataset():
    global X, Y,X_train,X_test,y_train,y_test
    #features processing, shuffling and splitting dataset into train and test
    X = X.values
    Y = Y.values
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    Y = to_categorical(Y)
    X = np.reshape(X, (X.shape[0], 17, 11, 3, 1))
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2,random_state=42) #split dataset into train and test
    print()
    print("Dataset train & test split as 80% dataset for training and 20% for testing")
    print("Training Size (80%): "+str(X_train.shape[0])) #print training and test size
    print("Testing Size (20%): "+str(X_test.shape[0]))
    content = (
        f"Dataset train & test split <br/> 80% dataset for training and 20% for testing<br/>"
        f"Total Size : {X.shape[0]}<br/>"
        f"Training Size (80%): {X_train.shape[0]}<br/>"
        f"Testing Size (20%): {X_test.shape[0]}"
    )
    
    return render_template('preprocess.html', content=content)

#function to calculate various metrics such as accuracy, precision etc
def calculateMetrics(algorithm, predict, testY):
    global accuracy, precision,recall,fscore
    p = precision_score(testY, predict,average='macro') * 100
    r = recall_score(testY, predict,average='macro') * 100
    f = f1_score(testY, predict,average='macro') * 100
    a = accuracy_score(testY,predict)*100     
    print(algorithm+' Accuracy  : '+str(a))
    print(algorithm+' Precision   : '+str(p))
    print(algorithm+' Recall      : '+str(r))
    print(algorithm+' FMeasure    : '+str(f))    
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    results = (f"{algorithm} Accuracy  : {round(a,2)}<br>"
               f"{algorithm} Precision   : {round(p,2)}<br>"
               f"{algorithm} Recall      : {round(r,2)}<br>"
               f"{algorithm} FMeasure    : {round(f,2)}")
    # conf_matrix = confusion_matrix(testY, predict) 
    # plt.figure(figsize =(4, 3)) 
    # ax = sns.heatmap(conf_matrix, xticklabels = labels, yticklabels = labels, annot = True, cmap="viridis" ,fmt ="g");
    # ax.set_ylim([0,len(labels)])
    # plt.title(algorithm+" Confusion matrix") 
    # plt.ylabel('True class') 
    # plt.xlabel('Predicted class') 
    # plt.show()
    return results

@app.route('/existingCNN')
def existing_cnn():
    global X, Y,X_train,X_test,y_train,y_test,X_test1
    #train existing CNN algorithm which will use many parameters for training and can increase computation complexity
    X_train1 = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], X_train.shape[2], (X_train.shape[3] * X_train.shape[4])))
    X_test1 = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], X_test.shape[2], (X_test.shape[3] * X_test.shape[4]))) 
    cnn_model = Sequential()
    #define cnn2d layer with 3 number of inout neurons and to filter dataset features
    cnn_model.add(Convolution2D(3, (1 , 1), input_shape = (X_train1.shape[1], X_train1.shape[2], X_train1.shape[3]), activation = 'relu'))
    #collect filtered features from CNN2D layer
    cnn_model.add(MaxPooling2D(pool_size = (1, 1)))
    #defining another layer t further optimize features
    cnn_model.add(Convolution2D(3, (1, 1), activation = 'relu'))
    cnn_model.add(MaxPooling2D(pool_size = (1, 1)))
    cnn_model.add(Flatten())
    #define output layer
    cnn_model.add(Dense(units = 16, activation = 'relu'))
    cnn_model.add(Dense(units = y_train.shape[1], activation = 'softmax'))
    #compile and train the model
    cnn_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    print(cnn_model.summary()) 
    if os.path.exists("model/cnn_weights.hdf5") == False:
        model_check_point = ModelCheckpoint(filepath='model/cnn_weights.hdf5', verbose = 1, save_best_only = True)
        hist = cnn_model.fit(X_train1, y_train, batch_size = 200, epochs = 20, validation_data=(X_test1, y_test), callbacks=[model_check_point], verbose=1)
        f = open('model/cnn_history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()    
    else:
        cnn_model.load_weights("model/cnn_weights.hdf5")
    #perform prediction on human behavior on test data using CNN existing model
    predict = cnn_model.predict(X_test1)
    predict = np.argmax(predict, axis=1)
    y_test1 = np.argmax(y_test, axis=1)
    content = calculateMetrics("Existing CNN Model", predict, y_test1)
    
    
    return render_template('existing_alg.html', content=content)


@app.route('/mdn')
def proposed_mdn_model():
    #train propose CNN3D model which will optimize model using space-time (ST) interaction module
    #of matrix operation and the depth separable convolution module. CNN3D is light in training which reduces the 
    #computational complexity of output weights and improves the compactness of the model structure. Propose algorithm also known
    #as MDN or MCNN
    #defining input shape
    inputs = keras.Input((X_train.shape[1], X_train.shape[2], X_train.shape[3], X_train.shape[4]))
    x = layers.Conv3D(filters=7, kernel_size=3, activation="relu")(inputs)#creating CNN3D layer with 7 neurons for data filter
    x = layers.MaxPool3D(pool_size=1)(x) #pool layer to collect filterd features from CNN3D layer
    x = layers.BatchNormalization()(x) #normalizing features
    x = layers.Conv3D(filters=7, kernel_size=1, activation="relu")(x)#another layer to optimze module using space time
    x = layers.MaxPool3D(pool_size=1)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv3D(filters=7, kernel_size=1, activation="relu")(x)
    x = layers.MaxPool3D(pool_size=1)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv3D(filters=32, kernel_size=1, activation="relu")(x)#cnn layer for separable convolution module
    x = layers.MaxPool3D(pool_size=1)(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling3D()(x)#defining global average pooling
    x = layers.Dense(units=64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(units=y_train.shape[1], activation="softmax")(x)
    mdn_model = keras.Model(inputs, outputs, name="3dcnn") #create model
    mdn_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    #displaying propose model complexity
    print(mdn_model.summary())
    if os.path.exists("model/mdn_weights.hdf5") == False:
        model_check_point = ModelCheckpoint(filepath='model/mdn_weights.hdf5', verbose = 1, save_best_only = True)
        hist = mdn_model.fit(X_train, y_train, batch_size = 32, epochs = 20, validation_data=(X_test, y_test), callbacks=[model_check_point], verbose=1)
        f = open('model/mdn_history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()    
    else:
        mdn_model.load_weights("model/mdn_weights.hdf5")
    
    #perform prediction  on test data using propose MDN model
    predict = mdn_model.predict(X_test)
    predict = np.argmax(predict, axis=1)
    y_test1 = np.argmax(y_test, axis=1)#calculate metrics
    content = calculateMetrics("Propose MDN Model", predict, y_test1)

    return render_template('proposed_alg.html', content=content)

@app.route('/extension')
def extension_hybrid_cnn_model():
    global extension_model
    #train extension hybrid model which is a combination ov CNN1 + Bidirectional + GRU and this model will optimize
    #features using 3 different models which can help in better prediction accuracy
    extension_model = Sequential()
    extension_model.add(Convolution2D(2, (1 , 1), input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3]), activation = 'relu'))
    extension_model.add(MaxPooling2D(pool_size = (1, 1)))
    extension_model.add(Convolution2D(1, (1, 1), activation = 'relu'))
    extension_model.add(MaxPooling2D(pool_size = (1, 1)))
    extension_model.add(Flatten())
    extension_model.add(RepeatVector(2))
    #adding bidirectional + GRU to CNN layer
    extension_model.add(Bidirectional(GRU(1, activation = 'relu')))
    extension_model.add(Dense(units = 1, activation = 'relu'))
    extension_model.add(Dense(units = y_train.shape[1], activation = 'softmax'))
    #compile and train the model
    extension_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    print(extension_model.summary()) 
    if os.path.exists("model/extension_weights.hdf5") == False:
        model_check_point = ModelCheckpoint(filepath='model/extension_weights.hdf5', verbose = 1, save_best_only = True)
        hist = extension_model.fit(X_train, y_train, batch_size = 32, epochs = 20, validation_data=(X_test, y_test), callbacks=[model_check_point], verbose=1)
        f = open('model/extension_history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()    
    else:
        extension_model = load_model("model/extension_weights.hdf5")

    #perform prediction on test data using extension model
    predict = extension_model.predict(X_test1)
    predict = np.argmax(predict, axis=1)
    y_test1 = np.argmax(y_test, axis=1)
    content = calculateMetrics("Extension Hybrid Model CNN + GRU + Bidirectional", predict, y_test1)
    return render_template('extension_alg.html', content=content)

@app.route('/accuracy_graph')
def accuracy_graph():

    
    df = pd.DataFrame([
        ['Existing CNN2D','Accuracy',accuracy[0]],['Existing CNN2D','Precision',precision[0]],['Existing CNN2D','Recall',recall[0]],['Existing CNN2D','FSCORE',fscore[0]],
        ['Propose MDN CNN3D','Accuracy',accuracy[1]],['Propose MDN CNN3D','Precision',precision[1]],['Propose MDN CNN3D','Recall',recall[1]],['Propose MDN CNN3D','FSCORE',fscore[1]],
        ['Extension CNN + Bidirectional + GRU','Accuracy',accuracy[2]],['Extension CNN + Bidirectional + GRU','Precision',precision[2]],['Extension CNN + Bidirectional + GRU','Recall',recall[2]],['Extension CNN + Bidirectional + GRU','FSCORE',fscore[2]]
    ], columns=['Algorithms', 'Parameters', 'Value'])
    
    fig = px.bar(df, x='Parameters', y='Value', color='Algorithms', barmode='group',
                 title="All Algorithms Performance Graph", height=600)
    
    # Convert the Plotly figure to HTML
    plot_html = pio.to_html(fig, full_html=False)
    
    return render_template('accuracy.html', plot_html=plot_html)


@app.route('/upload')
def upload():
    return render_template('predict_test_data.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    global df,extension_model
    if 'dataset' not in request.files:
        message = 'No file selected'
        return render_template('predict_test_data.html', message=message)

    dataset = request.files['dataset']

    if dataset.filename == '':
        message = 'No selected file'
        return render_template('upload.html', message=message)

    if dataset and allowed_file(dataset.filename):
        filename = secure_filename(dataset.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        dataset.save(filepath)
        try:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            #loading test behavior data
            testData = pd.read_csv(filepath, header=None, delim_whitespace=True)
            testData = testData.values
            indices = np.arange(testData.shape[0])
            np.random.shuffle(indices)#shuffling test data to select random 10 records
            testData = testData[indices]
            testData = testData[0:10,0:testData.shape[1]]#select 10 records
            testData1 = np.reshape(testData, (testData.shape[0], 17, 11, 3)) #convert test data as per CNN model
            # predict = extension_model.predict(testData1)#perform prediction on test data
            extension_model = load_model("model/extension_weights.hdf5")
            predictions = extension_model.predict(testData1)
            results = []

            for i in range(len(predictions)):
                pred = np.argmax(predictions[i])
                result = f"Test Data : {str(testData[i][0:30])} <br/> Predicted Activity ===> {labels[pred-1]}"
                results.append(result)

            return render_template('predict_test_data.html', results=results)
        except Exception as e:
            message = f"Error processing file: {e}"
        return render_template('predict_test_data.html', message=message)
    else:
        message = 'Allowed file types: .csv'
        return render_template('predict_test_data.html', message=message)





if __name__ == '__main__':
    app.run(debug=True)