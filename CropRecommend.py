import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn import tree
import warnings
warnings.filterwarnings('ignore')
df = pd.read_csv("crop_recommendation.csv")
features = df[['N', 'P','K','temperature', 'humidity', 'ph', 'rainfall']]
target = df['label']


# Initialzing empty lists to append all model's name and corresponding name
acc = []
model = []

# Splitting into train and test data

from sklearn.model_selection import train_test_split
Xtrain, Xtest, Ytrain, Ytest = train_test_split(features,target,test_size = 0.2,random_state =2)


##random forest algorith 
from sklearn.ensemble import RandomForestClassifier

RF = RandomForestClassifier(n_estimators=20, random_state=0)
RF.fit(Xtrain,Ytrain)

predicted_values = RF.predict(Xtest)

x = metrics.accuracy_score(Ytest, predicted_values)
acc.append(x)
model.append('RF')
print("RF's Accuracy is: ", x)

print(classification_report(Ytest,predicted_values))



import pickle
# Dump the trained Naive Bayes classifier with Pickle
RF_pkl_filename = 'RandomForest.pkl'
# Open the file to save as pkl file
RF_Model_pkl = open(RF_pkl_filename, 'wb')
pickle.dump(RF, RF_Model_pkl)
# Close the pickle instances
RF_Model_pkl.close()


###########################################################################################################################


from tkinter import *
from tkinter import ttk
from PIL import Image,ImageTk

root = Tk()
root.title('Crop Prediction System')
root.geometry('1500x750')
img=Image.open("aa.jpg")
img=img.resize((1500,750))
bg=ImageTk.PhotoImage(img)

lbl=Label(root,image=bg)
lbl.place(x=0,y=0)

var = StringVar()
label = Label( root, textvariable = var,font=('arial',20,'bold'),bd=20)
var.set('Crop Recommendation System')
label.place(x=200,y=20)

label_1 = ttk.Label(root, text ='nitrogen',font=("Helvetica", 16))
label_1.place(x=200,y=100)

    
Entry_1= Entry(root,justify=CENTER,font=("times",18,"bold"))
Entry_1.place(x=450,y=100)

label_2 = ttk.Label(root, text ='phosporus',font=("Helvetica", 16))
label_2.place(x=200,y=170)


Entry_2 = Entry(root,justify=CENTER,font=("times",18,"bold"))
Entry_2.place(x=450,y=170)

    
label_3 = ttk.Label(root, text ='pottasium',font=("Helvetica", 16))
label_3.place(x=200,y=240)

    
Entry_3 = Entry(root,justify=CENTER,font=("times",18,"bold"))
Entry_3.place(x=450,y=240)

label_4 = ttk.Label(root, text ='temperature',font=("Helvetica", 16))
label_4.place(x=200,y=310)

 
    
Entry_4= Entry(root,justify=CENTER,font=("times",18,"bold"))
Entry_4.place(x=450,y=310)


label_5 = ttk.Label(root, text ='humidity',font=("Helvetica", 16))
label_5.place(x=200,y=380)

 
    
Entry_5 = Entry(root,justify=CENTER,font=("times",18,"bold"))
Entry_5.place(x=450,y=380)


label_6 = ttk.Label(root, text ='ph',font=("Helvetica", 16))
label_6.place(x=200,y=450)
    
Entry_6 = Entry(root,justify=CENTER,font=("times",18,"bold"))
Entry_6.place(x=450,y=450)


label_7 = ttk.Label(root, text ='rainfall',font=("Helvetica", 16))
label_7.place(x=200,y=520)
    
Entry_7 = Entry(root,justify=CENTER,font=("times",18,"bold"))
Entry_7.place(x=450,y=520)



def predict():
    N = Entry_1.get()
    P = Entry_2.get()
    K = Entry_3.get()
    temperature =Entry_4.get()
    humidity =Entry_5.get()
    ph =Entry_6.get()  
    rainfall = Entry_7.get()   
    out = RF.predict([[float(N),
       float(P),
       float(K),
       float(temperature),
       float(humidity),
       float(ph),
       float(rainfall)]])     ##float(area)
    
    output.delete(0,END)
    output.insert(0,out[0])
   

b1 = Button(root, text = 'predict',font=("Helvetica", 16),command = predict)
b1.place(x=200,y=620)
    

output = Entry(root,justify=CENTER,font=("times",18,"bold"))
output.place(x=450,y=620)
    
root.mainloop()






#############################################################################################################






