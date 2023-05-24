
import matplotlib.pyplot as plt

#Importing Libraries to perform calculations
import numpy as np
import pandas as pd
import streamlit as st
#List of the symptoms is listed here in list l1.

l1=['itching',
    'skin_rash',
    'nodal_skin_eruptions',
    'continuous_sneezing',
    'shivering',
    'chills',
    'joint_pain',
    'stomach_pain',
    'acidity',
    'ulcers_on_tongue',
    'muscle_wasting',
    'vomiting',
    'burning_micturition',
    'fatigue',
    'weight_gain',
    'anxiety',
    'cold_hands_and_feets',
    'mood_swings',
    'weight_loss',
    'restlessness',
    'lethargy',
    'patches_in_throat',
    'irregular_sugar_level',
    'cough',
    'high_fever',
    'sunken_eyes',
    'breathlessness',
    'sweating',
    'dehydration',
    'indigestion',
    'headache',
    'yellowish_skin',
    'dark_urine',
    'nausea',
    'loss_of_appetite',
    'pain_behind_the_eyes',
    'back_pain',
    'constipation',
    'abdominal_pain',
    'diarrhoea',
    'mild_fever',
    'yellow_urine',
    'yellowing_of_eyes',
    'acute_liver_failure',
    'fluid_overload',
    'swelling_of_stomach',
    'swelled_lymph_nodes',
    'malaise',
    'blurred_and_distorted_vision',
    'phlegm',
    'throat_irritation',
    'redness_of_eyes',
    'sinus_pressure',
    'runny_nose',
    'congestion',
    'chest_pain',
    'weakness_in_limbs',
    'fast_heart_rate',
    'pain_during_bowel_movements',
    'pain_in_anal_region',
    'bloody_stool',
    'irritation_in_anus',
    'neck_pain',
    'dizziness',
    'cramps',
    'bruising',
    'obesity',
    'swollen_legs',
    'swollen_blood_vessels',
    'puffy_face_and_eyes',
    'enlarged_thyroid',
    'brittle_nails',
    'swollen_extremeties',
    'excessive_hunger',
    'extra_marital_contacts',
    'drying_and_tingling_lips',
    'slurred_speech',
    'knee_pain',
    'hip_joint_pain',
    'muscle_weakness',
    'stiff_neck',
    'swelling_joints',
    'movement_stiffness',
    'spinning_movements',
    'loss_of_balance',
    'unsteadiness',
    'weakness_of_one_body_side',
    'loss_of_smell',
    'bladder_discomfort',
    'continuous_feel_of_urine',
    'passage_of_gases',
    'internal_itching',
    'toxic_look_(typhos)',
    'depression',
    'irritability',
    'muscle_pain',
    'altered_sensorium',
    'red_spots_over_body',
    'belly_pain',
    'abnormal_menstruation',
    'watering_from_eyes',
    'increased_appetite',
    'polyuria',
    'family_history',
    'mucoid_sputum',
    'rusty_sputum',
    'lack_of_concentration',
    'visual_disturbances',
    'receiving_blood_transfusion',
    'receiving_unsterile_injections',
    'coma',
    'stomach_bleeding',
    'distention_of_abdomen',
    'history_of_alcohol_consumption',
    'fluid_overload',
    'blood_in_sputum',
    'prominent_veins_on_calf',
    'palpitations',
    'painful_walking',
    'pus_filled_pimples',
    'blackheads',
    'scurring',
    'skin_peeling',
    'silver_like_dusting',
    'small_dents_in_nails',
    'inflammatory_nails',
    'blister',
    'red_sore_around_nose',
    'yellow_crust_ooze']


# Add content to the sidebar

# Sidebar
st.sidebar.title("SmartHealth")

st.sidebar.markdown(
    """
    <div>
        <h3 style="color: white;">Description</h3>
        <p style="color: grey;">Are you interested in managing and improving your health? Our AI-driven health predictions system offers you insights about core metrics . By easily integrating with your wearable devices and analyzing past data intelligently, the system can accurately predict key metrics so that you can make proactive decisions on how to better take care of yourself. With our technology, it is now easier than ever to monitor and manage your health - giving you peace of mind for a better life!</p>
    </div>
    """,
    unsafe_allow_html=True
)
st.title("Disease Detection")
st.write("This app uses 5 inputs to predict your disease based on symptoms."
         "Please select the symptoms from the dropdown menus below:")
st.markdown('<p style="color:red;">Please wait , It can take some minutes !</p>', unsafe_allow_html=True)
st.sidebar.button("Diseases Dtetction")
st.sidebar.markdown(
    """
    <div>
        <p style="color: grey;">Coming soon... !    </p>
    </div>
    """,
    unsafe_allow_html=True
)


symp1=st.selectbox('Sypmtome 1:',l1)
symp2=st.selectbox('Sypmtome 2:',l1)
symp3=st.selectbox('Sypmtome 3:',l1)
symp4=st.selectbox('Sypmtome 4:',l1)
symp5=st.selectbox('Sypmtome 5:',l1)


#List of Diseases is listed in list disease.

disease=['Fungal infection','Allergy','GERD','Chronic cholestasis','Drug Reaction',
         'Peptic ulcer diseae','AIDS','Diabetes','Gastroenteritis','Bronchial Asthma','Hypertension',
         ' Migraine','Cervical spondylosis',
         'Paralysis (brain hemorrhage)','Jaundice','Malaria','Chicken pox','Dengue','Typhoid','hepatitis A',
         'Hepatitis B','Hepatitis C','Hepatitis D','Hepatitis E','Alcoholic hepatitis','Tuberculosis',
         'Common Cold','Pneumonia','Dimorphic hemmorhoids(piles)',
         'Heartattack','Varicoseveins','Hypothyroidism','Hyperthyroidism','Hypoglycemia','Osteoarthristis',
         'Arthritis','(vertigo) Paroymsal  Positional Vertigo','Acne','Urinary tract infection','Psoriasis',
         'Impetigo']

l2=[]
for i in range(0,len(l1)):
    l2.append(0)
print(l2)

#Reading the training .csv file
df=pd.read_csv("Dataset/training.csv")
DF= pd.read_csv('Dataset/training.csv', index_col='prognosis')

#Replace the values in the imported file by pandas by the inbuilt function replace in pandas.

df.replace({'prognosis':{'Fungal infection':0,'Allergy':1,'GERD':2,'Chronic cholestasis':3,'Drug Reaction':4,
                         'Peptic ulcer diseae':5,'AIDS':6,'Diabetes ':7,'Gastroenteritis':8,'Bronchial Asthma':9,'Hypertension ':10,
                         'Migraine':11,'Cervical spondylosis':12,
                         'Paralysis (brain hemorrhage)':13,'Jaundice':14,'Malaria':15,'Chicken pox':16,'Dengue':17,'Typhoid':18,'hepatitis A':19,
                         'Hepatitis B':20,'Hepatitis C':21,'Hepatitis D':22,'Hepatitis E':23,'Alcoholic hepatitis':24,'Tuberculosis':25,
                         'Common Cold':26,'Pneumonia':27,'Dimorphic hemmorhoids(piles)':28,'Heart attack':29,'Varicose veins':30,'Hypothyroidism':31,
                         'Hyperthyroidism':32,'Hypoglycemia':33,'Osteoarthristis':34,'Arthritis':35,
                         '(vertigo) Paroymsal  Positional Vertigo':36,'Acne':37,'Urinary tract infection':38,'Psoriasis':39,
                         'Impetigo':40}},inplace=True)

#printing the top 5 rows of the training dataset
df.head()

# Distribution graphs (histogram/bar graph) of column data
def plotPerColumnDistribution(df1, nGraphShown, nGraphPerRow):
    nunique = df1.nunique()
    df1 = df1[[col for col in df if nunique[col] > 1 and nunique[col] < 50]] # For displaying purposes, pick columns that have between 1 and 50 unique values
    nRow, nCol = df1.shape
    columnNames = list(df1)
    nGraphRow = (nCol + nGraphPerRow - 1) / nGraphPerRow
    plt.figure(num = None, figsize = (6 * nGraphPerRow, 8 * nGraphRow), dpi = 80, facecolor = 'w', edgecolor = 'k')
    for i in range(min(nCol, nGraphShown)):
        columnDf = df.iloc[:, i]
        if (not np.issubdtype(type(columnDf.iloc[0]), np.number)):
            valueCounts = columnDf.value_counts()
            valueCounts.plot.bar()
        else:
            columnDf.hist()
        plt.ylabel('counts')
        plt.xticks(rotation = 90)
        plt.title(f'{columnNames[i]} (column {i})')
    plt.tight_layout(pad = 1.0, w_pad = 1.0, h_pad = 1.0)
    plt.show()

# Scatter and density plots
def plotScatterMatrix(df1, plotSize, textSize):
    df1 = df1.select_dtypes(include =[np.number]) # keep only numerical columns
    # Remove rows and columns that would lead to df being singular
    df1 = df1.dropna('columns')
    df1 = df1[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values
    columnNames = list(df)
    if len(columnNames) > 10: # reduce the number of columns for matrix inversion of kernel density plots
        columnNames = columnNames[:10]
    df1 = df1[columnNames]
    ax = pd.plotting.scatter_matrix(df1, alpha=0.75, figsize=[plotSize, plotSize], diagonal='kde')
    corrs = df1.corr().values
    for i, j in zip(*plt.np.triu_indices_from(ax, k = 1)):
        ax[i, j].annotate('Corr. coef = %.3f' % corrs[i, j], (0.8, 0.2), xycoords='axes fraction', ha='center', va='center', size=textSize)
    plt.suptitle('Scatter and Density Plot')
    plt.show()

plotScatterMatrix(df, 20, 10)

X= df[l1]
y = df[["prognosis"]]
np.ravel(y)
print(X)
print(y)

#Reading the  testing.csv file
tr=pd.read_csv("Dataset/testing.csv")

#Using inbuilt function replace in pandas for replacing the values

tr.replace({'prognosis':{'Fungal infection':0,'Allergy':1,'GERD':2,'Chronic cholestasis':3,'Drug Reaction':4,
                         'Peptic ulcer diseae':5,'AIDS':6,'Diabetes ':7,'Gastroenteritis':8,'Bronchial Asthma':9,'Hypertension ':10,
                         'Migraine':11,'Cervical spondylosis':12,
                         'Paralysis (brain hemorrhage)':13,'Jaundice':14,'Malaria':15,'Chicken pox':16,'Dengue':17,'Typhoid':18,'hepatitis A':19,
                         'Hepatitis B':20,'Hepatitis C':21,'Hepatitis D':22,'Hepatitis E':23,'Alcoholic hepatitis':24,'Tuberculosis':25,
                         'Common Cold':26,'Pneumonia':27,'Dimorphic hemmorhoids(piles)':28,'Heart attack':29,'Varicose veins':30,'Hypothyroidism':31,
                         'Hyperthyroidism':32,'Hypoglycemia':33,'Osteoarthristis':34,'Arthritis':35,
                         '(vertigo) Paroymsal  Positional Vertigo':36,'Acne':37,'Urinary tract infection':38,'Psoriasis':39,
                         'Impetigo':40}},inplace=True)

#printing the top 5 rows of the testing data
tr.head()
plotPerColumnDistribution(tr, 10, 5)
plotScatterMatrix(tr, 20, 10)

X_test= tr[l1]
y_test = tr[["prognosis"]]
np.ravel(y_test)
print(X_test)
print(y_test)

#list1 = DF['prognosis'].unique()
def scatterplt(disea):
    x = ((DF.loc[disea]).sum())#total sum of symptom reported for given disease
    x.drop(x[x==0].index,inplace=True)#droping symptoms with values 0
    print(x.values)
    y = x.keys()#storing nameof symptoms in y
    print(len(x))
    print(len(y))
    plt.title(disea)
    plt.scatter(y,x.values)
    plt.show()

def scatterinp(sym1,sym2,sym3,sym4,sym5):
    x = [sym1,sym2,sym3,sym4,sym5]#storing input symptoms in y
    y = [0,0,0,0,0]#creating and giving values to the input symptoms
    if(sym1!='Select Here'):
        y[0]=1
    if(sym2!='Select Here'):
        y[1]=1
    if(sym3!='Select Here'):
        y[2]=1
    if(sym4!='Select Here'):
        y[3]=1
    if(sym5!='Select Here'):
        y[4]=1
    print(x)
    print(y)
    plt.scatter(x,y)
    plt.show()
def DecisionTree():
    from sklearn import tree

    clf3 = tree.DecisionTreeClassifier()
    clf3 = clf3.fit(X,y)

    from sklearn.metrics import confusion_matrix,accuracy_score
    y_pred=clf3.predict(X_test)
    print("Decision Tree")
    print("Accuracy")
    print(accuracy_score(y_test, y_pred))
    print(accuracy_score(y_test, y_pred,normalize=False))
    print("Confusion matrix")
    conf_matrix=confusion_matrix(y_test,y_pred)
    print(conf_matrix)

    psymptoms = [symp1,symp2,symp3,symp4,symp5]

    for k in range(0,len(l1)):
        for z in psymptoms:
            if(z==l1[k]):
                l2[k]=1

    inputtest = [l2]
    predict = clf3.predict(inputtest)
    predicted=predict[0]
    st.write(predicted)
    if predicted == 0:
        st.write("Your Disease is :  Fungal infection")
    elif predicted == 1:
        st.write("Your Disease is :  Allergy")
    elif predicted == 2:
        st.write("Your Disease is :  GERD")
    elif predicted == 3:
        st.write("Your Disease is :  Chronic cholestasis")
    elif predicted == 4:
        st.write("Your Disease is :  Drug Reaction")
    elif predicted == 5:
        st.write("Your Disease is :  Peptic ulcer disease")
    elif predicted == 6:
        st.write("Your Disease is :  AIDS")
    elif predicted == 7:
        st.write("Your Disease is :  Diabetes")
    elif predicted == 8:
        st.write("Your Disease is :  Gastroenteritis")
    elif predicted == 9:
        st.write("Your Disease is :  Bronchial Asthma")
    elif predicted == 10:
        st.write("Your Disease is :  Hypertension")
    elif predicted == 11:
        st.write("Your Disease is :  Migraine")
    elif predicted == 12:
        st.write("Your Disease is :  Cervical spondylosis")
    elif predicted == 13:
        st.write("Your Disease is :  Paralysis (brain hemorrhage)")
    elif predicted == 14:
        st.write("Your Disease is :  Jaundice")
    elif predicted == 15:
        st.write("Your Disease is :  Malaria")
    elif predicted == 16:
        st.write("Your Disease is :  Chicken pox")
    elif predicted == 17:
        st.write("Your Disease is :  Dengue")
    elif predicted == 18:
        st.write("Your Disease is :  Typhoid")
    elif predicted == 19:
        st.write("Your Disease is :  Hepatitis A")
    elif predicted == 20:
        st.write("Your Disease is :  Hepatitis B")
    elif predicted == 21:
        st.write("Your Disease is :  Hepatitis C")
    elif predicted == 22:
        st.write("Your Disease is :  Hepatitis D")
    elif predicted == 23:
        st.write("Your Disease is :  Hepatitis E")
    elif predicted == 24:
        st.write("Your Disease is :  Alcoholic hepatitis")
    elif predicted == 25:
        st.write("Your Disease is :  Tuberculosis")
    elif predicted == 26:
        st.write("Your Disease is :  Common Cold")
    elif predicted == 27:
        st.write("Your Disease is :  Pneumonia")
    elif predicted == 28:
        st.write("Your Disease is :  Dimorphic hemmorhoids (piles)")
    elif predicted == 29:
        st.write("Your Disease is :  Heart attack")
    elif predicted == 30:
        st.write("Your Disease is :  Varicose veins")
    elif predicted == 31:
        st.write("Your Disease is :  Hypothyroidism")
    elif predicted == 32:
        st.write("Your Disease is :  Hyperthyroidism")
    elif predicted == 33:
        st.write("Your Disease is :  Hypoglycemia")
    elif predicted == 34:
        st.write("Your Disease is :  Osteoarthristis")
    elif predicted == 35:
        st.write("Your Disease is :  Arthritis")
    elif predicted == 36:
        st.write("Your Disease is :  (vertigo) Paroymsal Positional Vertigo")
    elif predicted == 37:
        st.write("Your Disease is :  Acne")
    elif predicted == 38:
        st.write("Your Disease is :  Urinary tract infection")
    elif predicted == 39:
        st.write("Your Disease is :  Psoriasis")
    elif predicted == 40:
        st.write("Your Disease is :  Impetigo")
    else:
        st.write("Unknown Disease")
    st.header("Accuracy Score")
    accuracy = accuracy_score(y, clf3.predict(X))
    st.text("Accuracy: {:.2f}%".format(accuracy * 100))



def randomforest():
    from sklearn.ensemble import RandomForestClassifier
    clf4 = RandomForestClassifier(n_estimators=100)
    clf4 = clf4.fit(X,np.ravel(y))

    # calculating accuracy
    from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
    y_pred=clf4.predict(X_test)
    print("Random Forest")
    print("Accuracy")
    print(accuracy_score(y_test, y_pred))
    print(accuracy_score(y_test, y_pred,normalize=False))
    print("Confusion matrix")
    conf_matrix=confusion_matrix(y_test,y_pred)
    print(conf_matrix)

    psymptoms = [symp1,symp2,symp3,symp4,symp5]

    for k in range(0,len(l1)):
        for z in psymptoms:
            if(z==l1[k]):
                l2[k]=1

    inputtest = [l2]
    predict = clf4.predict(inputtest)
    predicted=predict[0]
    if predicted == 0:
        st.write("Your Disease is :  Fungal infection")
    elif predicted == 1:
        st.write("Your Disease is :  Allergy")
    elif predicted == 2:
        st.write("Your Disease is :  GERD")
    elif predicted == 3:
        st.write("Your Disease is :  Chronic cholestasis")
    elif predicted == 4:
        st.write("Your Disease is :  Drug Reaction")
    elif predicted == 5:
        st.write("Your Disease is :  Peptic ulcer disease")
    elif predicted == 6:
        st.write("Your Disease is :  AIDS")
    elif predicted == 7:
        st.write("Your Disease is :  Diabetes")
    elif predicted == 8:
        st.write("Your Disease is :  Gastroenteritis")
    elif predicted == 9:
        st.write("Your Disease is :  Bronchial Asthma")
    elif predicted == 10:
        st.write("Your Disease is :  Hypertension")
    elif predicted == 11:
        st.write("Your Disease is :  Migraine")
    elif predicted == 12:
        st.write("Your Disease is :  Cervical spondylosis")
    elif predicted == 13:
        st.write("Your Disease is :  Paralysis (brain hemorrhage)")
    elif predicted == 14:
        st.write("Your Disease is :  Jaundice")
    elif predicted == 15:
        st.write("Your Disease is :  Malaria")
    elif predicted == 16:
        st.write("Your Disease is :  Chicken pox")
    elif predicted == 17:
        st.write("Your Disease is :  Dengue")
    elif predicted == 18:
        st.write("Your Disease is :  Typhoid")
    elif predicted == 19:
        st.write("Your Disease is :  Hepatitis A")
    elif predicted == 20:
        st.write("Your Disease is :  Hepatitis B")
    elif predicted == 21:
        st.write("Your Disease is :  Hepatitis C")
    elif predicted == 22:
        st.write("Your Disease is :  Hepatitis D")
    elif predicted == 23:
        st.write("Your Disease is :  Hepatitis E")
    elif predicted == 24:
        st.write("Your Disease is :  Alcoholic hepatitis")
    elif predicted == 25:
        st.write("Your Disease is :  Tuberculosis")
    elif predicted == 26:
        st.write("Your Disease is :  Common Cold")
    elif predicted == 27:
        st.write("Your Disease is :  Pneumonia")
    elif predicted == 28:
        st.write("Your Disease is :  Dimorphic hemmorhoids (piles)")
    elif predicted == 29:
        st.write("Your Disease is :  Heart attack")
    elif predicted == 30:
        st.write("Your Disease is :  Varicose veins")
    elif predicted == 31:
        st.write("Your Disease is :  Hypothyroidism")
    elif predicted == 32:
        st.write("Your Disease is :  Hyperthyroidism")
    elif predicted == 33:
        st.write("Your Disease is :  Hypoglycemia")
    elif predicted == 34:
        st.write("Your Disease is :  Osteoarthristis")
    elif predicted == 35:
        st.write("Your Disease is :  Arthritis")
    elif predicted == 36:
        st.write("Your Disease is :  (vertigo) Paroymsal Positional Vertigo")
    elif predicted == 37:
        st.write("Your Disease is :  Acne")
    elif predicted == 38:
        st.write("Your Disease is :  Urinary tract infection")
    elif predicted == 39:
        st.write("Your Disease is :  Psoriasis")
    elif predicted == 40:
        st.write("Your Disease is :  Impetigo")
    else:
        st.write("Unknown Disease")
    st.header("Accuracy Score")
    accuracy = accuracy_score(y, clf4.predict(X))
    st.text("Accuracy: {:.2f}%".format(accuracy * 100))

def KNN():
    from sklearn.neighbors import KNeighborsClassifier
    knn=KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=2)
    knn=knn.fit(X,np.ravel(y))

    from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
    y_pred=knn.predict(X_test)
    print("KNN")
    print("Accuracy")
    print(accuracy_score(y_test, y_pred))
    print(accuracy_score(y_test, y_pred,normalize=False))
    print("Confusion matrix")
    conf_matrix=confusion_matrix(y_test,y_pred)
    print(conf_matrix)

    psymptoms = [symp1,symp2,symp3,symp4,symp5]

    for k in range(0,len(l1)):
        for z in psymptoms:
            if(z==l1[k]):
                l2[k]=1

    inputtest = [l2]
    predict = knn.predict(inputtest)
    predicted=predict[0]
    if predicted == 0:
        st.write("Your Disease is :  Fungal infection")
    elif predicted == 1:
        st.write("Your Disease is :  Allergy")
    elif predicted == 2:
        st.write("Your Disease is :  GERD")
    elif predicted == 3:
        st.write("Your Disease is :  Chronic cholestasis")
    elif predicted == 4:
        st.write("Your Disease is :  Drug Reaction")
    elif predicted == 5:
        st.write("Your Disease is :  Peptic ulcer disease")
    elif predicted == 6:
        st.write("Your Disease is :  AIDS")
    elif predicted == 7:
        st.write("Your Disease is :  Diabetes")
    elif predicted == 8:
        st.write("Your Disease is :  Gastroenteritis")
    elif predicted == 9:
        st.write("Your Disease is :  Bronchial Asthma")
    elif predicted == 10:
        st.write("Your Disease is :  Hypertension")
    elif predicted == 11:
        st.write("Your Disease is :  Migraine")
    elif predicted == 12:
        st.write("Your Disease is :  Cervical spondylosis")
    elif predicted == 13:
        st.write("Your Disease is :  Paralysis (brain hemorrhage)")
    elif predicted == 14:
        st.write("Your Disease is :  Jaundice")
    elif predicted == 15:
        st.write("Your Disease is :  Malaria")
    elif predicted == 16:
        st.write("Your Disease is :  Chicken pox")
    elif predicted == 17:
        st.write("Your Disease is :  Dengue")
    elif predicted == 18:
        st.write("Your Disease is :  Typhoid")
    elif predicted == 19:
        st.write("Your Disease is :  Hepatitis A")
    elif predicted == 20:
        st.write("Your Disease is :  Hepatitis B")
    elif predicted == 21:
        st.write("Your Disease is :  Hepatitis C")
    elif predicted == 22:
        st.write("Your Disease is :  Hepatitis D")
    elif predicted == 23:
        st.write("Your Disease is :  Hepatitis E")
    elif predicted == 24:
        st.write("Your Disease is :  Alcoholic hepatitis")
    elif predicted == 25:
        st.write("Your Disease is :  Tuberculosis")
    elif predicted == 26:
        st.write("Your Disease is :  Common Cold")
    elif predicted == 27:
        st.write("Your Disease is :  Pneumonia")
    elif predicted == 28:
        st.write("Your Disease is :  Dimorphic hemmorhoids (piles)")
    elif predicted == 29:
        st.write("Your Disease is :  Heart attack")
    elif predicted == 30:
        st.write("Your Disease is :  Varicose veins")
    elif predicted == 31:
        st.write("Your Disease is :  Hypothyroidism")
    elif predicted == 32:
        st.write("Your Disease is :  Hyperthyroidism")
    elif predicted == 33:
        st.write("Your Disease is :  Hypoglycemia")
    elif predicted == 34:
        st.write("Your Disease is :  Osteoarthristis")
    elif predicted == 35:
        st.write("Your Disease is :  Arthritis")
    elif predicted == 36:
        st.write("Your Disease is :  (vertigo) Paroymsal Positional Vertigo")
    elif predicted == 37:
        st.write("Your Disease is :  Acne")
    elif predicted == 38:
        st.write("Your Disease is :  Urinary tract infection")
    elif predicted == 39:
        st.write("Your Disease is :  Psoriasis")
    elif predicted == 40:
        st.write("Your Disease is :  Impetigo")
    else:
        st.write("Unknown Disease")
    st.header("Accuracy Score")
    accuracy = accuracy_score(y, knn.predict(X))
    st.text("Accuracy: {:.2f}%".format(accuracy * 100))


def NaiveBayes():
    from sklearn.naive_bayes import GaussianNB
    gnb = GaussianNB()
    gnb=gnb.fit(X,np.ravel(y))

    from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
    y_pred=gnb.predict(X_test)
    print("Naive Bayes")
    print("Accuracy")
    print(accuracy_score(y_test, y_pred))
    print(accuracy_score(y_test, y_pred,normalize=False))
    print("Confusion matrix")
    conf_matrix=confusion_matrix(y_test,y_pred)
    print(conf_matrix)

    psymptoms = [symp1,symp2,symp3,symp4,symp5]
    for k in range(0,len(l1)):
        for z in psymptoms:
            if(z==l1[k]):
                l2[k]=1

    inputtest = [l2]
    predict = gnb.predict(inputtest)
    predicted=predict[0]
    if predicted == 0:
        st.write("Your Disease is :  Fungal infection")
    elif predicted == 1:
        st.write("Your Disease is :  Allergy")
    elif predicted == 2:
        st.write("Your Disease is :  GERD")
    elif predicted == 3:
        st.write("Your Disease is :  Chronic cholestasis")
    elif predicted == 4:
        st.write("Your Disease is :  Drug Reaction")
    elif predicted == 5:
        st.write("Your Disease is :  Peptic ulcer disease")
    elif predicted == 6:
        st.write("Your Disease is :  AIDS")
    elif predicted == 7:
        st.write("Your Disease is :  Diabetes")
    elif predicted == 8:
        st.write("Your Disease is :  Gastroenteritis")
    elif predicted == 9:
        st.write("Your Disease is :  Bronchial Asthma")
    elif predicted == 10:
        st.write("Your Disease is :  Hypertension")
    elif predicted == 11:
        st.write("Your Disease is :  Migraine")
    elif predicted == 12:
        st.write("Your Disease is :  Cervical spondylosis")
    elif predicted == 13:
        st.write("Your Disease is :  Paralysis (brain hemorrhage)")
    elif predicted == 14:
        st.write("Your Disease is :  Jaundice")
    elif predicted == 15:
        st.write("Your Disease is :  Malaria")
    elif predicted == 16:
        st.write("Your Disease is :  Chicken pox")
    elif predicted == 17:
        st.write("Your Disease is :  Dengue")
    elif predicted == 18:
        st.write("Your Disease is :  Typhoid")
    elif predicted == 19:
        st.write("Your Disease is :  Hepatitis A")
    elif predicted == 20:
        st.write("Your Disease is :  Hepatitis B")
    elif predicted == 21:
        st.write("Your Disease is :  Hepatitis C")
    elif predicted == 22:
        st.write("Your Disease is :  Hepatitis D")
    elif predicted == 23:
        st.write("Your Disease is :  Hepatitis E")
    elif predicted == 24:
        st.write("Your Disease is :  Alcoholic hepatitis")
    elif predicted == 25:
        st.write("Your Disease is :  Tuberculosis")
    elif predicted == 26:
        st.write("Your Disease is :  Common Cold")
    elif predicted == 27:
        st.write("Your Disease is :  Pneumonia")
    elif predicted == 28:
        st.write("Your Disease is :  Dimorphic hemmorhoids (piles)")
    elif predicted == 29:
        st.write("Your Disease is :  Heart attack")
    elif predicted == 30:
        st.write("Your Disease is :  Varicose veins")
    elif predicted == 31:
        st.write("Your Disease is :  Hypothyroidism")
    elif predicted == 32:
        st.write("Your Disease is :  Hyperthyroidism")
    elif predicted == 33:
        st.write("Your Disease is :  Hypoglycemia")
    elif predicted == 34:
        st.write("Your Disease is :  Osteoarthristis")
    elif predicted == 35:
        st.write("Your Disease is :  Arthritis")
    elif predicted == 36:
        st.write("Your Disease is :  (vertigo) Paroymsal Positional Vertigo")
    elif predicted == 37:
        st.write("Your Disease is :  Acne")
    elif predicted == 38:
        st.write("Your Disease is :  Urinary tract infection")
    elif predicted == 39:
        st.write("Your Disease is :  Psoriasis")
    elif predicted == 40:
        st.write("Your Disease is :  Impetigo")
    else:
        st.write("Unknown Disease")
    st.header("Accuracy Score")
    accuracy = accuracy_score(y, gnb.predict(X))
    st.text("Accuracy: {:.2f}%".format(accuracy * 100))




if st.button("Decision Tree"):
    DecisionTree()
if st.button("Random Forest"):
    randomforest()
if st.button("Knn"):
    KNN()
if st.button("Naive Bayes"):
    NaiveBayes()
