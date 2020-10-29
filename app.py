
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
import numpy as np 
import pandas as pd
df=pd.read_csv(r"train_LZdllcl.csv")
df.education.replace({np.nan:"Bachelor's",
                     },inplace=True)
df.previous_year_rating.replace({np.nan:1},inplace=True)
first_list=["department","education","gender","recruitment_channel","previous_year_rating","length_of_service","KPIs_met >80%","awards_won?","avg_training_score","is_promoted"]
first=df[first_list]
le = LabelEncoder()
first['department'] = le.fit_transform(df.department)
first['education'] = le.fit_transform(df.education)
first['gender'] = le.fit_transform(df.gender)
first['recruitment_channel'] = le.fit_transform(df.recruitment_channel)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
first.avg_training_score = scaler.fit_transform(first)
X=first.drop("is_promoted",axis=1)
y=first.is_promoted
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier()
model.fit(X_train, y_train)
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def main():
    st.title("Hackathon Project")
    html_temp = """
    <div style="background-color:#931465  ;padding:10px">
    <h2 style="color:white;text-align:center;">Promotion Eligibility Tester</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    #1'st feature
    activities=['Sales & Marketing', 'Operations', 'Technology', 'Analytics','R&D', 'Procurement', 'Finance', 'HR', 'Legal']
    dept=st.selectbox('Select Your department',activities)
    #st.subheader(option)
    #2nd feature
    activities1=["Master's & above", "Bachelor's", 'Below Secondary']
    edu=st.selectbox('Your education',activities1)
    #3rd feature 
    activities2=["Male","Female"]
    gender=st.selectbox('Gender',activities2)
    #4th feature
    activities3=['sourcing', 'other', 'referred']
    Rec_channel=st.selectbox('Your Recruitment Channel',activities3)
    #5th feature
    activities4=[1,2,3,4,5]
    Last_year_ratings=st.selectbox('Last year ratings',activities4)
    #6th feature
    yrs_of_service=st.slider('Select yrs of service', 1.0, 37.0)
    #7th feature 
    activities6=["Yes","NO"]
    kpi=st.selectbox("KPIs_met >80%",activities6)
    #8th feature
    activities7=["Yes","NO"]
    Awards=st.selectbox("Any Awards won?",activities7)
    #9th feature 
    Avg_training_score=st.slider('Avg_training_score', 39.0, 99.0)
    #pw=st.slider('Select Petal Width', 0.0, 10.0)
    #inputs=[[sl,sw,pl,pw]]
    #ft 1
    ft={"Sales & Marketing":7,"Operations":4,"Procurement":8,"Technology":5,"Analytics":0,"Finance":1,"HR":2,"Legal":3,"R&D":6}
    retdept=ft[dept]
    #ft2
    ft2={"Bachelor's":0,"Master's & above":2,"Below Secondary":1}
    retedu=ft2[edu]
    #ft3
    ft3={"Male":1,"Female":0}
    retgender=ft3[gender] 
    #ft4
    ft4={'sourcing':2, 'other':0, 'referred':1} 
    retRec_channel=ft4[Rec_channel]
    #ft5
    Last_year_ratings
    #ft6
    retyrs_of_service=round(yrs_of_service)
    #ft7
    ft7={"Yes":1,"NO":0}
    retkpi=ft7[kpi]
    #ft8
    ft8={"Yes":1,"NO":0}
    retAwards=ft8[Awards]
    #ft9
    retAvg_training_score=round(Avg_training_score)
    #Creating data_frame to fit the model
    df=pd.DataFrame({"retdept":retdept,"retedu":retedu,"retgender":retgender,
    "retRec_channel":retRec_channel,"Last_year_ratings":Last_year_ratings,
    "retyrs_of_service":retyrs_of_service,"retkpi":retkpi,"retAwards":retAwards,"retAvg_training_score":retAvg_training_score
    },index=[1])
    #transforming the needed variable
    scaler = MinMaxScaler()
    df.retAvg_training_score = scaler.fit_transform(df)
    a=model.predict(df)
    safe_html="""  
      <div style="background-color:#F4D03F;padding:10px >
       <h2 style="color:white;text-align:center;"> Congrats you will be promoted </h2>
       </div>
    """
    danger_html="""  
      <div style="background-color:#F08080;padding:10px >
       <h2 style="color:black ;text-align:center;"> Unfortunately You are not eligible </h2>
       </div>
    """
    
    
    if st.button('Classify'):
        if a==1:
            st.markdown(safe_html,unsafe_allow_html=True)
        else :
            st.markdown(danger_html,unsafe_allow_html=True)
       

if __name__=='__main__':
    main()
