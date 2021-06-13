import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
def welcome():
    print("Welcome to Salary Prediction System")
    print("Press ENTER key to proceed")
    input()
def checkcsv():
    csv_files=[]
    cur_dir=os.getcwd()
    content_list=os.listdir(cur_dir)
    for x in content_list:
        if x.split('.')[-1]=='csv':
            csv_files.append(x)
    if len(csv_files)==0:
        return 'No csv file in the directory'
    else:
        return csv_files
def display_and_select_csv(csv_files):
    i=0
    for file_name in csv_files:
        print(i,'...',file_name)
        i+=1
    return csv_files[int(input("Select file to create ML model"))]
def graph(X_train,Y_train,regressionObject,X_test,Y_test,Y_pred):
    plt.scatter(X_train,Y_train,color='red',label='training data')
    plt.plot(X_train,regressionObject.predict(X_train),color='blue',label='Best Fit')
    plt.scatter(X_test,Y_test,color='green',label='test data')
    plt.scatter(X_test,Y_pred,color='black',label='Pred test data')
    plt.title("Salary vs Experience")
    plt.xlabel('Years of Experience')
    plt.ylabel('Salary')
    plt.legend()
    plt.show()
def main():
    welcome()
    try:
        csv_files=checkcsv()
        if csv_files=='No csv file in the directory':
            raise FileNotFoundError('No csv file in the directory')
        csv_file=display_and_select_csv(csv_files)
        print(csv_file,'is selected')
        print('Reading csv file')
        print('Creating Dataset')
        dataset=pd.read_csv(csv_file)
        print('Dataset created')
        X=dataset.iloc[:,:-1].values
        Y=dataset.iloc[:,-1].values
        s=float(input("Enter test data size (between 0 and 1)") )
        X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=s)
        print("Model creation in progression")
        regressionObject=LinearRegression()
        regressionObject.fit(X_train,Y_train)
        print("Model is created")
        print("Press ENTER key to predict test data in trained model")
        input()

        Y_pred=regressionObject.predict(X_test)
        i=0
        print(X_test,'  ...',Y_test,'  ...',Y_pred)
        while i<len(X_test):
            print(X_test[i],'...',Y_test[i],'...',Y_pred[i])
            i+=1
        print("Press ENTER key to see above result in graphical format")
        input()
        graph(X_train,Y_train, regressionObject, X_test, Y_test, Y_pred)
        r2=r2_score(Y_test,Y_pred)
        print("Our model is %2.2f%% accurate" %(r2*100))

        print("Now you can predict salary of an employee using our model")
        print("\nEnter experience in years of the candidates, separated by comma")

        exp=[float(e) for e in input().split(',')]
        ex=[]
        for x in exp:
            ex.append([x])
        experience =np.array(ex)
        salaries=regressionObject.predict(experience)

        plt.scatter(experience,salaries,color='black')
        plt.xlabel('Years of Experience')
        plt.ylabel('Salaries')
        plt.show()

        d=pd.DataFrame({'Experience':exp,'Salaries':salaries})
        print(d)
        
    except FileNotFoundError:
        print('No csv file in the directory')
        print("Press ENTER key to exit")
        input()
        exit()

if __name__=="__main__":
    main()
    input()
