import tkinter as tk
import tkinter.tix as tkx
from tkinter import messagebox
import pandas as pd
import numpy as np
from numpy import round


#function to import and view the data frame and display in another window all of the revelent infomation, size, variables, etc...
def get_df():
    global df
    #remove the diagaonals and repeated half of the correlation coefficients
    def get_redundant_pairs(df):
        pairs_to_drop = set()
        cols = df.columns
        for i in range(0, df.shape[1]):
            for j in range(0, i+1):
                pairs_to_drop.add((cols[i], cols[j]))
        return pairs_to_drop
    
    #from the unique correlations find the 3 most correlating variable pairs
    def get_top_abs_correlations(df, n=3):
    
        au_corr = df.corr().abs().unstack()
        labels_to_drop = get_redundant_pairs(df)
        au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
        return au_corr[0:n]

    #import the data frame as a global variable to be used
    df=pd.read_csv('diabetes.csv')

    df1=df.head()
    global cols, header_grid, header_rows, header_cols
    cols=[]
    for col in df.columns:
        cols.append(str(col))
    header_array=df1.to_numpy()
    header_list=header_array.tolist()
    #create the grid that can be indexed and printed out onto canvas
    header_grid=[cols,
                header_list[0],
                header_list[1],
                header_list[2],
                header_list[3],
                header_list[4]]
    header_rows=len(header_grid)
    header_cols=len(header_grid[0])
    
    #get data types as a grid to draw on the canvas
    df_dt=df.dtypes
    dt_list=df_dt.tolist()
    dt_grid=[cols, dt_list]
        
    #decide width of each box and height of each box
    MARGIN=10
    cell_width=140
    cell_height=26
    WIN_HEIGHT=MARGIN * 2 + cell_height * header_rows
    WIN_WIDTH=MARGIN * 2 + cell_width * header_cols
        
    #create a new window that displays all of the infomation about the data frame
    global df_infomation_window
    #get relative x and y values of the original window
    x = window.winfo_x()
    y = window.winfo_y()
    #open a new pop up window and re position it relative to the original window
    #log in window must be global to allow it to be destroyed after clicking
    global log_in_window,df_infomation_window
    df_infomation_window = tk.Toplevel(window)
    df_infomation_window.geometry("+%d+%d" % (x+ 1.63*windowWidth,  y-50))
    df_infomation_window.title('Dataframe Infomation')
    df_infomation_window.resizable(0,0)  #(x,y)
    
    global frm_new_window, main_canvas,df_lbl
    #frame to hold all of the labels of infomation
    frm_new_window=tk.Frame(master=df_infomation_window, bg=main_colors[1])
    frm_new_window.grid(row=0, column=0)
    
    #title
    df_lbl=tk.Label(master=frm_new_window, text='Header and infomation for the data imported to be analysed',font=('Arial', 18), fg=main_colors[0], bg=main_colors[1])
    df_lbl.grid(row=0, column=0)
    
    #df head drawn onto a canvas
    main_canvas=tk.Canvas(master=frm_new_window, width=WIN_WIDTH, height=WIN_HEIGHT, bg=main_colors[1])
    main_canvas.grid(row=1, column=0)
    for i in range(header_rows):
        for j in range(header_cols):
            element=header_grid[i][j]
            #draw a box around each element
            x0 = MARGIN + j * cell_width + 1
            y0 = MARGIN + i * cell_height + 1
            x1 = MARGIN + (j + 1) * cell_width - 1
            y1 = MARGIN + (i + 1) * cell_height - 1
            main_canvas.create_rectangle(x0, y0, x1, y1, outline='black', width=1, tags='outlines')    
            #draw out the text value for what is in each part of the data frame
            x_loc=MARGIN + (j * cell_width + cell_width/2)
            y_loc=MARGIN + (i * cell_height + cell_height/2)
            main_canvas.create_text(x_loc, y_loc, text=element,font=("Lucida Sans Typewriter", 11), tags='df', fill=main_colors[0])
    
    #get the type of data for each column
    df_dt_lbl=tk.Label(master=frm_new_window, text='\nData types for each column',font=('Arial', 14), fg=main_colors[0], bg=main_colors[1])
    df_dt_lbl.grid(row=2, column=0)
    main_canvas2=tk.Canvas(master=frm_new_window, width=WIN_WIDTH, height=WIN_HEIGHT/2, bg=main_colors[1])
    main_canvas2.grid(row=3, column=0)
    for i in range(len(dt_grid)):
        for j in range(header_cols):
            element=dt_grid[i][j]
            #draw a box around each element
            x0 = MARGIN + j * cell_width + 1
            y0 = MARGIN + i * cell_height + 1
            x1 = MARGIN + (j + 1) * cell_width - 1
            y1 = MARGIN + (i + 1) * cell_height - 1
            main_canvas2.create_rectangle(x0, y0, x1, y1, outline='black', width=1, tags='outlines')    
            #draw out the text value for what is in each part of the data frame
            x_loc=MARGIN + (j * cell_width + cell_width/2)
            y_loc=MARGIN + (i * cell_height + cell_height/2)
            main_canvas2.create_text(x_loc, y_loc, text=element,font=("Lucida Sans Typewriter", 11), tags='dt', fill=main_colors[0])   
    
    #set up empty lists with the label to display the infomation correctly
    df_means=['Mean']
    df_medians=['Median']
    df_mins=['Minimum']
    df_max=['Maximum']
    df_std=['Stan Dev.']
    stats_cols=[' ']
    for col in df.columns:
        stats_cols.append(str(col))
        
    #get statistics for each column and display here
    for i in range(len(cols)-1):
        #find mean of each column
        means=round(df[cols[i]].mean(),3)
        df_means.append(means)
        #find median of each column
        medians=df[cols[i]].median()
        df_medians.append(medians)
        #find min of each column
        mins=df[cols[i]].min()
        df_mins.append(mins)
        #find max of each column
        maxs=df[cols[i]].max()
        df_max.append(maxs)
        #find the standard deviation of each column
        stds=round(df[cols[i]].std(),3)
        df_std.append(stds)
    #create a grid that can be iterated over to draw out
    stats_grid=[stats_cols[:-1], df_means, df_std, df_mins, df_medians, df_max]

    #get the statistics for each column of the data frame
    stats_lbl=tk.Label(master=frm_new_window, text='\nStatistics for each column',font=('Arial', 14), fg=main_colors[0], bg=main_colors[1])
    stats_lbl.grid(row=4, column=0)
    main_canvas3=tk.Canvas(master=frm_new_window, width=WIN_WIDTH, height=WIN_HEIGHT, bg=main_colors[1])
    main_canvas3.grid(row=5, column=0)
    for i in range(len(stats_grid)):
        for j in range(len(stats_grid[0])):
            element=stats_grid[i][j]
            #draw a box around each element
            x0 = MARGIN + j * cell_width + 1
            y0 = MARGIN + i * cell_height + 1
            x1 = MARGIN + (j + 1) * cell_width - 1
            y1 = MARGIN + (i + 1) * cell_height - 1
            main_canvas3.create_rectangle(x0, y0, x1, y1, outline='black', width=1, tags='outlines')    
            #draw out the text value for what is in each part of the data frame
            x_loc=MARGIN + (j * cell_width + cell_width/2)
            y_loc=MARGIN + (i * cell_height + cell_height/2)
            main_canvas3.create_text(x_loc, y_loc, text=element,font=("Lucida Sans Typewriter", 11), tags='stats', fill=main_colors[0])   
    
    
    #df datapoints
    col = df.shape[1]
    row = df.shape[0]
    df_row_lbl=tk.Label(master=frm_new_window, text='\nData points: '+str(row),font=('Arial', 14), fg=main_colors[0], bg=main_colors[1])
    df_row_lbl.grid(row=6, column=0)
    #df_col_lbl=tk.Label(master=frm_new_window, text='Number of Columns: '+str(col), font=('Arial', 14),fg=main_colors[0], bg=main_colors[1])
    #df_col_lbl.grid(row=5, column=0)

    #check for empty values of the data frame
    check_for_null=df.isnull().values.any()
    if check_for_null==True: #there is an empty value in the df.
        df_null_lbl=tk.Label(master=frm_new_window, text='\nNull values found', font=('Arial', 14),fg=main_colors[0], bg=main_colors[1])
        df_null_lbl.grid(row=7, column=0)
    else:
        df_null_lbl=tk.Label(master=frm_new_window, text='\nNo null values found', font=('Arial', 14),fg=main_colors[0], bg=main_colors[1])
        df_null_lbl.grid(row=8, column=0)
    
    #from the correlation matrix return the top 3 correlating inputs, either negative or positive
    df_corr1_lbl=tk.Label(master=frm_new_window, text='\nTop correlating pairs',font=('Arial', 14), fg=main_colors[0], bg=main_colors[1])
    df_corr1_lbl.grid(row=9, column=0)
    df_corr2_lbl=tk.Label(master=frm_new_window, text=str(get_top_abs_correlations(df, 3)),font=('Arial', 14), fg=main_colors[0], bg=main_colors[1])
    df_corr2_lbl.grid(row=10, column=0)
    
#function to retreive weather the user wants the data to be scaled or not
def set_scalar():
    global scalar_choice
    scalar_choice=variable.get()
    #print(scalar_choice)
    #update data frame label to show user it has now been scaled 
    df_lbl['text']='The data has now been scaled and will update once the predicted column has been chosen.'

#function that will idenify the predicted column based on the index the user picks, starting at 0 as usual practice.
#this will also rescale the data depending on user choice above as well.
def set_outcome_column():
    global user_entry
    global df, y, X, scalar_choice
    user_entry=int(set_y_ent.get())
    from sklearn.preprocessing import LabelEncoder
    #Convert all the non-numeric columns to numeric
    for column in df.columns:
      if df[column].dtype == np.float64 or df[column].dtype == np.int64:
        continue
    else:
        df[column] = LabelEncoder().fit_transform(df[column])
        
    #delete the input
    #set_y_ent.delete(0, "end")
    #set_y_ent.insert(0, "")
    #print(user_entry)
    #user has decided that the data needs to be scaled.
    if scalar_choice=="YES":
        #scale the data so values are between 0 and 1
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        #y is the predicted column and will stay the same
        y = df.iloc[:, user_entry]
        #print(y)
        #X is the values input to predict y. This is all of the columns that arent y
        all_cols=list(range(df.shape[1]))
        all_cols.remove(user_entry)
        X = df.iloc[:, all_cols]
        X = scaler.fit_transform(X) #rescale the data 
        #print(X)
        
        #create a grid of just the top 5  values that are the same as the header
        scaled_header=np.round(X[:5],3).tolist()
        outcomes=y[:5].tolist()
        for i in range(5):
            scaled_header[i].append(outcomes[i])       
        #create the grid that can be indexed and printed out onto canvas
        header_scaled_grid=[cols,
                scaled_header[0],
                scaled_header[1],
                scaled_header[2],
                scaled_header[3],
                scaled_header[4]]
        header_scaled_rows=len(header_scaled_grid)
        header_scaled_cols=len(header_scaled_grid[0])
        
        #decide width of each box and height of each box
        MARGIN=10
        cell_width=140
        cell_height=26
        WIN_HEIGHT=MARGIN * 2 + cell_height * header_scaled_rows
        WIN_WIDTH=MARGIN * 2 + cell_width * header_scaled_cols 
         
        #update data frame label to show user it has now been scaled 
        df_lbl['text']='The data has been scaled as shown below. The predicted column is in blue'
         
        #df scaled head drawn onto the canvas and updated
        main_canvas=tk.Canvas(master=frm_new_window, width=WIN_WIDTH, height=WIN_HEIGHT, bg=main_colors[1])
        main_canvas.grid(row=1, column=0)
        for i in range(header_scaled_rows):
            for j in range(header_scaled_cols):
                element=header_scaled_grid[i][j]
                #draw a box around each element
                x0 = MARGIN + j * cell_width + 1
                y0 = MARGIN + i * cell_height + 1
                x1 = MARGIN + (j + 1) * cell_width - 1
                y1 = MARGIN + (i + 1) * cell_height - 1
                #highlight the users selected column
                if j == user_entry:
                    main_canvas.create_rectangle(x0, y0, x1, y1, outline='blue', width=2, tags='selected_column') 
                else:
                    main_canvas.create_rectangle(x0, y0, x1, y1, outline='black', width=1, tags='outlines')    
                #draw out the text value for what is in each part of the data frame
                x_loc=MARGIN + (j * cell_width + cell_width/2)
                y_loc=MARGIN + (i * cell_height + cell_height/2)
                main_canvas.create_text(x_loc, y_loc, text=element,font=("Lucida Sans Typewriter", 11), tags='df', fill=main_colors[0])
        main_canvas.update()
        
    #either the user has not selected a value or has selected NO
    else:
        #y is the predicted column
        y = df.iloc[:, user_entry]
        #print(y)
        #X is the values input to predict y. This is all of the columns that arent y
        all_cols=list(range(df.shape[1]))
        all_cols.remove(user_entry)
        X = df.iloc[:, all_cols]
        #print(X)
        #redrawn the infomation with a ghighlighed selected outcome box
        #decide width of each box and height of each box
        MARGIN=10
        cell_width=140
        cell_height=26
        WIN_HEIGHT=MARGIN * 2 + cell_height * header_rows
        WIN_WIDTH=MARGIN * 2 + cell_width * header_cols
        #update data frame label to show user it has now been scaled 
        df_lbl['text']='The data has not been scaled. The predicted column is in blue'
        #update the main canvas
        main_canvas=tk.Canvas(master=frm_new_window, width=WIN_WIDTH, height=WIN_HEIGHT, bg=main_colors[1])
        main_canvas.grid(row=1, column=0)
        for i in range(header_rows):
            for j in range(header_cols):
                element=header_grid[i][j]
                #draw a box around each element
                x0 = MARGIN + j * cell_width + 1
                y0 = MARGIN + i * cell_height + 1
                x1 = MARGIN + (j + 1) * cell_width - 1
                y1 = MARGIN + (i + 1) * cell_height - 1
                #highlight the users selected coloumn
                if j == user_entry:
                    main_canvas.create_rectangle(x0, y0, x1, y1, outline='blue', width=2, tags='selected_column') 
                else:
                    main_canvas.create_rectangle(x0, y0, x1, y1, outline='black', width=1, tags='outlines')  
                #draw out the text value for what is in each part of the data frame
                x_loc=MARGIN + (j * cell_width + cell_width/2)
                y_loc=MARGIN + (i * cell_height + cell_height/2)
                main_canvas.create_text(x_loc, y_loc, text=element,font=("Lucida Sans Typewriter", 11), tags='df', fill=main_colors[0])
        main_canvas.update()
        
#function to allow enter to be pressed to submit the training split proportion
def enter_pressed_set_outcome_column(event):
    #run the function that is the same as the button pressed
    set_outcome_column() 
    
#function to create the training and test splits based on users input to the entry box
def set_test_train_split():
    user_entry=float(test_train_ent.get())
    #delete the input
    #test_train_ent.delete(0, "end")
    #test_train_ent.insert(0, "")
    #check that the user has entered a valid value for the training split proportion
    if 0<user_entry<1:
        global testing_split
        testing_split=user_entry
    else:
        messagebox.showinfo(title="ERROR", message="Training split must be between 0 and 1(0% and 100% of the data)")
    
    #Split arrays or matrices into random train and test subsets
    from sklearn.model_selection import train_test_split 
    global X, y, X_train, X_test, y_train, y_test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testing_split)
    #print(X_train, X_test)
    #print(y_train, y_test)
    df_infomation_window.title('Dataframe Infomation - Testing split set to '+str(testing_split*100)+'%')
    
#function to allow enter to be pressed to submit the training split proportion
def enter_pressed_set_test_train_split(event):
    #run the function that is the same as the button pressed
    set_test_train_split()
    
#check how accurate the training was by comparing the test classification vs predicted classifcation
#element by element, either correct or not and return a %.
def check_model_accuracy(y_test, y_pred):
    import numpy as np
    y_test=list(y_test)
    counter=0
    for i in range(len(y_pred)):
      if y_test[i]==y_pred[i]:
        counter=counter+1
      else:
        continue
    accuracy=np.round((counter/len(y_test))*100, 2)
    return accuracy

#input the predicted values from pred(x-test) from the model and the actual values from the y-test split.
def percentage_outcomes_predicted(predicted, y_test):
    #input prediction, and y-test
    predicted_sample=list(predicted)
    actual_sample=list(y_test)
    #print('Actual   ', actual_sample)
    #print('Predicted',predicted_sample,)
    #find the out comes of the predicted column
    outcomes=list(set(y_test))
    #create a predcited stats dictionary that is empty to store infomation about each outcome.
    predicted_stats={"Outcome":[],
                    "Predicted Total":[],
                    "Actual Total":[],
                    "Correct prediction":[],
                    "Predicted Accuracy":[]
                    }
    for i in range(len(outcomes)):
        counter=0
        predicted_stats['Outcome'].append(outcomes[i])
        predicted_total=predicted_sample.count(outcomes[i])
        predicted_stats['Predicted Total'].append(predicted_total)
        actual_total=actual_sample.count(outcomes[i])
        predicted_stats['Actual Total'].append(actual_total)
        for j in range(len(predicted_sample)):
            if predicted_sample[j]==outcomes[i] and predicted_sample[j]==actual_sample[j]:
                counter+=1
        predicted_stats['Correct prediction'].append(counter)  
        if predicted_total==0:
            predicted_stats["Predicted Accuracy"].append(0)
        else:
            predicted_stats["Predicted Accuracy"].append((round(counter/predicted_total,4))*100)
    #print(predicted_stats)
    outcomes_dict_to_list=[]
    outcomes_accuracy=[]
    for i in range(len(predicted_stats['Outcome'])):
        outcomes_dict_to_list.append(predicted_stats['Outcome'][i])
        outcomes_accuracy.append(predicted_stats['Predicted Accuracy'][i])
        
    #print(outcomes_dict_to_list) 
    #print(outcomes_accuracy)
    #zip all outcomes to an accuarcy value of that prediction
    predicted_outcome_accuracy=list(zip(outcomes_dict_to_list,outcomes_accuracy,))
    return predicted_outcome_accuracy

#function to work out the best decisoin_tree_variables for the highest accuracy for the given data.
def dtc_best_parameters(x_train, x_test, y_train, y_test, max_depth):
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import cross_val_score
    #run the DTC algorithm for different max_depth values to see where the most efficient value is
    #there are three different methods that can be run to along side this. This means running the classifier
    #alot of times and comparing the accuracy at each value to return the most efficient model.
    features=['auto', 'sqrt', 'log2']
    dt_cvs = []
    dt_accuracy=[]
    dt_trainscores=[]
    depth=[]
    algo=[]
    for i in features:
        for n in range(1, max_depth):  
            decisiontree = DecisionTreeClassifier(max_depth=n, max_features=i)
            decisiontree.fit(x_train, y_train)
            y_pred=decisiontree.predict(x_test)
            cvs=np.round(cross_val_score(decisiontree,x_train,y_train).mean()*100,2)    #CVS training score
            trainscore=np.round(decisiontree.score(x_train, y_train)*100,2)  #training score
            dt_cvs.append(cvs.mean())
            dt_accuracy.append(check_model_accuracy(y_test,y_pred))
            dt_trainscores.append(trainscore)
            depth.append(n-1)
            algo.append(i)
    #zip together the max and the n values to return the max and the corresponding value of n
    dtc_CVSscoresN=list(zip(dt_cvs, depth, algo))    #CROSS VALUE SCORE
    dtc_PredaccuracyN=list(zip(dt_accuracy, depth, algo))   #ACCURACY
    dtc_TrainaccuracyN=list(zip(dt_trainscores, depth, algo))   #TRAIN SCORE
  
    #identify the best prediction and pull the relevent paramaters
    mostAccuracte=max(dtc_PredaccuracyN)  #best predicted accuracy method
    best_n=mostAccuracte[1]
    best_method=mostAccuracte[2]
    bestCVS=max(dtc_CVSscoresN)   #best CVS method 
    best_n_cvs=bestCVS[1]
    best_method_cvs=bestCVS[2]
    return best_n, best_method

#function to find the best classifier infomation and store the infomation as a variable that can be called back to later on
def dtc_run_classifier():
    global X_train, X_test, y_train, y_test
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import classification_report
    from numpy import round
    #initial classifier checks are False, once a classifier has run it will change to true. 
    #Find BEST parameters using the function that runs through all of the possible combinations
    global best_dtc_depth, best_dtc_method
    best_dtc_depth, best_dtc_method=dtc_best_parameters(X_train, X_test, y_train, y_test, 20)  
    #print('Best Max Depth for decision tree:', best_dtc_depth)
    #print('Best method for decision tree:', best_dtc_method)
    
    global decisiontree_pred
    #use the best parameters found to create an optimsed model
    decisiontree = DecisionTreeClassifier(max_depth=best_dtc_depth, max_features=best_dtc_method) #initiatie the learning models
    decisiontree.fit(X_train, y_train)          #train the model on the train variables
    decisiontree_pred=decisiontree.predict(X_test)   #create a prediction array based on the model 
    #using the best model variables determine the testing and training accuracy of the given data
    global decisiontree_train_accuracy, decisiontree_test_accuracy
    decisiontree_train_accuracy=decisiontree.score(X_train, y_train)
    decisiontree_test_accuracy=decisiontree.score(X_test, y_test)

#formatting and variable names for the window that the classifier will use to display infomation
def draw_dtc_window():
    from numpy import round
    global scalar_choice,user_entry,decision_tree_window
    #create a new window that displays all of the infomation about the classifier and the best out comes
    decision_tree_window=tk.Toplevel(window)
    x = window.winfo_x()
    y = window.winfo_y()
    decision_tree_window.geometry("+%d+%d" % (x,  y+3*windowHeight-35))
    decision_tree_window.title('Decision Tree Infomation - Testing split '+str(testing_split*100)+'%')
    decision_tree_window.resizable(1,0)  #(x,y)
    #frame to hold all of the labels of infomation
    frm_new_window2=tk.Frame(master=decision_tree_window, bg=main_colors[1])
    frm_new_window2.grid(row=0, column=0)
    
    title_lbl=tk.Label(master=frm_new_window2, text='Decision Tree Infomation and Results', font=('Arial', 18), fg=main_colors[0], bg=main_colors[1])
    title_lbl.grid(row=0,column=0)
    
    infomation_lbl=tk.Label(master=frm_new_window2, text='Predicted column '+str(user_entry)+' Scaled Data: '+scalar_choice,
                            font=('Arial', 14), fg=main_colors[0], bg=main_colors[1])
    infomation_lbl.grid(row=1,column=0)
    
    best_variables_lbl=tk.Label(master=frm_new_window2, text='Best Max Depth for decision tree: '+str(best_dtc_depth)+', Best method for decision tree: '+str(best_dtc_method),
                    font=('Arial', 14),fg=main_colors[0], bg=main_colors[1])
    best_variables_lbl.grid(row=2,column=0)
    
    train_acc_lbl=tk.Label(master=frm_new_window2, text='Training accuracy: '+ str(round(decisiontree_train_accuracy,4)),
                    font=('Arial', 14),fg=main_colors[0], bg=main_colors[1])
    train_acc_lbl.grid(row=3,column=0)
    
    test_acc_lbl=tk.Label(master=frm_new_window2, text='Testing accuracy: '+str(round(decisiontree_test_accuracy,4)),
                    font=('Arial', 14),fg=main_colors[0], bg=main_colors[1])
    test_acc_lbl.grid(row=4,column=0)
    
    #find the accuracy of each outcome, use the values here to satore variables about the classifier and siaply accordingly
    global outcome_lists_decision_tree
    from numpy import round
    outcome_accuracy=np.round(percentage_outcomes_predicted(decisiontree_pred, y_test),2)
    #print(outcome_accuracy)
    outcome_lists_decision_tree=[['Outcome','Predicted Accuracy']]
    for i in range(len(outcome_accuracy)):
        outcome_lists_decision_tree.append(list(outcome_accuracy[i]))
    #set up grid dimensions so that the grid can be drawn
    outcome_rows=len(outcome_lists_decision_tree)
    outcome_cols=len(outcome_lists_decision_tree[0])
    
    #decide width of each box and height of each box
    MARGIN=10
    cell_width=175
    cell_height=26
    outcome_HEIGHT=MARGIN * 2 + cell_height * outcome_rows
    outcome_WIDTH=MARGIN * 2 + cell_width * outcome_cols
    
    #predicted outcome accuracy draw on
    outcome_canvas=tk.Canvas(master=frm_new_window2, width=outcome_WIDTH, height=outcome_HEIGHT, bg=main_colors[1])
    outcome_canvas.grid(row=5, column=0)
    for i in range(outcome_rows):
        for j in range(outcome_cols):
            element=outcome_lists_decision_tree[i][j]
            #draw a box around each element
            x0 = MARGIN + j * cell_width + 1
            y0 = MARGIN + i * cell_height + 1
            x1 = MARGIN + (j + 1) * cell_width - 1
            y1 = MARGIN + (i + 1) * cell_height - 1
            outcome_canvas.create_rectangle(x0, y0, x1, y1, outline='black', width=1, tags='outlines')    
            #draw out the text value for what is in each part of the data frame
            x_loc=MARGIN + (j * cell_width + cell_width/2)
            y_loc=MARGIN + (i * cell_height + cell_height/2)
            outcome_canvas.create_text(x_loc, y_loc, text=element,font=("Lucida Sans Typewriter", 11), tags='outcomes', fill=main_colors[0])    
                    
#run the function and check if it has ran before and output the corect infomation window.
def decision_tree_classification_button():
    #if a check is false allow the script to run otherwise do not run again.
    global decision_tree_check
    if decision_tree_check==False:
        #run classifier
        dtc_run_classifier()  
        #create the pop up window that displays the infomation
        draw_dtc_window()
        #change the value of the check to true to keep track that the user has  ran this classifier already.
        decision_tree_check=True
    else:
        #create a check box that allows the user to check if they want to run the entire script since it can take a while for larger data sets.
        model_check=messagebox.askyesno(title='Warning!', message='You have already ran this classifier. Click yes to run again.\nThis can change your models current accuracy.')            
        #user has chosen to re run the model.
        if model_check==True:
            decision_tree_window.destroy()
            #run classifier
            dtc_run_classifier()  
            #create the pop up window that displays the infomation
            draw_dtc_window()
            #change the value of the check to true to keep track that the user has  ran this classifier already.
            decision_tree_check=True
    
    #check that all classifiers have been run, if they have then update the window with the new decision tree classification
    if all_classifiers_check==True:
        all_classifiers_window.destroy()
        draw_all_classifiers()
  
#function to find the best classifier infomation and store the infomation as a variable that can be called back to later on
def log_reg_run_classifier():
    #no parameters that can be variable to find best accuracy just initate the variable
    global X_train, X_test, y_train, y_test
    from sklearn.linear_model import LogisticRegression
    from numpy import round
    global  logisticregression_pred
    logisticregression = LogisticRegression(max_iter=250)
    logisticregression.fit(X_train, y_train)
    logisticregression_pred=logisticregression.predict(X_test)
    
    global scalar_choice,user_entry
    global logisticregression_train_accuracy, logisticregression_test_accuracy
    #calculate the scores for train and test accuracy
    logisticregression_train_accuracy=logisticregression.score(X_train, y_train)
    logisticregression_test_accuracy=logisticregression.score(X_test, y_test)

#function to draw the window
def draw_log_reg_window():
    global log_regression_window
    from numpy import round
    #create a new window that displays all of the infomation about the classifier and the best out comes
    log_regression_window=tk.Toplevel(window)
    x = window.winfo_x()
    y = window.winfo_y()
    log_regression_window.geometry("+%d+%d" % (x,  y+3*windowHeight-35))
    log_regression_window.title('Logistic Regression Infomation - Testing split '+str(testing_split*100)+'%')
    log_regression_window.resizable(0,0)  #(x,y)
    #frame to hold all of the labels of infomation
    frm_new_window2=tk.Frame(master=log_regression_window, bg=main_colors[1])
    frm_new_window2.grid(row=0, column=0)
    
    title_lbl=tk.Label(master=frm_new_window2, text='Logistic Regression and Results', font=('Arial', 18), fg=main_colors[0], bg=main_colors[1])
    title_lbl.grid(row=0,column=0)
    
    infomation_lbl=tk.Label(master=frm_new_window2, text='Predicted column '+str(user_entry)+' Scaled Data: '+scalar_choice,
                            font=('Arial', 14), fg=main_colors[0], bg=main_colors[1])
    infomation_lbl.grid(row=1,column=0)
        
    train_acc_lbl=tk.Label(master=frm_new_window2, text='Training accuracy: '+ str(round(logisticregression_train_accuracy,4)),
                    font=('Arial', 14),fg=main_colors[0], bg=main_colors[1])
    train_acc_lbl.grid(row=2,column=0)
    
    test_acc_lbl=tk.Label(master=frm_new_window2, text='Testing accuracy: '+str(round(logisticregression_test_accuracy,4)),
                    font=('Arial', 14),fg=main_colors[0], bg=main_colors[1])
    test_acc_lbl.grid(row=3,column=0)
    
    #find the accuracy of each outcome, use the values here to satore variables about the classifier and siaply accordingly
    global outcome_lists_log_regression
    from numpy import round
    outcome_accuracy=np.round(percentage_outcomes_predicted(logisticregression_pred, y_test),2)
    #print(outcome_accuracy)
    outcome_lists_log_regression=[['Outcome','Predicted Accuracy']]
    for i in range(len(outcome_accuracy)):
        outcome_lists_log_regression.append(list(outcome_accuracy[i]))
    #set up grid dimensions so that the grid can be drawn
    outcome_rows=len(outcome_lists_log_regression)
    outcome_cols=len(outcome_lists_log_regression[0])
    
    #decide width of each box and height of each box
    MARGIN=10
    cell_width=175
    cell_height=26
    outcome_HEIGHT=MARGIN * 2 + cell_height * outcome_rows
    outcome_WIDTH=MARGIN * 2 + cell_width * outcome_cols
    
    #predicted outcome accuracy draw on
    outcome_canvas=tk.Canvas(master=frm_new_window2, width=outcome_WIDTH, height=outcome_HEIGHT, bg=main_colors[1])
    outcome_canvas.grid(row=4, column=0)
    for i in range(outcome_rows):
        for j in range(outcome_cols):
            element=outcome_lists_log_regression[i][j]
            #draw a box around each element
            x0 = MARGIN + j * cell_width + 1
            y0 = MARGIN + i * cell_height + 1
            x1 = MARGIN + (j + 1) * cell_width - 1
            y1 = MARGIN + (i + 1) * cell_height - 1
            outcome_canvas.create_rectangle(x0, y0, x1, y1, outline='black', width=1, tags='outlines')    
            #draw out the text value for what is in each part of the data frame
            x_loc=MARGIN + (j * cell_width + cell_width/2)
            y_loc=MARGIN + (i * cell_height + cell_height/2)
            outcome_canvas.create_text(x_loc, y_loc, text=element,font=("Lucida Sans Typewriter", 11), tags='outcomes', fill=main_colors[0]) 

#function to calculate the logistic regression classifier
def log_regression_classification_button():
    #initial classifier checks are False, once a classifier has run it will change to true. 
    #if a check is false allow the script to run otherwise do not run again.
    global log_regression_check
    if log_regression_check == False: 
        #run the classifier
        log_reg_run_classifier()
        #draw the window
        draw_log_reg_window()
        #change the check value to allow computer to know this classifier has run         
        log_regression_check=True
    else:
        #create a check box that allows the user to check if they want to run the entire script since it can take a while for larger data sets.
        model_check=messagebox.askyesno(title='Warning!', message='You have already ran this classifier. Click yes to run again.\nThis can change your models current accuracy.')            
        #user has chosen to re run the model.
        if model_check==True:
            log_regression_window.destroy()
            #run the classifier
            log_reg_run_classifier()
            #draw the window
            draw_log_reg_window()
            #set value to allow program to know that the classifier has been run already.
            log_regression_check=True

    #check that all classifiers have been run, if they have then update the window with the new decision tree classification
    if all_classifiers_check==True:
        all_classifiers_window.destroy()
        draw_all_classifiers()

#function to find the best classifier infomation and store the infomation as a variable that can be called back to later on
def bernoulliNB_run_classifier():
    from numpy import round
    #no parameters that can be variable to find best accuracy just initate the variable
    global X_train, X_test, y_train, y_test
    global scalar_choice,user_entry
    
    #BernoulliNB has no variable parameters
    from sklearn.naive_bayes import BernoulliNB
    global bernoulli_naiveBayes_pred
    bernoulli_naiveBayes = BernoulliNB()
    bernoulli_naiveBayes.fit(X_train, y_train)
    bernoulli_naiveBayes_pred=bernoulli_naiveBayes.predict(X_test)
    
    #find the training and testing accuracy for the bernoulli nb model.
    global bernoulliNB_train_accuracy,bernoulliNB_test_accuracy
    bernoulliNB_train_accuracy=bernoulli_naiveBayes.score(X_train, y_train)
    bernoulliNB_test_accuracy=bernoulli_naiveBayes.score(X_test, y_test)

#formatting and variable names for the window that the classifier will use to display infomation
def draw_bernoulliNB_window():
    from numpy import round
    global bernoulliNB_window
    #create a new window that displays all of the infomation about the classifier and the best out comes
    bernoulliNB_window=tk.Toplevel(window)
    x = window.winfo_x()
    y = window.winfo_y()
    bernoulliNB_window.geometry("+%d+%d" % (x,  y+3*windowHeight-35))
    bernoulliNB_window.title('Bernoulli N-Bayes Infomation - Testing split '+str(testing_split*100)+'%')
    bernoulliNB_window.resizable(1,0)  #(x,y)
    #frame to hold all of the labels of infomation
    frm_new_window2=tk.Frame(master=bernoulliNB_window, bg=main_colors[1])
    frm_new_window2.grid(row=0, column=0)
    
    title_lbl=tk.Label(master=frm_new_window2, text='Bernoulli N-Bayes and Results', font=('Arial', 18), fg=main_colors[0], bg=main_colors[1])
    title_lbl.grid(row=0,column=0)
    
    infomation_lbl=tk.Label(master=frm_new_window2, text='Predicted column '+str(user_entry)+' Scaled Data: '+scalar_choice,
                            font=('Arial', 14), fg=main_colors[0], bg=main_colors[1])
    infomation_lbl.grid(row=1,column=0)
        
    train_acc_lbl=tk.Label(master=frm_new_window2, text='Training accuracy: '+ str(round(bernoulliNB_train_accuracy,4)),
                    font=('Arial', 14),fg=main_colors[0], bg=main_colors[1])
    train_acc_lbl.grid(row=2,column=0)
    
    test_acc_lbl=tk.Label(master=frm_new_window2, text='Testing accuracy: '+str(round(bernoulliNB_test_accuracy,4)),
                    font=('Arial', 14),fg=main_colors[0], bg=main_colors[1])
    test_acc_lbl.grid(row=3,column=0)
    
    #find the accuracy of each outcome, use the values here to satore variables about the classifier and siaply accordingly
    global outcome_lists_bernoulliNB
    outcome_accuracy=np.round(percentage_outcomes_predicted(bernoulli_naiveBayes_pred, y_test),2)
    #print(outcome_accuracy)
    outcome_lists_bernoulliNB=[['Outcome','Predicted Accuracy']]
    for i in range(len(outcome_accuracy)):
        outcome_lists_bernoulliNB.append(list(outcome_accuracy[i]))
    #set up grid dimensions so that the grid can be drawn
    outcome_rows=len(outcome_lists_bernoulliNB)
    outcome_cols=len(outcome_lists_bernoulliNB[0])
    
    #decide width of each box and height of each box
    MARGIN=10
    cell_width=175
    cell_height=26
    outcome_HEIGHT=MARGIN * 2 + cell_height * outcome_rows
    outcome_WIDTH=MARGIN * 2 + cell_width * outcome_cols
    
    #predicted outcome accuracy draw on
    outcome_canvas=tk.Canvas(master=frm_new_window2, width=outcome_WIDTH, height=outcome_HEIGHT, bg=main_colors[1])
    outcome_canvas.grid(row=4, column=0)
    for i in range(outcome_rows):
        for j in range(outcome_cols):
            element=outcome_lists_bernoulliNB[i][j]
            #draw a box around each element
            x0 = MARGIN + j * cell_width + 1
            y0 = MARGIN + i * cell_height + 1
            x1 = MARGIN + (j + 1) * cell_width - 1
            y1 = MARGIN + (i + 1) * cell_height - 1
            outcome_canvas.create_rectangle(x0, y0, x1, y1, outline='black', width=1, tags='outlines')    
            #draw out the text value for what is in each part of the data frame
            x_loc=MARGIN + (j * cell_width + cell_width/2)
            y_loc=MARGIN + (i * cell_height + cell_height/2)
            outcome_canvas.create_text(x_loc, y_loc, text=element,font=("Lucida Sans Typewriter", 11), tags='outcomes', fill=main_colors[0])
                
#function to calculate the bernoulli N-bayes classifier
def bernoulliNB_classification_button():
    from numpy import round
    #initial classifier checks are False, once a classifier has run it will change to true. 
    #if a check is false allow the script to run otherwise do not run again.
    global bernoulliNB_check
    if bernoulliNB_check==False:
        bernoulliNB_run_classifier()
        draw_bernoulliNB_window()
        #change the check value to allow computer to know this classifier has run                
        bernoulliNB_check=True
    else:
        #create a check box that allows the user to check if they want to run the entire script since it can take a while for larger data sets.
        model_check=messagebox.askyesno(title='Warning!', message='You have already ran this classifier. Click yes to run again.\nThis can change your models current accuracy.')            
        #user has chosen to re run the model.
        if model_check==True:
            bernoulliNB_window.destroy()
            bernoulliNB_run_classifier()
            draw_bernoulliNB_window()
            #change the check value to allow computer to know this classifier has run                
            bernoulliNB_check=True

    #check that all classifiers have been run, if they have then update the window with the new decision tree classification
    if all_classifiers_check==True:
        all_classifiers_window.destroy()
        draw_all_classifiers()
        
#function to find the best classifier infomation and store the infomation as a variable that can be called back to later on
def gaussianNB_run_classifier():
    from numpy import round
    #no parameters that can be variable to find best accuracy just initate the variable
    global X_train, X_test, y_train, y_test
    global scalar_choice,user_entry,gaussianNB_pred
    #GaussianNB has no major variable parameters to test.
    from sklearn.naive_bayes import GaussianNB
    gaussianNB = GaussianNB() 
    gaussianNB.fit(X_train, y_train)  
    gaussianNB_pred=gaussianNB.predict(X_test)

    #find the training and testing accuracy for the bernoulli nb model.
    global gaussianNB_train_accuracy, gaussianNB_test_accuracy
    gaussianNB_train_accuracy=gaussianNB.score(X_train, y_train)
    gaussianNB_test_accuracy=gaussianNB.score(X_test, y_test)
    
#formatting and variable names for the window that the classifier will use to display infomation
def draw_gaussianNB_window():
    from numpy import round
    global gaussianNB_window
    #create a new window that displays all of the infomation about the classifier and the best out comes
    gaussianNB_window=tk.Toplevel(window)
    x = window.winfo_x()
    y = window.winfo_y()
    gaussianNB_window.geometry("+%d+%d" % (x,  y+3*windowHeight-35))
    gaussianNB_window.title('Gaussian N-Bayes Infomation - Testing split '+str(testing_split*100)+'%')
    gaussianNB_window.resizable(1,0)  #(x,y)
    #frame to hold all of the labels of infomation
    frm_new_window2=tk.Frame(master=gaussianNB_window, bg=main_colors[1])
    frm_new_window2.grid(row=0, column=0)
    
    title_lbl=tk.Label(master=frm_new_window2, text='Gaussian N-Bayes and Results', font=('Arial', 18), fg=main_colors[0], bg=main_colors[1])
    title_lbl.grid(row=0,column=0)
    
    infomation_lbl=tk.Label(master=frm_new_window2, text='Predicted column '+str(user_entry)+' Scaled Data: '+scalar_choice,
                            font=('Arial', 14), fg=main_colors[0], bg=main_colors[1])
    infomation_lbl.grid(row=1,column=0)
        
    train_acc_lbl=tk.Label(master=frm_new_window2, text='Training accuracy: '+ str(round(gaussianNB_train_accuracy,4)),
                    font=('Arial', 14),fg=main_colors[0], bg=main_colors[1])
    train_acc_lbl.grid(row=2,column=0)
    
    test_acc_lbl=tk.Label(master=frm_new_window2, text='Testing accuracy: '+str(round(gaussianNB_test_accuracy,4)),
                    font=('Arial', 14),fg=main_colors[0], bg=main_colors[1])
    test_acc_lbl.grid(row=3,column=0) 
    
    #find the accuracy of each outcome, use the values here to satore variables about the classifier and siaply accordingly
    global outcome_lists_gaussianNB
    outcome_accuracy=np.round(percentage_outcomes_predicted(gaussianNB_pred, y_test),2)
    #print(outcome_accuracy)
    outcome_lists_gaussianNB=[['Outcome','Predicted Accuracy']]
    for i in range(len(outcome_accuracy)):
        outcome_lists_gaussianNB.append(list(outcome_accuracy[i]))
    #set up grid dimensions so that the grid can be drawn
    outcome_rows=len(outcome_lists_gaussianNB)
    outcome_cols=len(outcome_lists_gaussianNB[0])
    
    #decide width of each box and height of each box
    MARGIN=10
    cell_width=175
    cell_height=26
    outcome_HEIGHT=MARGIN * 2 + cell_height * outcome_rows
    outcome_WIDTH=MARGIN * 2 + cell_width * outcome_cols
    
    #predicted outcome accuracy draw on
    outcome_canvas=tk.Canvas(master=frm_new_window2, width=outcome_WIDTH, height=outcome_HEIGHT, bg=main_colors[1])
    outcome_canvas.grid(row=4, column=0)
    for i in range(outcome_rows):
        for j in range(outcome_cols):
            element=outcome_lists_gaussianNB[i][j]
            #draw a box around each element
            x0 = MARGIN + j * cell_width + 1
            y0 = MARGIN + i * cell_height + 1
            x1 = MARGIN + (j + 1) * cell_width - 1
            y1 = MARGIN + (i + 1) * cell_height - 1
            outcome_canvas.create_rectangle(x0, y0, x1, y1, outline='black', width=1, tags='outlines')    
            #draw out the text value for what is in each part of the data frame
            x_loc=MARGIN + (j * cell_width + cell_width/2)
            y_loc=MARGIN + (i * cell_height + cell_height/2)
            outcome_canvas.create_text(x_loc, y_loc, text=element,font=("Lucida Sans Typewriter", 11), tags='outcomes', fill=main_colors[0])  
    
#function to calculate the gaussian N-bayes classifier
def gaussianNB_classification_button():
    #initial classifier checks are False, once a classifier has run it will change to true. 
    #if a check is false allow the script to run otherwise do not run again.
    global gaussianNB_check
    if gaussianNB_check==False:
        gaussianNB_run_classifier()
        draw_gaussianNB_window()
        #change the check value to allow computer to know this classifier has run                
        gaussianNB_check=True
    else:
        #create a check box that allows the user to check if they want to run the entire script since it can take a while for larger data sets.
        model_check=messagebox.askyesno(title='Warning!', message='You have already ran this classifier. Click yes to run again.\nThis can change your models current accuracy.')            
        #user has chosen to re run the model.
        if model_check==True:
            gaussianNB_window.destroy()
            gaussianNB_run_classifier()
            draw_gaussianNB_window()
            gaussianNB_check=True   
            
    #check that all classifiers have been run, if they have then update the window with the new decision tree classification
    if all_classifiers_check==True:
        all_classifiers_window.destroy()
        draw_all_classifiers()

#function to find the best classifier infomation and store the infomation as a variable that can be called back to later on
def passive_aggressive_run_classifier():
    from numpy import round
    #no parameters that can be variable to find best accuracy just initate the variable
    global X_train, X_test, y_train, y_test
    global scalar_choice,user_entry,passiveAggressive_pred
    # Passive Aggressive no variable parameters
    from sklearn.linear_model import PassiveAggressiveClassifier
    passiveAggressive = PassiveAggressiveClassifier()
    passiveAggressive.fit(X_train, y_train)
    passiveAggressive_pred=passiveAggressive.predict(X_test)
    
    #find the training and testing accuracy for the passive agressive classifier
    global passive_agressive_train_accuracy,passive_agressive_test_accuracy
    passive_agressive_train_accuracy=passiveAggressive.score(X_train, y_train)
    passive_agressive_test_accuracy=passiveAggressive.score(X_test, y_test)
    
#formatting and variable names for the window that the classifier will use to display infomation
def draw_passive_aggressive_window():
    from numpy import round
    global passive_aggressive_window
    #create a new window that displays all of the infomation about the classifier and the best out comes
    passive_aggressive_window=tk.Toplevel(window)
    x = window.winfo_x()
    y = window.winfo_y()
    passive_aggressive_window.geometry("+%d+%d" % (x,  y+3*windowHeight-35))
    passive_aggressive_window.title('Passive Agressive Classifier Infomation - Testing split '+str(testing_split*100)+'%')
    passive_aggressive_window.resizable(1,0)  #(x,y)
    #frame to hold all of the labels of infomation
    frm_new_window2=tk.Frame(master=passive_aggressive_window, bg=main_colors[1])
    frm_new_window2.grid(row=0, column=0)
    
    title_lbl=tk.Label(master=frm_new_window2, text='Passive Agressive Classifier and Results', font=('Arial', 18), fg=main_colors[0], bg=main_colors[1])
    title_lbl.grid(row=0,column=0)
    
    infomation_lbl=tk.Label(master=frm_new_window2, text='Predicted column '+str(user_entry)+' Scaled Data: '+scalar_choice,
                            font=('Arial', 14), fg=main_colors[0], bg=main_colors[1])
    infomation_lbl.grid(row=1,column=0)
        
    train_acc_lbl=tk.Label(master=frm_new_window2, text='Training accuracy: '+ str(round(passive_agressive_train_accuracy,4)),
                    font=('Arial', 14),fg=main_colors[0], bg=main_colors[1])
    train_acc_lbl.grid(row=2,column=0)
    
    test_acc_lbl=tk.Label(master=frm_new_window2, text='Testing accuracy: '+str(round(passive_agressive_test_accuracy,4)),
                    font=('Arial', 14),fg=main_colors[0], bg=main_colors[1])
    test_acc_lbl.grid(row=3,column=0)

    #find the accuracy of each outcome, use the values here to satore variables about the classifier and siaply accordingly
    global outcome_lists_passive_agressive
    outcome_accuracy=np.round(percentage_outcomes_predicted(passiveAggressive_pred, y_test),2)
    #print(outcome_accuracy)
    outcome_lists_passive_agressive=[['Outcome','Predicted Accuracy']]
    for i in range(len(outcome_accuracy)):
        outcome_lists_passive_agressive.append(list(outcome_accuracy[i]))
    #set up grid dimensions so that the grid can be drawn
    outcome_rows=len(outcome_lists_passive_agressive)
    outcome_cols=len(outcome_lists_passive_agressive[0])
    
    #decide width of each box and height of each box
    MARGIN=10
    cell_width=175
    cell_height=26
    outcome_HEIGHT=MARGIN * 2 + cell_height * outcome_rows
    outcome_WIDTH=MARGIN * 2 + cell_width * outcome_cols
    
    #predicted outcome accuracy draw on
    outcome_canvas=tk.Canvas(master=frm_new_window2, width=outcome_WIDTH, height=outcome_HEIGHT, bg=main_colors[1])
    outcome_canvas.grid(row=4, column=0)
    for i in range(outcome_rows):
        for j in range(outcome_cols):
            element=outcome_lists_passive_agressive[i][j]
            #draw a box around each element
            x0 = MARGIN + j * cell_width + 1
            y0 = MARGIN + i * cell_height + 1
            x1 = MARGIN + (j + 1) * cell_width - 1
            y1 = MARGIN + (i + 1) * cell_height - 1
            outcome_canvas.create_rectangle(x0, y0, x1, y1, outline='black', width=1, tags='outlines')    
            #draw out the text value for what is in each part of the data frame
            x_loc=MARGIN + (j * cell_width + cell_width/2)
            y_loc=MARGIN + (i * cell_height + cell_height/2)
            outcome_canvas.create_text(x_loc, y_loc, text=element,font=("Lucida Sans Typewriter", 11), tags='outcomes', fill=main_colors[0])   
    
#function to calculate the passive agressive classifier
def passive_aggressive_classification_button():
    #initial classifier checks are False, once a classifier has run it will change to true. 
    #if a check is false allow the script to run otherwise do not run again.
    global passive_aggressive_check
    if passive_aggressive_check==False:
        passive_aggressive_run_classifier()
        draw_passive_aggressive_window()
        #change the check value to allow computer to know this classifier has run        
        passive_aggressive_check=True        
    else:
        #create a check box that allows the user to check if they want to run the entire script since it can take a while for larger data sets.
        model_check=messagebox.askyesno(title='Warning!', message='You have already ran this classifier. Click yes to run again.\nThis can change your models current accuracy.')            
        #user has chosen to re run the model.
        if model_check==True:
            passive_aggressive_window.destroy()
            passive_aggressive_run_classifier()
            draw_passive_aggressive_window()
            #change the check value to allow computer to know this classifier has run        
            passive_aggressive_check=True   
    
    #check that all classifiers have been run, if they have then update the window with the new decision tree classification
    if all_classifiers_check==True:
        all_classifiers_window.destroy()
        draw_all_classifiers()

#function to find the best variables for the highest accuracy
def SVC_best_parameters(x_train, x_test, y_train, y_test, max_degree):
    from sklearn.svm import SVC
    from sklearn.model_selection import cross_val_score
    #run the SVC algorithm for different methods to find the most efficent method
    kernels=['linear', 'poly', 'rbf', 'sigmoid']
    svc_cvs = []
    svc_accuracy=[]
    svc_trainscores=[]
    degree=[]
    kernel=[]
    for i in kernels:
        for n in range(1, max_degree+1):  
            svc = SVC(kernel=i, degree=n)
            svc.fit(x_train, y_train)
            y_pred=svc.predict(x_test)
            cvs=np.round(cross_val_score(svc,x_train,y_train).mean()*100,2)
            trainscore=np.round(svc.score(x_train, y_train)*100,2)  #training score
            svc_cvs.append(cvs.mean())
            svc_accuracy.append(check_model_accuracy(y_test,y_pred))
            svc_trainscores.append(trainscore)
            degree.append(n)
            kernel.append(i)
    #zip together the max and the n values to return the max and the corresponding value of n
    svc_CVSscores=list(zip(svc_cvs, degree, kernel))    #CROSS VALUE SCORE
    svc_Predaccuracy=list(zip(svc_accuracy, degree, kernel))   #ACCURACY
    #print(svc_Predaccuracy)
    svc_Trainaccuracy=list(zip(svc_trainscores, degree, kernel))   #TRAIN SCORE
    #identify the best prediction and pull the relevent paramaters
    mostAccuracte=max(svc_Predaccuracy)  #best predicted accuracy method
    best_degree=mostAccuracte[1]
    best_method=mostAccuracte[2]
    bestCVS=max(svc_CVSscores)   #best CVS method 
    best_degree_cvs=bestCVS[1]
    best_method_cvs=bestCVS[2]
    return best_degree, best_method
        
#function to find the best classifier infomation and store the infomation as a variable that can be called back to later on
def SVC_run_classifier():
    from numpy import round
    global X_train, X_test, y_train, y_test
    global scalar_choice,user_entry
    global SVC_check, best_SVC_degree, best_SVC_method,svc_classifier_pred
    best_SVC_degree, best_SVC_method=SVC_best_parameters(X_train, X_test, y_train, y_test, 6) #Find BEST parameters
  
    #initiate the model using the best parameters found
    from sklearn.svm import SVC
    svc_classifier = SVC(kernel=best_SVC_method, degree=best_SVC_degree)
    svc_classifier.fit(X_train, y_train)
    svc_classifier_pred=svc_classifier.predict(X_test)
    
    #calculate the training and test accuracy
    global SVC_train_accuracy, SVC_test_accuracy
    SVC_train_accuracy=svc_classifier.score(X_train, y_train)
    SVC_test_accuracy=svc_classifier.score(X_test, y_test)
    
#formatting and variable names for the window that the classifier will use to display infomation
def draw_SVC_window():
    global SVC_window
    #create a new window that displays all of the infomation about the classifier and the best out comes
    SVC_window=tk.Toplevel(window)
    SVC_window.title('SVC Infomation - Testing split '+str(testing_split*100)+'%')
    x = window.winfo_x()
    y = window.winfo_y()
    SVC_window.geometry("+%d+%d" % (x,  y+3*windowHeight-35))
    SVC_window.resizable(1,0)  #(x,y)
    #frame to hold all of the labels of infomation
    frm_new_window2=tk.Frame(master=SVC_window, bg=main_colors[1])
    frm_new_window2.grid(row=0, column=0)
    
    title_lbl=tk.Label(master=frm_new_window2, text='SVC classifier Infomation and Results', font=('Arial', 18), fg=main_colors[0], bg=main_colors[1])
    title_lbl.grid(row=0,column=0)
    
    infomation_lbl=tk.Label(master=frm_new_window2, text='Predicted column '+str(user_entry)+' Scaled Data: '+scalar_choice,
                            font=('Arial', 14), fg=main_colors[0], bg=main_colors[1])
    infomation_lbl.grid(row=1,column=0)
    
    best_variables_lbl=tk.Label(master=frm_new_window2, text='Best Max Degree SVC: '+str(best_SVC_degree)+', Best method for SVC: '+str(best_SVC_method),
                    font=('Arial', 14),fg=main_colors[0], bg=main_colors[1])
    best_variables_lbl.grid(row=2,column=0)
    
    train_acc_lbl=tk.Label(master=frm_new_window2, text='Training accuracy: '+ str(round(SVC_train_accuracy,4)),
                    font=('Arial', 14),fg=main_colors[0], bg=main_colors[1])
    train_acc_lbl.grid(row=3,column=0)
    
    test_acc_lbl=tk.Label(master=frm_new_window2, text='Testing accuracy: '+str(round(SVC_test_accuracy,4)),
                    font=('Arial', 14),fg=main_colors[0], bg=main_colors[1])
    test_acc_lbl.grid(row=4,column=0)
    
    #find the accuracy of each outcome, use the values here to satore variables about the classifier and siaply accordingly
    global outcome_lists_SVC
    outcome_accuracy=np.round(percentage_outcomes_predicted(svc_classifier_pred, y_test),2)
    #print(outcome_accuracy)
    outcome_lists_SVC=[['Outcome','Predicted Accuracy']]
    for i in range(len(outcome_accuracy)):
        outcome_lists_SVC.append(list(outcome_accuracy[i]))
    #set up grid dimensions so that the grid can be drawn
    outcome_rows=len(outcome_lists_SVC)
    outcome_cols=len(outcome_lists_SVC[0])
    
    #decide width of each box and height of each box
    MARGIN=10
    cell_width=175
    cell_height=26
    outcome_HEIGHT=MARGIN * 2 + cell_height * outcome_rows
    outcome_WIDTH=MARGIN * 2 + cell_width * outcome_cols
    
    #predicted outcome accuracy draw on
    outcome_canvas=tk.Canvas(master=frm_new_window2, width=outcome_WIDTH, height=outcome_HEIGHT, bg=main_colors[1])
    outcome_canvas.grid(row=5, column=0)
    for i in range(outcome_rows):
        for j in range(outcome_cols):
            element=outcome_lists_SVC[i][j]
            #draw a box around each element
            x0 = MARGIN + j * cell_width + 1
            y0 = MARGIN + i * cell_height + 1
            x1 = MARGIN + (j + 1) * cell_width - 1
            y1 = MARGIN + (i + 1) * cell_height - 1
            outcome_canvas.create_rectangle(x0, y0, x1, y1, outline='black', width=1, tags='outlines')    
            #draw out the text value for what is in each part of the data frame
            x_loc=MARGIN + (j * cell_width + cell_width/2)
            y_loc=MARGIN + (i * cell_height + cell_height/2)
            outcome_canvas.create_text(x_loc, y_loc, text=element,font=("Lucida Sans Typewriter", 11), tags='outcomes', fill=main_colors[0]) 

#function to calculate the SVC with the best method and the best degree
def SVC_classifier_button():
    #initial classifier checks are False, once a classifier has run it will change to true. 
    #if a check is false allow the script to run otherwise do not run again.
    global SVC_check
    if SVC_check==False:
        #run the classifier
        SVC_run_classifier()
        #draw window
        draw_SVC_window()
        #change the check value to allow computer to know this classifier has run         
        SVC_check=True
    else:
        #create a check box that allows the user to check if they want to run the entire script since it can take a while for larger data sets.
        model_check=messagebox.askyesno(title='Warning!', message='You have already ran this classifier. Click yes to run again.\nThis can change your models current accuracy.')            
        #user has chosen to re run the model.
        if model_check==True:
            SVC_window.destroy()
            #run the classifier
            SVC_run_classifier()
            #draw window
            draw_SVC_window()
            #change the check value to allow computer to know this classifier has run         
            SVC_check=True

    #check that all classifiers have been run, if they have then update the window with the new decision tree classification
    if all_classifiers_check==True:
        all_classifiers_window.destroy()
        draw_all_classifiers()

#RFC-  This can be quite slow to go thourgh many steps
def rfc_best_estimators(x_train, x_test, y_train, y_test, max_estimators, steps):
    #run the RFC algorithm for different values of K to see where the most efficient value is
    #check how accurate the training was by comparing the test classification vs predicted classifcation
    #element by element, either correct or not.
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score
    #run the RFC algorithm for different values of K to see where the most efficient value is
    algorithms=['auto', 'sqrt', 'log2']
    clf_cvs = []
    clf_accuracy=[]
    clf_trainscores=[]
    n_vals=[]
    algo=[]
    for i in algorithms:
        for n in range(1, max_estimators, steps):  #go from 5 to 200 in steps of 10
            clf = RandomForestClassifier(n_estimators=n, max_features=i)    #default randomforestclassifier
            clf.fit(x_train, y_train)
            y_pred=clf.predict(x_test)
            cvs=np.round(cross_val_score(clf,x_train,y_train).mean()*100,2)
            trainscore=np.round(clf.score(x_train, y_train)*100,2)  #training score
            clf_cvs.append(cvs.mean())
            clf_accuracy.append(check_model_accuracy(y_test,y_pred))
            clf_trainscores.append(trainscore)
            n_vals.append(n)
            algo.append(i)
    #zip together the max and the n values to return the max and the corresponding value of n
    clf_CVSscoresN=list(zip(clf_cvs, n_vals, algo))    #CROSS VALUE SCORE
    clf_PredaccuracyN=list(zip(clf_accuracy, n_vals, algo))   #ACCURACY
    clf_TrainaccuracyN=list(zip(clf_trainscores, n_vals, algo))   #TRAIN SCORE
  
    #identify the best prediction and pull the relevent paramaters
    mostAccuracte=max(clf_PredaccuracyN)  #best predicted accuracy method
    best_n=mostAccuracte[1]
    best_method=mostAccuracte[2]
    bestCVS=max(clf_CVSscoresN)   #best CVS method 
    best_n_cvs=bestCVS[1]
    best_method_cvs=bestCVS[2]
    return best_n, best_method

#function to find the best classifier infomation and store the infomation as a variable that can be called back to later on
def rfc_run_classifier():
    global X_train, X_test, y_train, y_test
    global scalar_choice,user_entry
    #find the best parameters
    global best_rfc_n, best_rfc_method, forest_classifier_pred
    best_rfc_n, best_rfc_method=rfc_best_estimators(X_train, X_test, y_train, y_test, 20,2)
  
    #initiate the model using the best parameters found
    from sklearn.ensemble import RandomForestClassifier
    forest_classifier = RandomForestClassifier(n_estimators=best_rfc_n, max_features=best_rfc_method)
    forest_classifier.fit(X_train, y_train)
    forest_classifier_pred=forest_classifier.predict(X_test)
    
    #calculate the training and test accuracy
    global RFC_train_accuracy, RFC_test_accuracy
    RFC_train_accuracy=forest_classifier.score(X_train, y_train)
    RFC_test_accuracy=forest_classifier.score(X_test, y_test)
    
#formatting and variable names for the window that the classifier will use to display infomation
def draw_rfc_window():
    global random_forest_window
    #create a new window that displays all of the infomation about the classifier and the best out comes
    random_forest_window=tk.Toplevel(window)
    random_forest_window.title('RFC Infomation - Testing split '+str(testing_split*100)+'%')
    x = window.winfo_x()
    y = window.winfo_y()
    random_forest_window.geometry("+%d+%d" % (x,  y+3*windowHeight-35))
    random_forest_window.resizable(1,0)  #(x,y)
    #frame to hold all of the labels of infomation
    frm_new_window2=tk.Frame(master=random_forest_window, bg=main_colors[1])
    frm_new_window2.grid(row=0, column=0)
    
    title_lbl=tk.Label(master=frm_new_window2, text='RFC classifier Infomation and Results', font=('Arial', 18), fg=main_colors[0], bg=main_colors[1])
    title_lbl.grid(row=0,column=0)
    
    infomation_lbl=tk.Label(master=frm_new_window2, text='Predicted column '+str(user_entry)+' Scaled Data: '+scalar_choice,
                            font=('Arial', 14), fg=main_colors[0], bg=main_colors[1])
    infomation_lbl.grid(row=1,column=0)
    
    best_variables_lbl=tk.Label(master=frm_new_window2, text='Best N-degree for RFC: '+str(best_rfc_n)+', Best method for RFC: '+str(best_rfc_method),
                    font=('Arial', 14),fg=main_colors[0], bg=main_colors[1])
    best_variables_lbl.grid(row=2,column=0)
    
    train_acc_lbl=tk.Label(master=frm_new_window2, text='Training accuracy: '+ str(round(RFC_train_accuracy,4)),
                    font=('Arial', 14),fg=main_colors[0], bg=main_colors[1])
    train_acc_lbl.grid(row=3,column=0)
    
    test_acc_lbl=tk.Label(master=frm_new_window2, text='Testing accuracy: '+str(round(RFC_test_accuracy,4)),
                    font=('Arial', 14),fg=main_colors[0], bg=main_colors[1])
    test_acc_lbl.grid(row=4,column=0)
    
    #find the accuracy of each outcome, use the values here to satore variables about the classifier and siaply accordingly
    global outcome_lists_forest_classifier
    outcome_accuracy=np.round(percentage_outcomes_predicted(forest_classifier_pred, y_test),2)
    #print(outcome_accuracy)
    outcome_lists_forest_classifier=[['Outcome','Predicted Accuracy']]
    for i in range(len(outcome_accuracy)):
        outcome_lists_forest_classifier.append(list(outcome_accuracy[i]))
    #set up grid dimensions so that the grid can be drawn
    outcome_rows=len(outcome_lists_forest_classifier)
    outcome_cols=len(outcome_lists_forest_classifier[0])
    
    #decide width of each box and height of each box
    MARGIN=10
    cell_width=175
    cell_height=26
    outcome_HEIGHT=MARGIN * 2 + cell_height * outcome_rows
    outcome_WIDTH=MARGIN * 2 + cell_width * outcome_cols
    
    #predicted outcome accuracy draw on
    outcome_canvas=tk.Canvas(master=frm_new_window2, width=outcome_WIDTH, height=outcome_HEIGHT, bg=main_colors[1])
    outcome_canvas.grid(row=5, column=0)
    for i in range(outcome_rows):
        for j in range(outcome_cols):
            element=outcome_lists_forest_classifier[i][j]
            #draw a box around each element
            x0 = MARGIN + j * cell_width + 1
            y0 = MARGIN + i * cell_height + 1
            x1 = MARGIN + (j + 1) * cell_width - 1
            y1 = MARGIN + (i + 1) * cell_height - 1
            outcome_canvas.create_rectangle(x0, y0, x1, y1, outline='black', width=1, tags='outlines')    
            #draw out the text value for what is in each part of the data frame
            x_loc=MARGIN + (j * cell_width + cell_width/2)
            y_loc=MARGIN + (i * cell_height + cell_height/2)
            outcome_canvas.create_text(x_loc, y_loc, text=element,font=("Lucida Sans Typewriter", 11), tags='outcomes', fill=main_colors[0]) 

#function to determine the RFC best method and best max estimators
#this function can take a while to run through sometimes.
def random_forest_classifier_button():
    #initial classifier checks are False, once a classifier has run it will change to true. 
    #if a check is false allow the script to run otherwise do not run again.
    global random_forest_check,best_rfc_n, best_rfc_method
    if random_forest_check==False:
        #run the classifier
        rfc_run_classifier()
        #draw the window
        draw_rfc_window()
        #change the check value to allow computer to know this classifier has run         
        random_forest_check=True
    else:
        #create a check box that allows the user to check if they want to run the entire script since it can take a while for larger data sets.
        model_check=messagebox.askyesno(title='Warning!', message='You have already ran this classifier. Click yes to run again.\nThis can change your models current accuracy.')            
        #user has chosen to re run the model.
        if model_check==True:
            random_forest_window.destroy()
            #run the classifier
            rfc_run_classifier()
            #draw the window
            draw_rfc_window()
            #change the check value to allow computer to know this classifier has run         
            random_forest_check=True
    
    #check that all classifiers have been run, if they have then update the window with the new decision tree classification
    if all_classifiers_check==True:
        all_classifiers_window.destroy()
        draw_all_classifiers()

#KNN BEST FUNCTION
def KNN_best_parameters(x_train, x_test, y_train, y_test, max_k):
    #run the knn algorithm for different values of K to see where the most efficient value is
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import cross_val_score
    #run the Knn algorithm for different values of K to see where the most efficient value is
    algorithms=['auto', 'ball_tree', 'kd_tree', 'brute']
    knn_CVS = []
    knn_accuracy=[]
    knn_trainscores=[]
    k_vals=[]
    algo=[]
    for i in algorithms:
        for k in range(1,max_k+1):
            knn = KNeighborsClassifier(n_neighbors = k,  algorithm=i) #algorithm{‘auto’, ‘ball_tree’, ‘kd_tree’, ‘brute’}
            knn.fit(x_train, y_train)
            y_pred=knn.predict(x_test)
            cvs=np.round(cross_val_score(knn, x_train, y_train).mean()*100,6)  # CVS training score
            trainscore=np.round(knn.score(x_train, y_train)*100,6)  #training score
            knn_CVS.append(cvs.mean())
            knn_accuracy.append(check_model_accuracy(y_test,y_pred))
            knn_trainscores.append(trainscore)
            k_vals.append(k)
            algo.append(i)
    #zip together the max and the k values to return the max and the corresponding value of k
    knn_CVSscoresK=list(zip(knn_CVS, k_vals, algo))               #CVS score
    knn_PredaccuracyK=list(zip(knn_accuracy, k_vals, algo))       #prediction accuracy
    knn_TrainaccuracyK=list(zip(knn_trainscores, k_vals, algo))   #training accuracy
    #print(knn_accuracyK)

    mostAccuracte=max(knn_PredaccuracyK)  #best predicted accuracy method
    best_k=mostAccuracte[1]
    best_method_k=mostAccuracte[2]
    bestCVS=max(knn_CVSscoresK)   #best CVS method 
    best_k_cvs=bestCVS[1]
    best_method_cvs=bestCVS[2]
    return best_k, best_method_k

#function to find the best classifier infomation and store the infomation as a variable that can be called back to later on
def KNN_run_classifier():
    global X_train, X_test, y_train, y_test
    global scalar_choice,user_entry
    global KNN_check,best_knn_neighbours, best_knn_method, knearestclassifier_pred
    #Find BEST parameters
    best_knn_neighbours, best_knn_method=KNN_best_parameters(X_train, X_test, y_train, y_test, 10)
    
    #initiate the model using the best parameters found
    from sklearn.neighbors import KNeighborsClassifier
    knearestclassifier = KNeighborsClassifier(n_neighbors=best_knn_neighbours, algorithm=best_knn_method)
    knearestclassifier.fit(X_train, y_train)
    knearestclassifier_pred=knearestclassifier.predict(X_test)

    #calculate the training and test accuracy
    global KNN_train_accuracy, KNN_test_accuracy
    KNN_train_accuracy=knearestclassifier.score(X_train, y_train)
    KNN_test_accuracy=knearestclassifier.score(X_test, y_test)
    
#formatting and variable names for the window that the classifier will use to display infomation
def draw_KNN_window():
    global KNN_window
    #create a new window that displays all of the infomation about the classifier and the best out comes
    KNN_window=tk.Toplevel(window)
    x = window.winfo_x()
    y = window.winfo_y()
    KNN_window.geometry("+%d+%d" % (x,  y+3*windowHeight-35))
    KNN_window.title('KNN Infomation - Testing split '+str(testing_split*100)+'%')
    KNN_window.resizable(1,0)  #(x,y)
    #frame to hold all of the labels of infomation
    frm_new_window2=tk.Frame(master=KNN_window, bg=main_colors[1])
    frm_new_window2.grid(row=0, column=0)
    
    title_lbl=tk.Label(master=frm_new_window2, text='KNN classifier Infomation and Results', font=('Arial', 18), fg=main_colors[0], bg=main_colors[1])
    title_lbl.grid(row=0,column=0)
    
    infomation_lbl=tk.Label(master=frm_new_window2, text='Predicted column '+str(user_entry)+' Scaled Data: '+scalar_choice,
                            font=('Arial', 14), fg=main_colors[0], bg=main_colors[1])
    infomation_lbl.grid(row=1,column=0)
    
    best_variables_lbl=tk.Label(master=frm_new_window2, text='Best K for KNN: '+str(best_knn_neighbours)+', Best method for KNN: '+str(best_knn_method),
                    font=('Arial', 14),fg=main_colors[0], bg=main_colors[1])
    best_variables_lbl.grid(row=2,column=0)
    
    train_acc_lbl=tk.Label(master=frm_new_window2, text='Training accuracy: '+ str(round(KNN_train_accuracy,4)),
                    font=('Arial', 14),fg=main_colors[0], bg=main_colors[1])
    train_acc_lbl.grid(row=3,column=0)
    
    test_acc_lbl=tk.Label(master=frm_new_window2, text='Testing accuracy: '+str(round(KNN_test_accuracy,4)),
                    font=('Arial', 14),fg=main_colors[0], bg=main_colors[1])
    test_acc_lbl.grid(row=4,column=0)
    
    #find the accuracy of each outcome, use the values here to satore variables about the classifier and siaply accordingly
    global outcome_lists_knn
    outcome_accuracy=np.round(percentage_outcomes_predicted(knearestclassifier_pred, y_test),2)
    #print(outcome_accuracy)
    outcome_lists_knn=[['Outcome','Predicted Accuracy']]
    for i in range(len(outcome_accuracy)):
        outcome_lists_knn.append(list(outcome_accuracy[i]))
    #set up grid dimensions so that the grid can be drawn
    outcome_rows=len(outcome_lists_knn)
    outcome_cols=len(outcome_lists_knn[0])
    
    #decide width of each box and height of each box
    MARGIN=10
    cell_width=175
    cell_height=26
    outcome_HEIGHT=MARGIN * 2 + cell_height * outcome_rows
    outcome_WIDTH=MARGIN * 2 + cell_width * outcome_cols
    
    #predicted outcome accuracy draw on
    outcome_canvas=tk.Canvas(master=frm_new_window2, width=outcome_WIDTH, height=outcome_HEIGHT, bg=main_colors[1])
    outcome_canvas.grid(row=5, column=0)
    for i in range(outcome_rows):
        for j in range(outcome_cols):
            element=outcome_lists_knn[i][j]
            #draw a box around each element
            x0 = MARGIN + j * cell_width + 1
            y0 = MARGIN + i * cell_height + 1
            x1 = MARGIN + (j + 1) * cell_width - 1
            y1 = MARGIN + (i + 1) * cell_height - 1
            outcome_canvas.create_rectangle(x0, y0, x1, y1, outline='black', width=1, tags='outlines')    
            #draw out the text value for what is in each part of the data frame
            x_loc=MARGIN + (j * cell_width + cell_width/2)
            y_loc=MARGIN + (i * cell_height + cell_height/2)
            outcome_canvas.create_text(x_loc, y_loc, text=element,font=("Lucida Sans Typewriter", 11), tags='outcomes', fill=main_colors[0]) 

#function to determine best k for k-nearest neighbour classification
def KNN_classifier_button():
    #initial classifier checks are False, once a classifier has run it will change to true. 
    #if a check is false allow the script to run otherwise do not run again.
    global KNN_check
    if KNN_check==False:
        #run the classifier
        KNN_run_classifier()
        #draw the window
        draw_KNN_window()
        #change the check value to allow computer to know this classifier has run         
        KNN_check=True
    else:
        #create a check box that allows the user to check if they want to run the entire script since it can take a while for larger data sets.
        model_check=messagebox.askyesno(title='Warning!', message='You have already ran this classifier. Click yes to run again.\nThis can change your models current accuracy.')            
        #user has chosen to re run the model.
        if model_check==True:
            KNN_window.destroy()
            #run the classifier
            KNN_run_classifier()
            #draw the window
            draw_KNN_window()
            #change the check value to allow computer to know this classifier has run         
            KNN_check=True

    #check that all classifiers have been run, if they have then update the window with the new decision tree classification
    if all_classifiers_check==True:
        all_classifiers_window.destroy()
        draw_all_classifiers()

#MLP- This can take a while to run despite only being 4 iterations.
def MLP_best_estimators(x_train, x_test, y_train, y_test):
    #run the PAC algorithm for different values of K to see where the most efficient value is
    #check how accurate the training was by comparing the test classification vs predicted classifcation
    #element by element, either correct or not.
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import cross_val_score
    #run the RFC algorithm for different values of K to see where the most efficient value is
    activations=['relu', 'identity', 'logistic', 'tanh']    #un comment to have a slower check
    #activations=['relu', 'identity', 'logistic']
    from sklearn.neural_network import MLPClassifier
    mlp_cvs=[]
    pred_acc=[]
    algo=[]
    for i in activations:
        mlpClassifier = MLPClassifier(max_iter=10000, activation=i, random_state=0)
        mlpClassifier.fit(x_train, y_train)
        #print('MLP', i)  #print check for each loop
        y_pred=mlpClassifier.predict(x_test)
        pred_acc.append(check_model_accuracy(y_test, y_pred))   #accuracy of predictions
        algo.append(i)
    #zip together the max and the n values to return the max and the corresponding value of n
    mlp_Predaccuracy=list(zip(pred_acc, algo))   # PREDICTED ACCURACY
  
    #identify the best prediction and pull the relevent paramaters
    mostAccurate=max(mlp_Predaccuracy)  #best predicted accuracy method
    best_method=mostAccurate[1]
    return best_method

#function to find the best classifier infomation and store the infomation as a variable that can be called back to later on
def MLP_run_classifier():
    global X_train, X_test, y_train, y_test
    global scalar_choice, user_entry, mlp_pred
    global MLP_best_method
    #find the best method for mlp    
    MLP_best_method=MLP_best_estimators(X_train, X_test, y_train, y_test)
  
    #initiate the model using the best parameters found 
    from sklearn.neural_network import MLPClassifier
    mlpClassifier = MLPClassifier(max_iter=10000, activation=MLP_best_method, random_state=0)
    mlpClassifier.fit(X_train, y_train)
    mlp_pred=mlpClassifier.predict(X_test)

    #calculate the training and test accuracy
    global MLP_train_accuracy, MLP_test_accuracy
    MLP_train_accuracy=mlpClassifier.score(X_train, y_train)
    MLP_test_accuracy=mlpClassifier.score(X_test, y_test)
        
#formatting and variable names for the window that the classifier will use to display infomation
def draw_MLP_window():
    global MLP_window
    #create a new window that displays all of the infomation about the classifier and the best out comes
    MLP_window=tk.Toplevel(window)
    x = window.winfo_x()
    y = window.winfo_y()
    MLP_window.geometry("+%d+%d" % (x,  y+3*windowHeight-35))
    MLP_window.title('MLP Infomation - Testing split '+str(testing_split*100)+'%')
    MLP_window.resizable(1,0)  #(x,y)
    #frame to hold all of the labels of infomation
    frm_new_window2=tk.Frame(master=MLP_window, bg=main_colors[1])
    frm_new_window2.grid(row=0, column=0)
    
    title_lbl=tk.Label(master=frm_new_window2, text='MLP classifier Infomation and Results', font=('Arial', 18), fg=main_colors[0], bg=main_colors[1])
    title_lbl.grid(row=0,column=0)
    
    infomation_lbl=tk.Label(master=frm_new_window2, text='Predicted column '+str(user_entry)+' Scaled Data: '+scalar_choice,
                            font=('Arial', 14), fg=main_colors[0], bg=main_colors[1])
    infomation_lbl.grid(row=1,column=0)
    
    best_variables_lbl=tk.Label(master=frm_new_window2, text='Best method for MLP: '+str(MLP_best_method),
                    font=('Arial', 14),fg=main_colors[0], bg=main_colors[1])
    best_variables_lbl.grid(row=2,column=0)
    
    train_acc_lbl=tk.Label(master=frm_new_window2, text='Training accuracy: '+ str(round(MLP_train_accuracy,4)),
                    font=('Arial', 14),fg=main_colors[0], bg=main_colors[1])
    train_acc_lbl.grid(row=3,column=0)
    
    test_acc_lbl=tk.Label(master=frm_new_window2, text='Testing accuracy: '+str(round(MLP_test_accuracy,4)),
                    font=('Arial', 14),fg=main_colors[0], bg=main_colors[1])
    test_acc_lbl.grid(row=4,column=0)
    
    #find the accuracy of each outcome, use the values here to satore variables about the classifier and siaply accordingly
    global outcome_lists_mlp
    outcome_accuracy=np.round(percentage_outcomes_predicted(mlp_pred, y_test),2)
    #print(outcome_accuracy)
    outcome_lists_mlp=[['Outcome','Predicted Accuracy']]
    for i in range(len(outcome_accuracy)):
        outcome_lists_mlp.append(list(outcome_accuracy[i]))
    #set up grid dimensions so that the grid can be drawn
    outcome_rows=len(outcome_lists_mlp)
    outcome_cols=len(outcome_lists_mlp[0])
    
    #decide width of each box and height of each box
    MARGIN=10
    cell_width=175
    cell_height=26
    outcome_HEIGHT=MARGIN * 2 + cell_height * outcome_rows
    outcome_WIDTH=MARGIN * 2 + cell_width * outcome_cols
    
    #predicted outcome accuracy draw on
    outcome_canvas=tk.Canvas(master=frm_new_window2, width=outcome_WIDTH, height=outcome_HEIGHT, bg=main_colors[1])
    outcome_canvas.grid(row=5, column=0)
    for i in range(outcome_rows):
        for j in range(outcome_cols):
            element=outcome_lists_mlp[i][j]
            #draw a box around each element
            x0 = MARGIN + j * cell_width + 1
            y0 = MARGIN + i * cell_height + 1
            x1 = MARGIN + (j + 1) * cell_width - 1
            y1 = MARGIN + (i + 1) * cell_height - 1
            outcome_canvas.create_rectangle(x0, y0, x1, y1, outline='black', width=1, tags='outlines')    
            #draw out the text value for what is in each part of the data frame
            x_loc=MARGIN + (j * cell_width + cell_width/2)
            y_loc=MARGIN + (i * cell_height + cell_height/2)
            outcome_canvas.create_text(x_loc, y_loc, text=element,font=("Lucida Sans Typewriter", 11), tags='outcomes', fill=main_colors[0])

#function to calculate the best method for multi-layer perception classification
def MLP_classifier_button():
    #initial classifier checks are False, once a classifier has run it will change to true. 
    #if a check is false allow the script to run otherwise do not run again.
    global MLP_check
    if MLP_check==False:
        #run the classifier
        MLP_run_classifier()
        #draw the window
        draw_MLP_window()
        #change the check value to allow computer to know this classifier has run         
        MLP_check=True
    else:
        #create a check box that allows the user to check if they want to run the entire script since it can take a while for larger data sets.
        model_check=messagebox.askyesno(title='Warning!', message='You have already ran this classifier. Click yes to run again.\nThis can change your models current accuracy.')            
        #user has chosen to re run the model.
        if model_check==True:
            MLP_window.destroy()
            #run the classifier
            MLP_run_classifier()
            #draw the window
            draw_MLP_window()
            #change the check value to allow computer to know this classifier has run         
            MLP_check=True
    
    #check that all classifiers have been run, if they have then update the window with the new decision tree classification
    if all_classifiers_check==True:
        all_classifiers_window.destroy()
        draw_all_classifiers()

#function to create the dataframe of all classifier infomation
def all_classifier_dataframe():
#create a dictionary of all of the classifiers and the coresponding accuracies
    data={"Classification Type": ["Decision Tree", "Logistic Regression", "Bernouli Naive Bayes", 'Gaussian Naive Bayes',
            "Passive Agressive", "SVC", "Random Forest", "K-Nearest Neighbour", "MLP"],
            "Best Method": [best_dtc_method, 'NaN', 'NaN', 'NaN', 'NaN', best_SVC_method, best_rfc_method, best_knn_method, MLP_best_method],
            "Best Variable": ['Depth: '+str(best_dtc_depth),'NaN', 'NaN', 'NaN', 'NaN', 'Degree: '+str(best_SVC_degree),'Estimators: '+str(best_rfc_n),
            'Neighbours: '+str(best_knn_neighbours), 'NaN'],
            "Train Accuracy": [decisiontree_train_accuracy, logisticregression_train_accuracy, 
            bernoulliNB_train_accuracy, gaussianNB_train_accuracy, passive_agressive_train_accuracy, SVC_train_accuracy,
            RFC_train_accuracy, KNN_train_accuracy, MLP_train_accuracy],
            "Test Accuracy": [decisiontree_test_accuracy, logisticregression_test_accuracy, 
            bernoulliNB_test_accuracy, gaussianNB_test_accuracy, passive_agressive_test_accuracy, SVC_test_accuracy,
            RFC_test_accuracy, KNN_test_accuracy, MLP_test_accuracy]
            }
            
    #output the data in a format easily readable        
    classifier_scores = pd.DataFrame(data)
    classifier_scores['Train Accuracy']=np.round(classifier_scores['Train Accuracy']*100, 4)
    classifier_scores['Test Accuracy']=np.round(classifier_scores['Test Accuracy']*100, 4)
    classifier_scores_df=pd.DataFrame.from_dict(classifier_scores)  #convert to dataframe for easier editing and
    global classifier_scores_sorted
    classifier_scores_sorted=classifier_scores_df.sort_values(by=['Test Accuracy'], ascending=False)  #sort in order best to worst

#function to draw all classifier tables in a pop up window
def draw_all_classifiers():
    #update the dataframe before redrawing
    all_classifier_dataframe()
    #get the columns of the data frame as a list
    cols1=[]
    for col in classifier_scores_sorted:
        cols1.append(str(col))
    #now get each row of the data frame as a list.
    row_list=classifier_scores_sorted.values.tolist()
    accuracy_grid=[cols1]
    for row in row_list:
        accuracy_grid.append(row)
    #get the rows and columns of the grid as to automate the drawing of it
    accuracy_rows=len(accuracy_grid)
    accuracy_cols=len(accuracy_grid[0])
    
    #decide width of each box and height of each box
    MARGIN=10
    cell_width=200
    cell_height=30
    WIN_HEIGHT2=MARGIN * 2 + cell_height * accuracy_rows
    WIN_WIDTH2=MARGIN * 2 + cell_width * accuracy_cols
    
    global all_classifiers_window
    #create a new window that displays all of the infomation about the classifier and the best out comes
    all_classifiers_window=tk.Toplevel(window)
    x = window.winfo_x()
    y = window.winfo_y()
    all_classifiers_window.geometry("+%d+%d" % (x+ 1.63*windowWidth,  y-50))
    all_classifiers_window.title('All classifiers training and testing split')
    all_classifiers_window.resizable(0,0)  #(x,y)
    #frame to hold all of the labels of infomation
    
    #frame for classifiers
    classifiers_frm=tk.Frame(master=all_classifiers_window, bg = main_colors[1])
    classifiers_frm.grid(row=0, column=0)
    
    #title for classifiers output
    acc_lbl=tk.Label(master=classifiers_frm, text='Each classifiers accuracy scores (%)',font=('Arial', 18), fg=main_colors[0], bg=main_colors[1])
    acc_lbl.grid(row=0, column=0)
    
    #return and show the variables the classifiers have been run on
    infomation_lbl=tk.Label(master=classifiers_frm, text='Predicted column '+str(user_entry)+', Scaled Data: '+scalar_choice+', Test split: '+str(testing_split*100)+'%',
                            font=('Arial', 14), fg=main_colors[0], bg=main_colors[1])
    infomation_lbl.grid(row=1,column=0)
    
    #df head drawn onto a canvas
    acc_canvas=tk.Canvas(master=classifiers_frm, width=WIN_WIDTH2, height=WIN_HEIGHT2, bg=main_colors[1])
    acc_canvas.grid(row=2, column=0)
    for i in range(accuracy_rows):
        for j in range(accuracy_cols):
            element=accuracy_grid[i][j]
            #draw a box around each element
            x0 = MARGIN + j * cell_width + 1
            y0 = MARGIN + i * cell_height + 1
            x1 = MARGIN + (j + 1) * cell_width - 1
            y1 = MARGIN + (i + 1) * cell_height - 1
            acc_canvas.create_rectangle(x0, y0, x1, y1, outline='black', width=1, tags='outlines')    
            #draw out the text value for what is in each part of the data frame
            x_loc=MARGIN + (j * cell_width + cell_width/2)
            y_loc=MARGIN + (i * cell_height + cell_height/2)
            acc_canvas.create_text(x_loc, y_loc, text=element,font=("Lucida Sans Typewriter", 11), tags='accuracy', fill=main_colors[0])


#function that will call every other function and display multiple windows as well as a table of all classifiers accuracies
def all_classifiers():
    import pandas as pd
    #initial classifier checks are False, once a classifier has run it will change to true. 
    #if a check is false allow the script to run otherwise do not run again.
    global decision_tree_check, log_regression_check, bernoulliNB_check, gaussianNB_check, passive_aggressive_check
    global SVC_check, random_forest_check, KNN_check, MLP_check
    global all_classifiers_check
    #create a check box that allows the user to check if they want to run the entire script since it can take a while for larger data sets.
    check=messagebox.askyesno(title='Warning!', message='This may take a while to compute. Do you want to continue?')
    if check==True:
        #run each classifier to get the accuracy values and do checks for each classifier as you run through
        #DTC  CHECK
        if decision_tree_check==False:
            #run the classifier
            dtc_run_classifier()
            #draw the window
            draw_dtc_window()
            #change the check value to allow computer to know this classifier has run         
            decision_tree_check=True
        else:
            #dont run the DTC script again if it has already been run just close the window and re open it
            decision_tree_window.destroy
            #draw the window
            draw_dtc_window()

        
        #LOG REGGRESSION CHECK
        if log_regression_check==False:
            #run the classifier
            log_reg_run_classifier()
            #draw the window
            draw_log_reg_window()
            #change the check value to allow computer to know this classifier has run         
            log_regression_check=True
        else:
            #dont run the LOG REGGRESSION script again if it has already been run just close the window and re open it
            log_regression_window.destroy
            #draw the window
            draw_log_reg_window()
        
        
        #BERNOULLI NB CHECK
        if bernoulliNB_check==False:
            #run the classifier
            bernoulliNB_run_classifier()
            #draw the window
            draw_bernoulliNB_window()
            #change the check value to allow computer to know this classifier has run         
            bernoulliNB_check=True
        else:
            #dont run the BERNOULLI NB script again if it has already been run just close the window and re open it
            bernoulliNB_window.destroy
            #draw the window
            draw_bernoulliNB_window()
        
        
        #GAUSSIAN NB CHECK
        if gaussianNB_check==False:
            #run the classifier
            gaussianNB_run_classifier()
            #draw the window
            draw_gaussianNB_window()
            #change the check value to allow computer to know this classifier has run         
            gaussianNB_check=True
        else:
            #dont run the GAUSSIAN script again if it has already been run just close the window and re open it
            gaussianNB_window.destroy
            #draw the window
            draw_gaussianNB_window()
        
        
        #PASSIVE AGGRESSIVE CHECK
        if passive_aggressive_check==False:
            #run the classifier
            passive_aggressive_run_classifier()
            #draw the window
            draw_passive_aggressive_window()
            #change the check value to allow computer to know this classifier has run         
            passive_aggressive_check=True
        else:
            #dont run the PASSIVE AGGRESSIVE script again if it has already been run just close the window and re open it
            passive_aggressive_window.destroy
            #draw the window
            draw_passive_aggressive_window()
        
        
        #SVC CHECK
        if SVC_check==False:
            #run the classifier
            SVC_run_classifier()
            #draw the window
            draw_SVC_window()
            #change the check value to allow computer to know this classifier has run         
            SVC_check=True
        else:
            #dont run the SVC script again if it has already been run just close the window and re open it
            SVC_window.destroy
            #draw the window
            draw_SVC_window()
        
        
        #RFC CHECK
        if random_forest_check==False:
            #run the classifier
            rfc_run_classifier()
            #draw the window
            draw_rfc_window()
            #change the check value to allow computer to know this classifier has run         
            random_forest_check=True
        else:
            #dont run the RFC script again if it has already been run just close the window and re open it
            random_forest_window.destroy
            #draw the window
            draw_rfc_window()
        
        
        #KNN check
        if KNN_check==False:
            #run the classifier
            KNN_run_classifier()
            #draw the window
            draw_KNN_window()
            #change the check value to allow computer to know this classifier has run         
            KNN_check=True
        else:
            #dont run the KNN script again if it has already been run just close the window and re open it
            KNN_window.destroy
            #draw the window
            draw_KNN_window()
        
        
        #MLP CHECK
        if MLP_check==False:
            #run the classifier
            MLP_run_classifier()
            #draw the window
            draw_MLP_window()
            #change the check value to allow computer to know this classifier has run         
            MLP_check=True
        else:
            #dont run the MLP script again if it has already been run just close the window and re open it
            MLP_window.destroy
            #draw the window
            draw_MLP_window()
        
        #draw the window to hold all of the infomation of each classifier
        draw_all_classifiers()
        #change variable to true to know that the all classifiers function has been run
        all_classifiers_check=True
    else:
        messagebox.showinfo(title="Warning!", message="You have chosen not to proceed. \nYou can run each classifier individually if you wish.")
        
        
    
#-------------------ALL OF THE BELOW IS FORMATING FOR THE GUI AND ITS DISPLAYS-----------------------------
global window,windowHeight,windowWidth
# Create instance
#have to use tkx if using a bloon widget to hover over tool tips.
window = tkx.Tk()
# Gets the requested distance of the height and width from top left
#of the computer screen and where the app will open
windowWidth = window.winfo_reqwidth()
windowHeight = window.winfo_reqheight()
#print("Width",windowWidth,"Height",windowHeight)
# Gets both half the screen width/height and window width/height
from math import floor
positionRight = int(floor(window.winfo_screenwidth()/100*10))
positionDown = int(window.winfo_screenheight()/2 - 2*windowHeight-15)
#print('Right position:',positionRight)
#print('Down position:',positionDown)
# Positions the window in the center of the screen.
window.geometry("+{}+{}".format(positionRight, positionDown))
window.title("SK-Learn Classification - APP (jlf)")

#initial classifier checks are False, once a classifier has run it will change to true. 
#if a check is false allow the script to run otherwise do not run again.
global decision_tree_check, log_regression_check, bernoulliNB_check, gaussianNB_check, passive_aggressive_check
global SVC_check, random_forest_check, KNN_check, MLP_check
global all_classifiers_check
decision_tree_check=False
log_regression_check=False
bernoulliNB_check=False
gaussianNB_check=False
passive_aggressive_check=False
SVC_check=False
random_forest_check=False
KNN_check=False
MLP_check=False
#variable that will allow the system to track if all classifiers have been run and update the table accordingly.
all_classifiers_check=False


#color variables that can change the whole program straight away
main_colors=['#8B4513','#FAFAD2']   #brown, light yellow
buttons_colors=['#000000','#AFEEEE']    #black, light blue
highlights=['#DAA520', '#0000FF']   #gold, dark blue
font_colors=['#000000', '#696969']  #black, dark gray.

#create a title when app opens
gui_title_frm=tk.Frame(master=window, bg=main_colors[1])
gui_title_frm.grid(row=0, column=0)
gui_title_lbl=tk.Label(master=gui_title_frm, text=' SkLearn Classification ',font=('Arial', 21), fg=main_colors[0], bg=highlights[0])
gui_title_lbl.grid(row=0, column=0, sticky='ew')

#frame to hold button to import and get the dataframes infomation
frm_buttons=tk.Frame(master=window, bg=main_colors[1])
frm_buttons.grid(row=1, column=0)
frm_buttons.grid_columnconfigure([0,1], weight=1)

#button to get the data frame infomation and display it all in another window
get_df_btn=tk.Button(master=frm_buttons,text='Import Data', width=20, relief=tk.RAISED, borderwidth=2,
                    fg=buttons_colors[0], bg=buttons_colors[1], command=get_df)
get_df_btn.grid(row=0, column=0, columnspan=2, padx=5,pady=5)
#Create a tooltip for get df button
df_btn_tip = tkx.Balloon(window, initwait=200)
#Bind the tooltip with button
df_btn_tip.bind_widget(get_df_btn, balloonmsg="Import the data as a dataframe that will be \ndisplayed in another window with infomation \nabout the data.")

#set up an options box to let the user tick if the data needs scaling or not
scalar_options = ["NO", "YES",]
#initiate the different choices the option box can take
variable = tk.StringVar(frm_buttons)
variable.set(scalar_options[0]) # default value
scale_data_optionbox = tk.OptionMenu(frm_buttons, variable, *scalar_options)
scale_data_optionbox.config(width=14)
scale_data_optionbox.grid(row=1, column=0)
# add a button that allows the user to pick if the data needs scaling or not
scale_data_btn = tk.Button(frm_buttons, text="Scale the data",width=20, relief=tk.RAISED, borderwidth=2,
                    fg=buttons_colors[0], bg=buttons_colors[1], command=set_scalar)
scale_data_btn.grid(row=1, column=1)
#Create a tooltip for get scale data button
scale_data_btn_tip = tkx.Balloon(window, initwait=200)
#Bind the tooltip with button
scale_data_btn_tip.bind_widget(scale_data_btn, balloonmsg="This will rescale all of the data values except the outcome\n column chosen below to values between 0 and 1.")

#a entry box that allows the user to enter a value between 0(0%) and 1(100%) determining the training data proportion
set_y_ent=tk.Entry(master=frm_buttons)
set_y_ent.insert(tk.END, "8")
set_y_ent.grid(row=2, column=0, padx=5,pady=5)
set_y_ent.bind("<Return>", enter_pressed_set_outcome_column)        #bind ENTER to the entry that will run the same as button pressed

#button to take the data frame and split it into an input matrix (X) and the predicted column(y)
set_y_btn=tk.Button(master=frm_buttons,text='Set predicted column', width=20, relief=tk.RAISED, borderwidth=2,
                    fg=buttons_colors[0], bg=buttons_colors[1], command=set_outcome_column)
set_y_btn.grid(row=2, column=1, padx=5,pady=5)
#Create a tooltip for get set_y column function button
set_y_btn_tip = tkx.Balloon(window, initwait=200)
#Bind the tooltip with button
set_y_btn_tip.bind_widget(set_y_btn, balloonmsg="Input an index starting at 0, which will be \nthe column to be predicted by the program.")

#a entry box that allows the user to enter a value between 0(0%) and 1(100%) determining the training data proportion
test_train_ent=tk.Entry(master=frm_buttons)
test_train_ent.insert(tk.END, "0.2")
test_train_ent.grid(row=3, column=0, padx=5,pady=5)
test_train_ent.bind("<Return>", enter_pressed_set_test_train_split)        #bind ENTER to the entry that will run the same as button pressed

#button to take the data frame and split it into a test and train data set ready for analysis.
set_test_train_btn=tk.Button(master=frm_buttons,text='Set test split', width=20, relief=tk.RAISED, borderwidth=2,
                    fg=buttons_colors[0], bg=buttons_colors[1], command=set_test_train_split)
set_test_train_btn.grid(row=3, column=1, padx=5,pady=5)
#Create a tooltip for the test train split button
test_train_btn_tip = tkx.Balloon(window, initwait=200)
#Bind the tooltip with button
test_train_btn_tip.bind_widget(set_test_train_btn, balloonmsg="Input the proportion of the data the user wants \nto have as the training set. \nThis should be a number between 0 and 1.")

#create a label that allows the user to see which classifier they are using
decision_tree_label=tk.Label(master=frm_buttons, text='Decision Tree Classifier',fg=main_colors[0], bg=main_colors[1])
decision_tree_label.grid(row=4, column=0, sticky='ew', padx=5,pady=5)

#button to take the data frame and split it into a test and train data set ready for analysis.
decision_tree_btn=tk.Button(master=frm_buttons,text='Find Model', width=20, relief=tk.RAISED, borderwidth=2,
                    fg=buttons_colors[0], bg=buttons_colors[1], command=decision_tree_classification_button)
decision_tree_btn.grid(row=4, column=1, padx=5,pady=5)
#Create a tooltip for classification model infomation
decision_tree_btn_tip = tkx.Balloon(window, initwait=200)
#Bind the tooltip with button
decision_tree_btn_tip.bind_widget(decision_tree_btn, balloonmsg="Use a decision tree classifier that will calculate the \nbest parameters for the best accuracy. \nThis will also return what the best parameters are \nand the given testing and training accuracy scores.")

#create a label that allows the user to see which classifier they are using
log_regression_label=tk.Label(master=frm_buttons, text='Log Regression Classifier',fg=main_colors[0], bg=main_colors[1])
log_regression_label.grid(row=5, column=0, sticky='ew', padx=5,pady=5)

#button to take the data frame and split it into a test and train data set ready for analysis.
log_regression_btn=tk.Button(master=frm_buttons,text='Find Model', width=20, relief=tk.RAISED, borderwidth=2,
                    fg=buttons_colors[0], bg=buttons_colors[1], command=log_regression_classification_button)
log_regression_btn.grid(row=5, column=1, padx=5,pady=5)
#Create a tooltip for classification model infomation
log_regression_btn_tip = tkx.Balloon(window, initwait=200)
#Bind the tooltip with button
log_regression_btn_tip.bind_widget(log_regression_btn, balloonmsg="Use a logistic regression classifier and the give the \ntesting and training accuracy scores.")

#create a label that allows the user to see which classifier they are using
bernoulliNB_label=tk.Label(master=frm_buttons, text='Bernoulli N-Bayes Classifier',fg=main_colors[0], bg=main_colors[1])
bernoulliNB_label.grid(row=6, column=0, sticky='ew', padx=5,pady=5)

#button to take the data frame and split it into a test and train data set ready for analysis.
bernoulliNB_btn=tk.Button(master=frm_buttons,text='Find Model', width=20, relief=tk.RAISED, borderwidth=2,
                    fg=buttons_colors[0], bg=buttons_colors[1], command=bernoulliNB_classification_button)
bernoulliNB_btn.grid(row=6, column=1, padx=5,pady=5)
#Create a tooltip for classification model infomation
bernoulliNB_btn_tip = tkx.Balloon(window, initwait=200)
#Bind the tooltip with button
bernoulliNB_btn_tip.bind_widget(bernoulliNB_btn, balloonmsg="Use a Bernoulli N-Bayes classifier and the give the \ntesting and training accuracy scores.")

#create a label that allows the user to see which classifier they are using
GaussianNB_label=tk.Label(master=frm_buttons, text='Gaussian N-Bayes Classifier',fg=main_colors[0], bg=main_colors[1])
GaussianNB_label.grid(row=7, column=0, sticky='ew', padx=5,pady=5)

#button to take the data frame and split it into a test and train data set ready for analysis.
GaussianNB_btn=tk.Button(master=frm_buttons,text='Find Model', width=20, relief=tk.RAISED, borderwidth=2,
                    fg=buttons_colors[0], bg=buttons_colors[1], command=gaussianNB_classification_button)
GaussianNB_btn.grid(row=7, column=1, padx=5,pady=5)
#Create a tooltip for classification model infomation
GaussianNB_btn_tip = tkx.Balloon(window, initwait=200)
#Bind the tooltip with button
GaussianNB_btn_tip.bind_widget(GaussianNB_btn, balloonmsg="Use a Gaussian N-Bayes classifier and the give the \ntesting and training accuracy scores.")

#create a label that allows the user to see which classifier they are using
passive_aggressive_label=tk.Label(master=frm_buttons, text='Passive Aggressive Classifier',fg=main_colors[0], bg=main_colors[1])
passive_aggressive_label.grid(row=8, column=0, sticky='ew', padx=5,pady=5)

#button to take the data frame and split it into a test and train data set ready for analysis.
passive_aggressive_btn=tk.Button(master=frm_buttons,text='Find Model', width=20, relief=tk.RAISED, borderwidth=2,
                    fg=buttons_colors[0], bg=buttons_colors[1], command=passive_aggressive_classification_button)
passive_aggressive_btn.grid(row=8, column=1, padx=5,pady=5)
#Create a tooltip for classification model infomation
passive_aggressive_btn_tip = tkx.Balloon(window, initwait=200)
#Bind the tooltip with button
passive_aggressive_btn_tip.bind_widget(passive_aggressive_btn, balloonmsg="Use a Passive Agressive classifier and the give the \ntesting and training accuracy scores.")

#create a label that allows the user to see which classifier they are using
SVC_label=tk.Label(master=frm_buttons, text='SVC Classifier',fg=main_colors[0], bg=main_colors[1])
SVC_label.grid(row=9, column=0, sticky='ew', padx=5,pady=5)

#button to take the data frame and split it into a test and train data set ready for analysis.
SVC_btn=tk.Button(master=frm_buttons,text='Find Model', width=20, relief=tk.RAISED, borderwidth=2,
                    fg=buttons_colors[0], bg=buttons_colors[1], command=SVC_classifier_button)
SVC_btn.grid(row=9, column=1, padx=5,pady=5)
#Create a tooltip for classification model infomation
SVC_btn_tip = tkx.Balloon(window, initwait=200)
#Bind the tooltip with button
SVC_btn_tip.bind_widget(SVC_btn, balloonmsg="Use an SVC classifier and  find the optimal method as well\n as the give the testing and training accuracy scores.")

#create a label that allows the user to see which classifier they are using
RFC_label=tk.Label(master=frm_buttons, text='RFC Classifier',fg=main_colors[0], bg=main_colors[1])
RFC_label.grid(row=10, column=0, sticky='ew', padx=5,pady=5)

#button to take the data frame and split it into a test and train data set ready for analysis.
RFC_btn=tk.Button(master=frm_buttons,text='Find Model', width=20, relief=tk.RAISED, borderwidth=2,
                    fg=buttons_colors[0], bg=buttons_colors[1], command=random_forest_classifier_button)
RFC_btn.grid(row=10, column=1, padx=5,pady=5)
#Create a tooltip for classification model infomation
RFC_btn_tip = tkx.Balloon(window, initwait=200)
#Bind the tooltip with button
RFC_btn_tip.bind_widget(RFC_btn, balloonmsg="Use an RFC classifier and  find the optimal method as well\n as the give the testing and training accuracy scores. \nThis classifier can take a while to run")

#create a label that allows the user to see which classifier they are using
KNN_label=tk.Label(master=frm_buttons, text='KNN Classifier',fg=main_colors[0], bg=main_colors[1])
KNN_label.grid(row=11, column=0, sticky='ew', padx=5,pady=5)

#button to take the data frame and split it into a test and train data set ready for analysis.
KNN_btn=tk.Button(master=frm_buttons,text='Find Model', width=20, relief=tk.RAISED, borderwidth=2,
                    fg=buttons_colors[0], bg=buttons_colors[1], command=KNN_classifier_button)
KNN_btn.grid(row=11, column=1, padx=5,pady=5)
#Create a tooltip for classification model infomation
KNN_btn_tip = tkx.Balloon(window, initwait=200)
#Bind the tooltip with button
KNN_btn_tip.bind_widget(KNN_btn, balloonmsg="Use an KNN classifier and  find the optimal method as well\n as the give the testing and training accuracy scores.")

#create a label that allows the user to see which classifier they are using
MLP_label=tk.Label(master=frm_buttons, text='MLP Classifier',fg=main_colors[0], bg=main_colors[1])
MLP_label.grid(row=12, column=0, sticky='ew', padx=5,pady=5)

#button to take the data frame and split it into a test and train data set ready for analysis.
MLP_btn=tk.Button(master=frm_buttons,text='Find Model', width=20, relief=tk.RAISED, borderwidth=2,
                    fg=buttons_colors[0], bg=buttons_colors[1], command=MLP_classifier_button)
MLP_btn.grid(row=12, column=1, padx=5,pady=5)
#Create a tooltip for classification model infomation
MLP_btn_tip = tkx.Balloon(window, initwait=200)
#Bind the tooltip with button
MLP_btn_tip.bind_widget(MLP_btn, balloonmsg="Use an MLP classifier and  find the optimal method as well\n as the give the testing and training accuracy scores.")

#button to get the data frame infomation and display it all in another window
all_classifiers_btn=tk.Button(master=frm_buttons,text='Compare every classifier', width=20, relief=tk.RAISED, borderwidth=2,
                    fg=buttons_colors[0], bg=buttons_colors[1], command=all_classifiers)
all_classifiers_btn.grid(row=13, column=0, columnspan=2, padx=5,pady=5)
#Create a tooltip for get df button
all_classifiers_tip = tkx.Balloon(window, initwait=200)
#Bind the tooltip with button
all_classifiers_tip.bind_widget(all_classifiers_btn, balloonmsg="This will run every classifier and output all the infomation\n independently as well as a table to compare all\n classifiers and thier accuracy.")

# Run the application
window.mainloop()