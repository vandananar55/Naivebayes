

class Description:
    def category_columns(self,new_data,column_name):
        print("categorical function running ==>")
        self.column_name =  column_name

        count = new_data[column_name].value_counts()
        print("counts in {} are \n {}".format(column_name,count))


        Mode = new_data[column_name].mode()
        print("Mode value of {} is \n {}".format(column_name,Mode))

        new_data[column_name].fillna(new_data[column_name].mode()[0], inplace=True)
        print( "After filling with mode {} is......" .format(column_name))
        print(new_data[column_name])

        NULL = new_data[column_name].isnull().sum()
        print ("Remaining null values are {}".format(NULL))
        print("again value counts of {} are".format(column_name))
        print(new_data[column_name].value_counts())
        print("categorical function ended ==>")


    def numerical_columns(self,new_data,n_column_name):  
        self.n_column_name =  n_column_name  

        Mean = new_data[n_column_name].mean()
        print("Mean value of {} is \n {}".format(n_column_name,Mean))

        new_data[n_column_name].fillna(new_data[n_column_name].mean(), inplace=True)
        print( "After filling with mean {} is......" .format(n_column_name))
        print(new_data[n_column_name].head(15))

        NULL = new_data[n_column_name].isnull().sum()
        print ("Remaining null values are {}".format(NULL))

