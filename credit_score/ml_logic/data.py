import pandas as pd
import numpy as np
import scipy.stats as stats

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    print("🧹 Cleaning data ...")
    # Remove special characters
    cleaned_df = df.applymap(__remove_special_characters).replace(['', 'nan', '!@9#%8', '#F%$D@*&8'], np.NaN)

    # Change relevant data types
    cleaned_df = __change_data_type(cleaned_df)
    print('🔧 Data types changed')
    cleaned_df['Credit_History_Age'] = cleaned_df.Credit_History_Age.apply(lambda x: __convert_to_months(x)).astype(float)
    print('🔧 Credit_History_Age converted to months')

    # Reassign missing object values with mode grouping by 'Customer_ID'
    __reassign_object_missing_with_mode(cleaned_df, 'Customer_ID', 'Credit_Mix')
    print('🔧 Credit_Mix missing values re-assigned')

    # Reassign missing numerical values with mode grouping by 'Customer_ID'
    numerical_col = ['Age', 'Annual_Income', 'Monthly_Inhand_Salary', 'Num_Bank_Accounts', 'Num_Credit_Card',
                     'Interest_Rate', 'Num_of_Loan', 'Delay_from_due_date', 'Num_of_Delayed_Payment', 'Changed_Credit_Limit',
                     'Num_Credit_Inquiries', 'Outstanding_Debt', 'Monthly_Balance']

    if df.shape[0] > 1:
        print('🔧 Interpolating Credit_History_Age')
        cleaned_df['Credit_History_Age'] = cleaned_df.groupby('Customer_ID')['Credit_History_Age'].apply(
            lambda x: x.interpolate().bfill().ffill())

    for col in numerical_col:
        print('🔧 Reassigning missing values for', col)
        __reassign_numeric_missing_with_mode(cleaned_df, 'Customer_ID', col)

    # Compress DataFrame
    print('🔧 Compressing DataFrame')
    cleaned_df = __compress(cleaned_df)

    print("✅ Data cleaned")
    print(cleaned_df.shape)
    return cleaned_df


def __compress(df, **kwargs):
    """
    Reduces the size of the DataFrame by downcasting numerical columns
    """
    for type in ["float", "integer"]:
        l_cols = list(df.select_dtypes(include=type))
        for col in l_cols:
            df[col] = pd.to_numeric(df[col], downcast=type)

    return df

def __remove_special_characters(data):
    """
    Removes special characters from a string
    """
    if data is np.NaN or not isinstance(data, str):
        return data
    else:
        return str(data).strip('_ ,"')

def __change_data_type(df):
    """
    Changes the data type of the columns
    """
    df['Age'] = df.Age.astype(int)
    df['Annual_Income'] = df.Annual_Income.astype(float)
    df['Num_of_Loan'] = df.Num_of_Loan.astype(int)
    df['Num_of_Delayed_Payment'] = df.Num_of_Delayed_Payment.astype(float)
    df['Changed_Credit_Limit'] = df.Changed_Credit_Limit.astype(float)
    df['Outstanding_Debt'] = df.Outstanding_Debt.astype(float)
    df['Monthly_Balance'] = df.Monthly_Balance.astype(float)

    return df

def __convert_to_months(x):
    """
    Convert years and months to months
    """
    if pd.notnull(x):
        num1 = int(x.split(' ')[0])
        num2 = int(x.split(' ')[3])
        return (num1 * 12) + num2
    else:
        return x

def __reassign_object_missing_with_mode(df, groupby, column, inplace=True):
    """
    Reassign missing object values with mode grouping by 'Customer_ID'
    """

    # Assigning Wrong values Make Simple Function
    def make_NaN_and_fill_mode(df, groupby, column, inplace=True):
        # Assign None to np.NaN
        if df[column].isin([None]).sum():
            df[column][df[column].isin([None])] = np.NaN

        # fill with local mode
        result = df.groupby(groupby)[column].transform(lambda x: x.fillna(stats.mode(x)[0][0]))

        if inplace:
            df[column]=result
        else:
            return result

    if inplace:
        make_NaN_and_fill_mode(df, groupby, column, inplace)
    else:
        return make_NaN_and_fill_mode(df, groupby, column, inplace)

def __reassign_numeric_missing_with_mode(df, groupby, column, inplace=True):
    """
    Reassign missing numerical values with mode grouping by 'Customer_ID'
    """

    # Assigning Wrong values
    def make_group_NaN_and_fill_mode(df, groupby, column, inplace=True):
        df_dropped = df[df[column].notna()].groupby(groupby)[column].apply(list)
        x, y = df_dropped.apply(lambda x: stats.mode(x)).apply([min, max])
        mini, maxi = x[0][0], y[0][0]

        # assign Wrong Values to NaN
        col = df[column].apply(lambda x: np.NaN if ((x<mini)|(x>maxi)) else x)

        # fill with local mode
        mode_by_group = df.groupby(groupby)[column].transform(lambda x: x.mode()[0] if not x.mode().empty else np.NaN)
        result = col.fillna(mode_by_group)

        if inplace:
            df[column]=result
        else:
            return result

    if inplace:
        make_group_NaN_and_fill_mode(df, groupby, column, inplace)
    else:
        return make_group_NaN_and_fill_mode(df, groupby, column, inplace)
