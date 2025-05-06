import matplotlib
matplotlib.use("TkAgg")
import pandas as pd
import tkinter as tk
from tkinter import filedialog
import math

def excel():
    # Popup for chosing excel file
    root = tk.Tk()
    root.withdraw()
    filepath = filedialog.askopenfilename(title='choose timing excel')
    root.destroy()

    df = pd.read_excel(filepath, header=None)

    # Adjust df labels to index by shot number and "mcpx frame x"
    df.columns = df.iloc[0]
    df = df[1:].reset_index(drop=True)
    df.set_index(df.columns[0], inplace=True)
    df.columns = df.columns.astype(int)

    return df


def shotsheet():
    root = tk.Tk()
    root.withdraw()

    # Pop up for chosing the shot sheet
    filepath = filedialog.askopenfilename(title='choose shotsheet')

    root.destroy()

    df  = pd.read_excel(filepath,
                    skiprows=0,)

    ### REORGANIZE DATAFRAME ###
    # find all the unnamed columns and columns with wrong names
    # and assign them a non-zero value
    i=0
    ind=[]
    for col in df.columns:
        if df.columns.get_loc(col) < len(df.columns)-1:
            next = df.columns[df.columns.get_loc(col)+1]
        elif df.columns.get_loc(col) == len(df.columns)-1:
            next = 'end'
        if next.startswith('Unnamed'):
            i+=1
            ind.append(i)
        elif col.startswith('Unnamed'):
            i+=1
            ind.append(i)
            i=0
        else:
            i=0
            ind.append(i)

    # Take name from next row if the row was unnamed or wrong
    for col in df.columns:
        if ind[df.columns.get_loc(col)] != 0:
            new = df[col][0].replace('\n',' ').split('(')[0].lower().strip()
            df.rename(columns={col:new},inplace=True)

    # remove first row
    df = df.drop(0)
    df = df.reset_index()

    # remove * from any shot number and drop date rows
    for index, row in df.iterrows():
        if '*' in str(row['SHOT #']):
            row['SHOT #']=float(str(row['SHOT #']).replace('*',''))
            df.loc[index,'SHOT #']=row['SHOT #']
            df.loc[index,'post-shot notes'] = str(df.loc[index,'post-shot notes'])
            df.loc[index,'post-shot notes']+='*a B-dot is not working'
        if math.isnan(row['SHOT #']):
            df = df.drop(index)

    # Selecting relevant columns
    columns_to_include = ['lv gas', 'tv gas',
                          'mcp1 frame 1', 'mcp1 frame 2',
                          'mcp1 frame 3', 'mcp1 frame 4',
                          'mcp2 frame 1', 'mcp2 frame 2',
                          'mcp2 frame 3', 'mcp2 frame 4']

    # Ensure all necessary columns exist in DataFrame before selecting
    df_selected = df[['SHOT #'] + [col for col in columns_to_include if col in df.columns]].set_index('SHOT #')

    # Transpose the DataFrame so shot numbers are columns
    df_transposed = df_selected.T

    return df_transposed
