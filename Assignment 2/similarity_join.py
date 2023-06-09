# -*- coding: utf-8 -*-
"""similarity_join.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1f1yQ1-LcdayiKV_HxzfNmdTn7IQghPIT
"""

import re
import pandas as pd

from collections import defaultdict
import numpy as np

class SimilarityJoin:
    def __init__(self, data_file1, data_file2):
        self.df1 = pd.read_csv(data_file1)
        self.df2 = pd.read_csv(data_file2)

    def preprocess_df(self, df, cols): 
      ## Concatenate the columns while filling NaN values with empty string to avoid error
      df["joinKey"] = df[cols[0]].fillna('') + " " + df[cols[1]].fillna('')

      ## Convert To Lowercase 
      df["joinKey"]= df['joinKey'].str.lower()

      ## To Strip Out Added Spaces Before Splitting Create A Temporary List and Append Stripped Values
      temp = []
      for i in df['joinKey']:
        temp.append(i.strip())
      df['joinKey'] = temp
      
      ## Split into a list of tokens eg:  0 [iview, mediapro, 2, 5, global, marketing]  
      df["joinKey"] = df["joinKey"].str.split(r'\W+')
      return df

    def filtering(self, df1, df2):

      ## Initialize cand_df dataframe  
      cand_df = pd.DataFrame()
      
      ## Create another column duplicate for joinKey to explode next
      df1['ExplodedKeys'] = df1['joinKey']
      df2['ExplodedKeys'] = df2['joinKey']
      
      ## Explode the joinKey for Each & Drop Duplicate Columns in Each
      df1 = df1.explode('ExplodedKeys')
      df1 = df1.drop(columns = ['title', 'description', 'manufacturer', 'price'])
      df2 = df2.explode('ExplodedKeys')
      df2 = df2.drop(columns = ['name', 'description', 'manufacturer', 'price'])
      
      ## Perform merge operation on Exploded Keys and Drop
      df_merged = df1.merge(df2, on='ExplodedKeys').drop(columns = ['ExplodedKeys'])
      
      ## Rename Columns & Drop Duplictes of id1 and id2
      cand_df = df_merged.rename(columns={'id_x': 'id1', 'joinKey_x': 'joinKey1', 'id_y': 'id2', 'joinKey_y': 'joinKey2'})
      cand_df = cand_df.drop_duplicates(subset = ['id1', 'id2'], keep = 'first')

      return cand_df


    def verification(self, cand_df, threshold):

        ## Create a Temp List to hold All Jaccard Values
        jaccardAllValues = []

        ## Iterate To Get Set Of joinKey1 & joinKey2 and Get Their Counts
        for index, row in cand_df.iterrows():
            setOne = set(row['joinKey1'])
            setTwo = set(row['joinKey2'])
            countSetOne = len(setOne)
            countSetTwo = len(setTwo)

        ## Find Intersection & Get Its Count
            afterIntersect = setOne.intersection(setTwo)
            countIntersect = len(afterIntersect)

        ## Calculate Jaccard for each & append to list
            jaccard = countIntersect / ((countSetOne + countSetTwo) - countIntersect)
            jaccardAllValues.append(jaccard)

        ## Append All Jaccard Values
        cand_df['jaccard'] = jaccardAllValues

        ## Filter out the rows with jaccard greater than or equal to threshold
        results_df = cand_df[cand_df['jaccard'] >= threshold]
        return results_df



    def evaluate(self, result, ground_truth):
        
        ## Create List to Hold Truly Matching Values
        trulyMatching = []

        # R is the count of result
        R = len(result)

        # T can be calculated as the values in result that exist in ground_truth
        for i in result:
            if i in ground_truth: 
              trulyMatching.append(i)
        T = len(trulyMatching)

        ## A is the count of ground_truths
        A = len(ground_truth)

        ## Calculate & Return Precision, Recall and F1-Scores
        precision = T / R
        recall = T / A
        fmeasure = (2 * precision * recall) / (precision + recall)
        return (precision, recall, fmeasure)


    def jaccard_join(self, cols1, cols2, threshold):
        new_df1 = self.preprocess_df(self.df1, cols1)
        new_df2 = self.preprocess_df(self.df2, cols2)

        print ("Before filtering: %d pairs in total" %(self.df1.shape[0] *self.df2.shape[0])) 

        cand_df = self.filtering(new_df1, new_df2)
        print ("After Filtering: %d pairs left" %(cand_df.shape[0]))

        result_df = self.verification(cand_df, threshold)
        print ("After Verification: %d similar pairs" %(result_df.shape[0]))

        return result_df



if __name__ == "__main__":
    er = SimilarityJoin("Amazon_sample.csv", "Google_sample.csv")
    amazon_cols = ["title", "manufacturer"]
    google_cols = ["name", "manufacturer"]
    result_df = er.jaccard_join(amazon_cols, google_cols, 0.5)

    result = result_df[['id1', 'id2']].values.tolist()
    ground_truth = pd.read_csv("Amazon_Google_perfectMapping_sample.csv").values.tolist()
    print ("(precision, recall, fmeasure) = ", er.evaluate(result, ground_truth))