from sklearn.cluster import KMeans
import pandas as pd
import numpy as np 

class AnomalyDetection():
    
    def get_all_features_set(self, df, indices):
        
        ## Define a hashet 'feature_set' to hold all the unique values 
        feature_set = set()
        
        ## Iterate through list
        for each_features in df['features']:
            
            ## Make features only from specified indices
            each_set = [each_features[i] for i in indices]
            for feature in each_set:
                feature_set.add(feature)
        return feature_set
    
    
    
    def get_each_feature_set(self, df, feature_set, indices):
        rows = []

        ## Define set of indices that are categorical and need to be encoded and the rest
        all_indices = set(range(len(df['features'][0])))
        req_indices = set(indices)
        remaining_indices = list(all_indices - req_indices)
        
        ## Iterate through feature lists
        for each_features in df['features']:
            each_set = [each_features[i] for i in indices]
            other_values = [each_features[i] for i in remaining_indices]
            row = dict.fromkeys(list(feature_set), 0)
            
            ## If feature in feature_set then append 1 else 0 
            for feature in each_set:
                row[feature] = 1 + row.get(feature, 0)
            list_row = list(row.values())
            for other_value in other_values:
                list_row.append(other_value)
            rows.append(list_row)
        return rows
         
        
    def cat2Num(self, df, indices):
        
        ## Get feature_set
        feature_set = self.get_all_features_set(df, indices)
        
        ## Get list of all encoded rows
        all_features_encoded = self.get_each_feature_set(df, feature_set, indices)
        
        ## Set Value of each row to encoded features
        for i, row in df.iterrows():
            df.at[i, 'features'] = all_features_encoded[i]
        return df
            
                

    def scaleNum(self, df, indices):
        
        values = []
        for each_features in df['features']:
            
            ## Create list of values at specified indices
            values_set = [each_features[i] for i in indices]
            for value in values_set:
                values.append(value)
        
        ## Find mean and std deviation of the values
        values_arr = np.array(values)
        mean = np.mean(values_arr, axis=0)
        std_dev = np.std(values_arr, ddof=1, axis=0)
        
        ## Calulate List of Standardized Values 
        std_values=[]
        for i in range(len(values_arr)):
            value = (values_arr[i] - mean)/std_dev
            std_values.append(value)
        
        ## Create full row including encoded values and standardized values
        i=0
        all_values = []
        for each_features in df['features']:
            each_set = [each_features[i] for i in indices]
            other_values = [each_features[i] for i in range(len(each_features)) if i not in indices]
            other_values.append(std_values[i])
            all_values.append(other_values)
            i+=1
        
        for i, row in df.iterrows():
            df.at[i, 'features'] = all_values[i]
        return df


    def detect(self, df, k, t):
        
        ## Use KMeans Clustering to assign clusters to all the data points
        df['clusterNo'] = KMeans(n_clusters=k, random_state=0).fit_predict(list(df['features']))
        
        ## Group all the datapoints in each cluster and count to get size of cluster
        clusters = df.groupby('clusterNo').count().reset_index().rename(columns = {'features': 'clusterSize'})
        
        ## Calculate max and min cluster sizes as Nmax & Nmin
        Nmax = clusters['clusterSize'].max()
        Nmin = clusters['clusterSize'].min()
        
        ## Merge on cluster number to get the size of the cluster each data point belongs to
        df = df.merge(clusters, on = 'clusterNo').rename(columns = {'clusterSize': 'Nx'})
        
        ## Calculate score
        df['score'] = (Nmax-df['Nx'])/(Nmax-Nmin)
        
        ## Check if score is above threshold t 
        df = df[df['score']>t].reset_index().rename(columns = {'index': 'id'})
        
        ## Return req_df with selected columns only
        req_df = df[['id', 'features', 'score']]
        return req_df
        
if __name__ == "__main__":

    df = pd.read_csv('logs-features-sample.csv', converters={'features': eval}).set_index('id')
    ad = AnomalyDetection()
    df1 = ad.cat2Num(df, [0,1])
    print(df1)

    df2 = ad.scaleNum(df1, [6])
    print(df2)

    df3 = ad.detect(df2, 8, 0.97)
    print(df3)