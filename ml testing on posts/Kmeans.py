import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
import sys
from sklearn.decomposition import PCA

#main 
def main(in_data):
    
    #reading in data
    data = pd.read_csv(in_data)

    #get columns that are numeric and important for clustering
    important_features = data[['sentiment_mean', 'most_common_hour', 'avg_score_Conservative', 'avg_score_Liberal', 'avg_score_Republican', 'avg_score_democrats', 'avg_score_politics']]

    #pipeline for scaling and KMeans
    kmeans_model = make_pipeline(
        StandardScaler(),
        KMeans(n_clusters=4)
    )

    #fitting important_features
    kmeans_model.fit(important_features)

    #add cluster label to the orginal data
    important_features['cluster'] = kmeans_model.named_steps['kmeans'].labels_


    #PCA
    pca = PCA(n_components = 2)
    important_features_nc = important_features.drop(columns=['cluster'])
    important_features_pca = pca.fit_transform(important_features_nc)

    #plot
    plt.scatter(important_features_pca[:,0], important_features_pca[:,1], c =
                important_features['cluster'], cmap = 'viridis')

    #make plot look nice
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.title('KMeans Clusters')
    plt.savefig('KMeans4.png')


    #for analysis of the PCA components
    pca_df = pd.DataFrame(pca.components_, columns =
                                  important_features_nc.columns)
    pca1_most_important = pca_df.iloc[0].sort_values(ascending = False)
    pca2_most_important = pca_df.iloc[1].sort_values(ascending = False)

    #printing most important features
    print("Most important features for PCA 1:")
    print(pca1_most_important)
    print("Most important features for PCA 2:")
    print(pca2_most_important)

    #for analysis of the means for each cluster
    print(important_features.groupby('cluster').mean())
    
    
    

if __name__ == '__main__':
    in_data = sys.argv[1]
    main(in_data)