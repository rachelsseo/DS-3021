# %% [markdown]
# ### Decision Tree Lab 
# 
# ##### Rachel Seo ydp7xv

# %%
# imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import graphviz 

from sklearn.model_selection import train_test_split,GridSearchCV,RepeatedStratifiedKFold
from sklearn import metrics
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier, export_graphviz 

# %%
# data loading
movie_metadata=pd.read_csv("../data/movie_metadata.csv")

# %%
# total function 
def cleaning(df):
    # drop irrelevant columns
    columns = ['duration', 'actor_3_facebook_likes', 'facenumber_in_poster', 'actor_2_facebook_likes', 'aspect_ratio', 'color',
           'actor_2_name', 'actor_3_name', 'plot_keywords', 'movie_imdb_link', 'language', 'director_name', 'title_year', 'actor_1_name', 'movie_title']
    df.drop(columns=columns, inplace=True)

    def parse_genres(genres):
        if isinstance(genres, str):  # checking if genres is a string
            # split the string by '|' and return the first genre
            return genres.split('|')[0].strip()
        elif isinstance(genres, list) and len(genres) > 0:  # if it's already a list, return the first element
            return genres[0]
        else:
            return None  # return None if the value is not a string or a non-empty list
    df['genres'] = df['genres'].apply(parse_genres)

    # collapsing country 
    top_countries = ['USA', 'UK']
    df.country = (df.country.apply(lambda x: x if x in top_countries else "Other")).astype('category')

    # collapsing genres
    top_genres = ['Comedy', 'Action']
    df.genres = (df.genres.apply(lambda x: x if x in top_genres else "Other")).astype('category')

    # collapsing content ratings 
    top_ratings = ['R', 'PG-13']
    df['content_rating'] = df['content_rating'].apply(lambda x: x if x in top_ratings else 'Other').astype('category')

    # dropping null values
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df

# %%
# testing the function
df = cleaning(movie_metadata)

df.head()

# %%
# cut imdb_score into 1 and 0 from top 25% percentile 

def cut_imdb_score(df):
    # calculate the 75th percentile
    percentile_75 = df['imdb_score'].quantile(0.75)
    
    # create a new column 'top_imdb_score'
    df['top_imdb_score'] = np.where(df['imdb_score'] >= percentile_75, 1, 0)
    
    return df
# applying the cut function
df = cut_imdb_score(df)

df.drop(columns=['imdb_score'], inplace=True)
df['top_imdb_score'] = df['top_imdb_score'].astype('category')
df.info()

# %%
# collapsing/encoding target variable
df[["top_imdb_score"]] = OrdinalEncoder().fit_transform(df[["top_imdb_score"]])
print(df["top_imdb_score"].value_counts())

# %%
# prevalence for the target variable
print(1038/(2849+1038))

# The prevalence of the target variable represents the proportion of positive cases (1s) in the dataset. 
# In this case, the prevalence is approximately 26.7%, meaning that about 26.7% of the movies in the dataset have a high IMDb score (top 25%).
# This is a relatively imbalanced dataset, as there are significantly more movies with lower IMDb scores (0s) compared to those with higher scores (1s).

# %%
# one-hot encoding
categorical_columns = df.select_dtypes(include=['category']).columns
df2 = pd.get_dummies(df, columns=categorical_columns)

# %%
# splitting the dataset into test, tune, and train sets (80% train, 10% tune, 10% test)

X = df2.drop(columns=['top_imdb_score'])
y = df2.top_imdb_score

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.80, stratify= y, random_state=21)
X_tune, X_test, y_tune, y_test = train_test_split(X_test,y_test,  train_size = 0.5,stratify= y_test, random_state=49)

# %%
# kfold object for cross validation

kf = RepeatedStratifiedKFold(n_splits=10,n_repeats=5, random_state=42)

# %%
# scoring metric and the max depth hyperparameter (grid search) 
scoring = 'f1_weighted'
param = {"max_depth" : [1,2,3,4,5,6,7,8,9,10,11]}

# %%
# fitting model to training data
cl= DecisionTreeClassifier(random_state=1000)

# search for best DecisionTreeClassifier estimator across all folds based f1_weighted
search = GridSearchCV(cl, param, scoring=scoring, n_jobs=-1, cv=kf,refit='f1_weighted')

# execute on training data
model = search.fit(X_train, y_train)

# best hyperparameters and score
best = model.best_estimator_
print(best) 

# %%
print(model.cv_results_)

# %%
# extracting scores to view model

# scores: 
f1_w = model.cv_results_['mean_test_score']
SD_f1_w = model.cv_results_['std_test_score']

# parameter:
depth= np.unique(model.cv_results_['param_max_depth']).data

# building dataframe
final_model = pd.DataFrame(list(zip(depth, 
                                    f1_w, 
                                    SD_f1_w)),
                           columns=['depth', 'f1_weighted', 'f1_weightedSD'])

# final model
final_model.style.hide(axis='index')

# The model seems to be performing well with the metrics that I have chosen. 

# %%
# checking the depth
print(plt.plot(final_model.depth,final_model.f1_weighted))

# %%
# variable importance
varimp = pd.DataFrame(best.feature_importances_,index = X.columns,columns=['importance']).sort_values('importance', ascending=False)
print(varimp)

# num_voted_users and genres_others seem to have the most importance

# %%
# predicting on tuning data
y_tune_pred = model.best_estimator_.predict(X_tune)

# printing predictions
print("Predictions on Tune Data:")
print(y_tune_pred)

# evaluating performance
from sklearn.metrics import f1_score
f1_tune = f1_score(y_tune, y_tune_pred, average='weighted')
print("\nF1-Weighted Score on Tune Data:", f1_tune)

# The model's performance increased on the tuning set.

# %%
# confusion matrix
print(ConfusionMatrixDisplay.from_estimator(best,X_tune,y_tune, display_labels = 
                                            ['Avg','Top'], colorbar=True))

# This confusion matrix is telling me that the model performed well in predicting most of the
# average movies (0s) correctly, with only a few false positives (1s predicted as 0s).

# The model also correctly identified a good number of top-rated movies (1s)
# but there were some false negatives (0s predicted as 1s).

# %%
# top three movies 
tune_data = X_tune.copy()
tune_data['top_imdb_score'] = y_tune 

top_movies = tune_data.sort_values(by='top_imdb_score', ascending=False)

# Unfortunately, when I was cleaning in the first part, I removed the movie titles.
# Merging the original dataset to get the movie titles back.

mm = pd.read_csv("../data/movie_metadata.csv")

if 'movie_title' in mm.columns:
	top_movies = top_movies.merge(mm[['movie_title']], left_index=True, right_index=True, how='left')
	top_movies = top_movies[['movie_title', 'top_imdb_score']]
	print("Top 3 Movies with Highest IMDb Scores:")
	print(top_movies.sort_values(by='top_imdb_score', ascending=False).head(3))

# %%
# different hyperparameter

scoring = 'f1_weighted'
param2 = {"criterion" : ['gini', 'entropy']}

# fitting model
cl2= DecisionTreeClassifier(random_state=1000)

# search for best DecisionTreeClassifier estimator across all folds based f1_weighted
search2 = GridSearchCV(cl2, param2, scoring=scoring, n_jobs=-1, cv=kf,refit='f1_weighted')

# execute on training data
model2 = search2.fit(X_train, y_train)

# best hyperparameters and score
best2 = model2.best_estimator_
print(best2) 

# %%
# extracting scores to view model

# scores: 
f1_w = model2.cv_results_['mean_test_score']
SD_f1_w = model2.cv_results_['std_test_score']

# parameter:
crit = np.unique(model2.cv_results_['param_criterion']).data

# building dataframe
final_model2 = pd.DataFrame(list(zip(depth, 
                                    f1_w, 
                                    SD_f1_w)),
                           columns=['crit', 'f1_weighted', 'f1_weightedSD'])

# final model
final_model2.style.hide(axis='index')

# The model performed well but not better than my first model with max_depth as the hyperparameter.

# %%
# predicting test data 
y_test_pred = model.best_estimator_.predict(X_test)

# printing predictions
print("Predictions on Test Data:")
print(y_test_pred)

# evaluating performance
f1_test = f1_score(y_test, y_test_pred, average='weighted')
print("\nF1-Weighted Score on Test Data:", f1_test)

# The F1-Weighted Score on the test data is approx 0.836, which indicates that the model is performing well on unseen data.

# %% [markdown]
# **Final Recommendation:**
# 
# The Decision Tree Classification Model can be used to predict movie success based on the features provided. I learned that I need to one-hot-encode my categorical variables because the sci-kit learn package only takes numerical variables to make decisions and that I need to collapse my target variable to also be numeric. 
# 
# 
# My model's accuracy on the test data is higher than usual, but this is also because I dropped some features that I believed were not relevant. However, everyone's idea of a "good" movie is different so it is truly difficult to determine a final recommendation. But for now, my final recommendation would be to use this model to predict movie success if there is a large number of votes and a genre that is not in the 'Comedy' or 'Action' category.
# 


