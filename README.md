## Improving the Yelp Review Experience by Stardardizing Reviewer Sentiment

### Team: 
  * Angela Detweiler
  * Hee Kang
  * Alexander Lam
  * Behesteh Mostaghni

**Dataset link:** Yelp Dataset in Kaggle with a focus on Restaurants- https://www.kaggle.com/yelp-dataset/yelp-dataset

**Problem:** When you are researching restaurants on Yelp, do you look at the star rating or do you read the review? Do you look at both? Given that reviews are highly subjective, and star ratings can be influenced by various aspects of business performance, can we use machine learning to standardize the interpretation of reviews? 

**Goal:** Our goal is to apply Natural Language Processing (NLP) and other features from the Yelp reviews into a model that outputs a new 5-star-rating, so that there is less discrepancy between reviews and star ratings. In order to make our model more robust, we will also incorporate new user star-ratings based on reviews read (meaning that someone who did not write the review gives a star-rating based on the review text alone) into our model so that it better reflects the review sentiment. 

**Hypothesis:** We hypothesize that automating star ratings based on NLP of restaurant reviews will improve Yelp review experience by normalizing reviewer sentiment.

**ML algorithms:**
  1. Naive Bayes
  2. k-NN
  3. K-Means
  4. LSTM
  5. N-Gram
  6. TD-IDF
  7. Linear Regression
  
 **Libraries:**
 1. Numpy
 2. Scipy
 3. Scikit_Learn
 4. Pandas
 5. Matplotlib
 6. NLTK
 7. PySpark
 8. Keras
 9. HTML/ CSS/ Bootstrap
 10. Tableau
 
 **Sentiment Analysis Lexicon:**
 1. AFINN 
 2. VADER
 
**Project components, steps, analyses, and final products:**
  1. Components and final products 
      * ML algorithms
      * Game (user rates reviews)/HTML page
      * Database with game data to be reincorporated into model
      * Model output/vizualizations in JN

  2. Steps and analyses
      * Select and clean restaurant/food category data from Yelp
      * Cluster reviews into 5 categories (5 star-rating)
      * Use NLP to train model
      * Test Yelp rating/review data (user inputs both)
      * Incorporate new user star-rating from game into the model
      * Other...

**Questions/Topics of Interest:**

1. (ML) Are yelp reviews highly correlated to restaurant quality (based on star rating) ? In other words, are the reviews useful? 
2. What percentage of reviews talk about the quality of the food versus the quality of the service?
3. Correlate photo captions to reviews.
4. (ML) Is there consistency in review style for a particular user?
5. Distribution of  ratings (stars)- Is it a bell curve or does it peak at both extremes (1 and/or 5 star ratings)?
6. (ML) Is there a pattern to Yelp Elite status? Elite vs non-elite.
7. Patterns in ratings/review sentiment correlated to business attributes? (Outdoor seating, live music, etc.)
8. Patterns in 'useful' reviews?
9. Use NLP to train model, test then have HUMANS rate as well and compare the difference
