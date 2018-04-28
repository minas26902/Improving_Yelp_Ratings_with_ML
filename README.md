## UCBBEL Projet #3 ML Proposal

## Title: Correlating Yelp Reviews with Star Ratings 
  OR Automating star ratings in Yelp based on Natural Language Processing
  OR Yelpgorithm 


## Team: Alex, Hee, Beheshteh, and Angela

## Dataset link - Yelp Dataset in Kaggle with a focus on Restaurants- https://www.kaggle.com/yelp-dataset/yelp-dataset

## Link to Github repo: https://github.com/minas26902/UCB_Project3_ML

### Goal: When you are researching restaurants on Yelp, do you look at the star rating or do you read the review? Do you look at both? A star rating doesn’t always reflect the review and vice-versa, since these are input separately by the reviewer, introducing some rating bias. Our goal is to apply Natural Language Processing (NLP) on Yelp reviews in order to output a star rating so that reviews and ratings are more strongly correlated. In order to make our model more robust, we will have humans rate reviews, and we will incorporate these ratings into our test model to get a sense of how humans are reading the sentiment.

### Hypothesis: We hypothesize that automating star ratings based on NLP of restaurant reviews will improve Yelp review experience, by normalizing reviewer sentiment.

### ML algorithms to be used: Clustering, NLP (best training model? TBD)

### Project components, steps, analyses, and final products -TBD
1. Components and final products 
  * ML 
  * Game (user rates reviews)/HTML page
  * Database with game data to be reincorporated into model
  * Visualizations

2. Steps and analyses - 
  * Select and clean restaurant/food category data from Yelp
  * Cluster reviews into 5 categories (5 star-rating)
  * Use NLP to train model
  * Test Yelp rating/review data (user inputs both)
  * Incorporate new human ratings from game into the model
  * Other...

-------
### Potential Questions/Topics (Team - PLEASE TAKE A LOOK AT THE DATA AND KEEP ADDING QUESTIONS. WE CAN NARROW THESE DOWN LATER)

1. (ML) Are yelp reviews highly correlated to restaurant quality (based on star rating) ? In other words, are the reviews useful? 
2. What percentage of reviews talk about the quality of the food versus the quality of the service?
~~3. Is there a difference in reviews in touristy versus local neighborhoods?~~
4. Correlate photo captions to reviews.
~~5. Correlate reviews for ethnic restaurants to demographics of the area. Would need to add U.S. Census data.~~
~~6. How do chain restaurant ratings compare in different demographic areas? Would need U.S. Census data.~~
7. (ML) Is there consistency in review style for a particular user?
8. Distribution of  ratings (stars)- Is it a bell curve or does it peak at both extremes (1 and/or 5 star ratings)?
9. (ML) Is there a pattern to Yelp Elite status? Elite vs non-elite.
10. Patterns in ratings/review sentiment correlated to business attributes? (Outdoor seating, live music, etc.)
11. Patterns in 'useful' reviews?
12. TEST IDEA: Use NLP to train model, test then have HUMANS rate as well and compare the difference

