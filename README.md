# JokeRecommender
My solution to the Jester Practice Problem on analyticsvidhya.com -- https://datahack.analyticsvidhya.com/contest/jester-practice-problem/

To run the model and write a prediction csv, simply run:

  python3 run.py
  
Runs on Python 3 and requires numpy and pandas. 
The output will show up in /predictions/pred.csv. 
Parameters X and Theta are cached in /cache/ after training, and loaded for use the next time. 
If you want to train a new model you can delete /cache/X.npy and /cache/Theta.npy or modify run.py.
Modify run.py to change the hyper parameters of the model.
