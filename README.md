# JokeRecommender
My solution to the Jester Practice Problem on analyticsvidhya.com -- https://datahack.analyticsvidhya.com/contest/jester-practice-problem/

To train a model and write predictions to a csv, simply run:

  python3 run.py
  
Runs on Python 3 and requires numpy and pandas. 
The output will show up in /predictions/pred.csv. 
Parameters X and Theta are stored as files in /cache/ after training, and loaded for use the next time. 
If you want to train a new model you can delete /cache/X.npy and /cache/Theta.npy or modify run.py.
Modify run.py to change the hyper parameters of the model.

If you want to continue training a model using cached parameters X and Theta for more iterations, you can do so like this:

    X = np.load('cache/X.npy')
    Theta = np.load('cache/Theta.npy')
    X, Theta = model(params = (X, Theta))
    
The function model() will continue training with the given parameters X and Theta. Note that the number of features must be the same or the dimensions won't match up and it will cause an error. 
