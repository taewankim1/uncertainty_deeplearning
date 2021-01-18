# Uncertainty in deep learning
This reposit compares the uncertainty in Gaussian Process Regression and other methods developed for deep learning such as Ensemble, Bayesian approach based on the dropout. It performs 1D regression simulations where the training data is corrupted with both epistemic and aleatoric unceratinty. Your can start with Comparison.ipynb in notebooks folder.

## Summary of results
-  Training data
<img src="images/data.png">
Two types uncertainty in training data.

-  GP regression
<img src="images/GP.png">
Good at epistemic uncertainty. Not valid for aleatoric uncertainty.

-  Neural net
<img src="images/NN.png">
No generating uncertainty.

-  Ensemble 
<img src="images/Ensemble.png">
Good at epistemic uncertainty, but somewhat underestimate the uncertainty. Not valid for aleatoric uncertainty

-  Bayeisn neural net with dropout
<img src="images/NN_dropout.png">
Valid for epistemic uncertainty, but not good.

-  Density network
<img src="images/density.png">
Valid only for aleatoric uncertainty. Unstable in traning process due to negative log-likelihood loss.

## References
* Bayesiean NN based on dropout: https://arxiv.org/pdf/1703.04977.pdf
* Gaussian process regression Code from Sungjoon Choi(https://github.com/sjchoi86)
