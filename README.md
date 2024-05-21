# Likelihood-ratio confidence intervals

This repository contains the code used for the experiments in our paper 'Likelihood-ratio-based confidence intervals for neural networks'. The following files correspond to the experiments:

- toy_example_regression.py: A short one-dimensional example where DeepLR is used to construct confidence intervals for a quadratic regression function.
- xgboost_example: The same example but with the XGBoost model.
- toy_example_classification.py: A short one-dimensional binary classification example where DeepLR is used to construct confidence intervals for the predicted class-1 probability.
- two_moon_example.py: A two-dimensional binary classificaiton problem based on the well-known two-moon data set. Notably, DeepLR produces high uncertainty in regions further away from the data, a behavior that is not seen in Deep Ensembles and MC-dropout.
- mnist_example.py: An experiment that shows that DeepLR confidence intervals are notably larger for OoD inputs.
- cifar_example.py: An experiment that shows that DeepLR confidence intervals are notably larger for OoD inputs. Additionally, two adversarial examples are tested.
- brain_tumor_example: An experiment that shows that DeepLR applies DeepLR to brain-tumor detection.

The code for the actual construction of the confidence intervals can be found in 'intervals_classification.py' and 'intervals_regression.py'. The file 'adversarial_example.py' contains a function to create adverarial examples based on the FGSM-method. All necessary modules including the versions used for this project can be found in 'requirements.txt'.

