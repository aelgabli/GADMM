# GADMM: fast and communication efficient framework for distributed machine learning


 For linear regression on synthetic data, run LinearRegression_synthetic.m.
 
 For linear regression on real data, run LinearRegression_real.m.
 
 For logistic regression on synthetic data, run LogisticRegression_synthetic.m.
 
 For logistic regression on real data, run LogisticRegression_real.m.
 
All codes will run regression tasks using our proposed algorithm (GADMM) and all baseline schemes described in our paper (see below).

For linear regression using D-GADMM (regression over dynamic network) and synthetic dataset run Dynamic_LinearRegression_Synthetic.m.

For linear regression using D-GADMM (regression over dynamic network) and real dataset run dynamic_LinearRegression_Real.m


To Compare D-GADMM with GADMM and standared ADMM (ADMM with parameter server. i.e., star topology) using synthetic dataset, run LinearRegression_gadmm_vs_admm.m

The datasets used in this code are available at:
https://www.dropbox.com/sh/kphv5o9uelynci2/AADNQF3HCr1tWS6_OlJSWzZea?dl=0

# Citation

@article{elgabli2019gadmm,
  title={GADMM: Fast and communication efficient framework for distributed machine learning},
  author={Elgabli, Anis and Park, Jihong and Bedi, Amrit S and Bennis, Mehdi and Aggarwal, Vaneet},
  journal={arXiv preprint arXiv:1909.00047},
  year={2019}
}
