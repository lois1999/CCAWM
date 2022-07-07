# Compositional generalization capability of Transformer

This project is part of the course Cognitive and Computational Aspects of Word Meaning (2022) at Utrecht University.

We have expanded on experiments done by J. Athmer and D. Paperno, investigating if morphological transparency is able to help the Transformer model perform better on tasks requiring compositional or systematic generalization.


### References
Code used as a baseline : J. Athmer https://github.com/JanAthmer/Compositional-generalization-capabillity-of-Transformer

Transformer model: https://arxiv.org/abs/1706.03762

SCAN dataset: https://arxiv.org/abs/1711.00350

### Repository structure
* transformer_scan
  * data (all data used in experiments)
    * scan (SCAN task)
      * addjump 		(systematic generalization split)
      * addleft 		(systematic generalization split)
      * addleft_removedleft	(split to check whether the "TURN LEFT" command was used by the model)
      * addx			  (replaced "left" with "x")
      * churny			(replaced "turn left" with "churn y")
      * length			(split based on length)
      * simple 			(random split)
      * split1      (morphological transparency)
      * split2      (morphological ambiguity)
      * uniquesuffix (unique suffix for each word)
      * jumpo       (only verbs have suffix)
      * verbarg     (suffix for verbs conditioned on whether they necessarily take an argument)
      * simplesplit1 (split 1 applied to the simple split)
    * nacs (NACS task)
      * addjump 		(systematic generalization split)
      * addleft 		(systematic generalization split)
      * addleft_removedleft	(split to check whether the "TURN LEFT" command was used by the model)
      * addx			(replaced "left" with "x")
      * churny			(replaced "turn left" with "churn y")
      * length			(split based on length)
      * simple 			(random split)	
      * split1      (morphological transparency)
      * split2      (morphological ambiguity)
      * uniquesuffix (unique suffix for each word)
      * jumpo       (only verbs have suffix)
      * verbarg     (suffix for verbs conditioned on whether they necessarily take an argument)
      * simplesplit1 (split 1 applied to the simple split)
  * models
    * transformer (PyTorch code implementing transformer model)
  * results (raw results from all experiments)
  * main.py (main function - mostly just gathers arguments and runs train())
  * optimization.py (script for Bayesian optimization to find optimal hyperparameters)
  * train.py (main training function)
  * train_optimize.py (training function used for Bayesian optimization)
  * data.py (code for dealing with SCAN dataset)
  * test.py (function used for testing and computing accuracy)
  * test_optimize.py (testing function used for Bayesian optimization)
  * results.ipynb (notebook for displaying results in results/)
  * requirements.txt (list of all requirements)
  
### Dependencies
[pytorch](https://pytorch.org/)
```
conda install pytorch==1.4.0 -c pytorch
```
```
pip3 install torch==1.4.0
```
[torchtext](https://pytorch.org/text/)
```
conda install -c pytorch torchtext==0.6.0
```
```
pip3 install torchtext==0.6.0
```

The complete list of requirments are in requirements.txt. The most important dependencies are already listed above.

### To run (example):
Simple SCAN split:
```
python main.py --split simple_scan --out_data_file simple_scan
```

Simple NACS split:
```
python main.py --split simple_nacs --out_data_file simple_nacs
```
