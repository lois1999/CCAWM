from bayes_opt import BayesianOptimization
from bayes_opt import SequentialDomainReductionTransformer
from train_optimize import train_ints


# Bounded region of parameter space
pbounds = {'d_model': (2, 100), 'nhead': (2, 100), 'num_encoder_layers': (2, 8), 'num_decoder_layers': (2, 8), 
           'dim_feedforward': (15, 150), 'dropout': (0, 1), 'learning_rate': (0.001, 0.05), 'num_epochs': (10, 200)}

bounds_transformer = SequentialDomainReductionTransformer()

optimizer = BayesianOptimization(
    f=train_ints,
    pbounds=pbounds,
    random_state=1,
    bounds_transformer=bounds_transformer
)

optimizer.probe(
    params={"nhead": 4, "d_model": 16, "dim_feedforward": 75, "dropout": 0, "learning_rate":0.005, "num_decoder_layers":3, "num_encoder_layers":3, "num_epochs":17},
    lazy=True,
)


optimizer.maximize(
    init_points=4,
    n_iter=30
)

print(optimizer.max)
