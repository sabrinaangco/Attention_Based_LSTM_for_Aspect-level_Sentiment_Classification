# NLP_project

## Files to download
Download glove.42B.300d.txt and add it to main folder in order to run code, can be downloaded here:
https://www.kaggle.com/yutanakamura/glove42b300dtxt

## Required Libraries
numpy==1.20.1
transformers==3.5.1
pytorch==1.7.1
sklearn
spacy
python -m spacy download en_core_web_sm

## Run Training
### train.py
%run train.py --model_name lstm --dataset restaurant
### train_k_fold_cross_val.py
%run train_k_fold_cross_val.py --model_name lstm --dataset restaurant

## Arguments for training
('--model_name', default='lstm', type=str, help='lstm, td_lstm, tc_lstm)
('--dataset', default='laptop', type=str, help='twitter, restaurant, laptop')
('--optimizer', default='adam', type=str, help= 'adadelta, adagrad, adam, adamax, Adamax, asgd, rmsprop, sgd')
('--initializer', default='xavier_uniform_', type=str, help='xavier_uniform_, xavier_normal_, orthogonal_)
('--lr', default=2e-5, type=float, help='try 1e-3')
('--dropout', default=0.1, type=float)
('--l2reg', default=0.01, type=float)
('--num_epoch', default=20, type=int, help='try larger number ')
('--batch_size', default=16, type=int, help='try 16, 32, 64')
('--log_step', default=10, type=int)
('--embed_dim', default=300, type=int)
('--hidden_dim', default=300, type=int)
('--max_seq_len', default=85, type=int)
('--polarities_dim', default=3, type=int)
('--hops', default=3, type=int)
('--patience', default=5, type=int)
('--device', default=None, type=str, help='e.g. cpu')
('--seed', default=1234, type=int, help='set seed for reproducibility')
('--valset_ratio', default=0, type=float, help='set ratio between 0 and 1 for validation support')



