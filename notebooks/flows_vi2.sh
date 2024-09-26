
# Prepare data
mkdir data/
wget https://raw.githubusercontent.com/AaltoML/stationary-activations/main/data/datapoints.csv
mv datapoints.csv data/
wget https://raw.githubusercontent.com/AaltoML/stationary-activations/main/data/classes.csv
mv classes.csv data/

# lower_beta
python flows_vi.py --nf_model=0  --beta=0.3 --beta_annealing=False --n_posterior_samples_total=100 --pretrain_nf=0 --random_seed=0
python flows_vi.py --nf_model=-1 --beta=0.3 --beta_annealing=False --n_posterior_samples_total=100 --pretrain_nf=0 --random_seed=0
python flows_vi.py --nf_model=1  --beta=0.3 --beta_annealing=False --n_posterior_samples_total=100 --pretrain_nf=0 --random_seed=0
python flows_vi.py --nf_model=11 --beta=0.3 --beta_annealing=False --n_posterior_samples_total=100 --pretrain_nf=0 --random_seed=0
python flows_vi.py --nf_model=2  --beta=0.3 --beta_annealing=False --n_posterior_samples_total=100 --pretrain_nf=0 --random_seed=0
python flows_vi.py --nf_model=21 --beta=0.3 --beta_annealing=False --n_posterior_samples_total=100 --pretrain_nf=0 --random_seed=0

python flows_vi.py --nf_model=3 --beta=0.3 --beta_annealing=False --n_posterior_samples_total=10 --pretrain_nf=0 --random_seed=0
python flows_vi.py --nf_model=31 --beta=0.3 --beta_annealing=False --n_posterior_samples_total=10 --pretrain_nf=0 --random_seed=0