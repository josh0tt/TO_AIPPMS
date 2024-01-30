using Random
####################################################
# Shared parameters
####################################################
rng = MersenneTwister(1234)
verbose=false
logging=false
use_ssh_dir = false

objective = "variance"#"expected_improvement" # "variance"
path_name = "/Users/data/results_ssh_TO_AIPPMS/"*objective #"/Users/data/results/" * objective #"/Users/data/additional_results/" * objective #"/Users/data/additional_results/larger_map/"# #use_ssh_dir ? "/data/results_ssh_TO_AIPPMS/" * objective : "/data/results/" * objective

total_budget = 60.0#30.0#100.0
σ_drill = 1e-9
σ_spec = 0.1#1.0 #1.0#0.1#1.0
sample_cost = 3#10
init_num_drills = 3

bins_x = 10 #100 #100
bins_y = 10 #100 #50

x0 = [0.003,0.01]
xf = [1.0, 1.0]
N = Int(total_budget) 
h = 1.0 
ud_max = 0.1#we set to 0.1 because SBO incurs more cost for diag movement b/c it moves further, but for PTO we constrain it to max movemento of 0.1 in any direction so they are equivalent #sqrt(0.1^2 + 0.1^2)#0.10 

replan_rate = 1

# NOTE: if the domain is 1x1 then length scale of 0.01 corresponds to a length scale of 1 on the 100x100 domain
length_scale = 0.1
k = with_lengthscale(SqExponentialKernel(), length_scale) 
@everywhere m(x) = 0.0 
X_query = [[i,j] for i = range(0, 1, length=bins_x+1), j = range(0, 1, length=bins_y+1)]
query_size = size(X_query)
X_query = reshape(X_query, size(X_query)[1]*size(X_query)[2])
KXqXq = K(X_query, X_query, k)
GP = GaussianProcess(m, μ(X_query, m), k, [], X_query, [], [], [], [], KXqXq);
f_prior = GP

domain_min = [0.,0.]
domain_max = [1., 1.]
domain_bins = [bins_x+1, bins_y+1]


# Optim.jl parameters
optim_time_limit = 1.0
optim_iterations = 20 
optim_f_calls_limit = 30
optim_g_calls_limit = 30

####################################################
# Ergodic
####################################################
K_fc = 5 # number of Fourier Coefficients
phi =  Matrix(reshape(query_no_data(GP)[2], (bins_x+1, bins_y+1))') #gaussian(d, [mu1,mu2], [Sigma1, Sigma2])
phi_end = 1
phi[end,end] = phi_end
####################################################
# GP_PTO
####################################################
max_iters = 50
# perc improvement should be negative if the score decreased (we want to minimize the score)
sample_addition_threshold = 0.0 #120#5.0 # percent improvement variance has to reach for a drill to be added in PTO methods


####################################################
# BMDP
####################################################
num_trials = 50
number_of_sample_types=10
rollout_depth = 5
rollout_iterations = 100
plot_results=true
run_raster=false
run_random=true

map_size_sboaippms = (bins_x+1, bins_y+1)

k_sboaippms = with_lengthscale(SqExponentialKernel(), length_scale*10) # NOTE: check length scale
X_query_sboaippms = [[i,j] for i = 1:map_size_sboaippms[1], j = 1:map_size_sboaippms[2]]#[[i,j] for i = range(0, 1, length=bins_x+1), j = range(0, 1, length=bins_y+1)]
query_size_sboaippms = size(X_query_sboaippms)
X_query_sboaippms = reshape(X_query_sboaippms, size(X_query_sboaippms)[1]*size(X_query_sboaippms)[2])
KXqXq_sboaippms = K(X_query_sboaippms, X_query_sboaippms, k)
GP_sboaippms = GaussianProcess(m, μ(X_query_sboaippms, m), k_sboaippms, [], X_query_sboaippms, [], [], [], [], KXqXq_sboaippms);
f_prior_sboaippms = GP_sboaippms