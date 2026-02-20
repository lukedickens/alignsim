import numpy as np
import pandas as pd

from alignsim.simulate import (
    generate_uniform_confusion_matrix, 
    generate_annotations, 
    generate_fixed_accuracy_annotations,
    generate_latent_bernoulli_results
)
from alignsim.calculate import (
    calculate_3d_agreement,
    calculate_empirical_accuracy,
    calculate_geirhos_error_matrix,
    calculate_geirhos_metrics,
    calculate_geirhos_from_validations
)

from alignsim.distributions import (
    get_beta_dist
)

def run_accuracy_vs_kappa_sim(gt, fixed_acc1, acc2_range, K, n_trials=100):
    results = []
    n = len(gt)
    c1 = int(fixed_acc1 * n)
    
    for acc2 in acc2_range:
        c2 = int(acc2 * n)
        for _ in range(n_trials):
            # Generate annotations with exact correct counts
            anno1 = generate_fixed_accuracy_annotations(gt, c1)
            anno2 = generate_fixed_accuracy_annotations(gt, c2)
            
            # Calculate agreement metrics
            T, A, C1, C2 = calculate_3d_agreement(gt, anno1, anno2, K=K)
            geirhos_mtx = calculate_geirhos_error_matrix(T)
            metrics = calculate_geirhos_metrics(geirhos_mtx)
            
            results.append({
                'accuracy_2': acc2,
                'kappa': metrics['kappa']
            })
        
    return pd.DataFrame(results)

def run_facility_experiment(means, fixed_var, N=1000, n_trials=50):
    """
    Runs a simulation sweeping through mean accuracies with fixed variance.
    Calculates Error Consistency (Kappa) for shared item difficulty.
    """
    results = []
    
    for mu in means:
        try: 
            dist = get_beta_dist(mu, fixed_var)
        except:
            dist = None
            
        if dist is None: # Skip if variance is mathematically impossible for the mean
            continue
            
        for _ in range(n_trials):
            p_samples, val1, val2  = generate_latent_bernoulli_results(N, dist)
            
            # 2. Calculate Geirhos Matrix & Kappa
            mtx = calculate_geirhos_from_validations(val1, val2)
            kappa = calculate_geirhos_metrics(mtx)['kappa']
            
            results.append({'mean_accuracy': mu, 'kappa': kappa})
            
    return pd.DataFrame(results)

