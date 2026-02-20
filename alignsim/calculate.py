import numpy as np
from scipy.stats import multinomial
from scipy.stats import dirichlet_multinomial
from scipy.special import gammaln


def calculate_3d_agreement(gt, anno_1, anno_2, K=None):
    """
    Constructs a 3D contingency tensor and 2D marginal agreement matrices.
    
    The tensor T tracks the joint occurrences of (ground_truth, obs_1, obs_2), 
    allowing for a granular analysis of error consistency.

    Args:
        gt (np.array): Ground truth labels.
        anno_1 (np.array): Labels from the first annotator/model.
        anno_2 (np.array): Labels from the second annotator/model.
        K (int, optional): Number of classes. Inferred from gt if None.

    Returns:
        tuple: (T, A, C1, C2) where:
            T: 3D tensor of shape (K, K, K).
            A: Agreement matrix between anno_1 and anno_2.
            C1: Confusion matrix for anno_1 relative to ground truth.
            C2: Confusion matrix for anno_2 relative to ground truth.
            
    """
    if K is None:
        K = np.max(gt)+1
    # create tensor of agreement by groundtruth
    T = np.empty((K,K,K), dtype=int)
    for k0 in range(K):
        for k1 in range(K):
            for k2 in range(K):
                T[k0,k1,k2] = np.sum((gt==k0)&(anno_1==k1)&(anno_2==k2))
    # agreement matrix
    A = np.sum(T,axis=0)
    C1 = np.sum(T,axis=2)
    C2 = np.sum(T,axis=1)
    return T, A, C1, C2
    
def calculate_empirical_accuracy(conf_mtx):
    """
    Calculates the global accuracy from a confusion matrix.

    Args:
        conf_mtx (np.array): A K x K confusion matrix where rows are ground truth 
                             and columns are predicted labels.

    Returns:
        float: The proportion of correct classifications (trace / total sum).
    """
    total_samples = np.sum(conf_mtx)
    if total_samples == 0:
        return 0.0
    
    # The trace is the sum of the diagonal elements (correct predictions)
    correct_predictions = np.trace(conf_mtx)
    
    return correct_predictions / total_samples

def calculate_geirhos_error_matrix(T):
    """
    Collapses a 3D agreement tensor into a 2x2 error agreement matrix.
    
    The resulting matrix follows the orientation:
    [ [Both Incorrect,       Ann1 Incorrect & Ann2 Correct]
      [Ann1 Correct & Ann2 Incorrect,       Both Correct] ]

    Args:
        T (np.array): 3D tensor of shape (K, K, K) where indices are 
                      (ground_truth, anno_1, anno_2).

    Returns:
        np.array: A 2x2 contingency matrix of error patterns.
    """
    K = T.shape[0]
    
    # Initialize counts
    both_correct = 0
    ann1_corr_ann2_inc = 0
    ann1_inc_ann2_corr = 0
    both_incorrect = 0
    
    for k in range(K):
        # Slice the tensor for the current ground truth class k
        # row = anno_1, col = anno_2
        matrix_k = T[k, :, :]
        
        # 1. Both Correct: index [k, k] in the slice
        both_correct += matrix_k[k, k]
        
        # 2. Ann1 Correct, Ann2 Incorrect: index [k, not k]
        ann1_corr_ann2_inc += (np.sum(matrix_k[k, :]) - matrix_k[k, k])
        
        # 3. Ann1 Incorrect, Ann2 Correct: index [not k, k]
        ann1_inc_ann2_corr += (np.sum(matrix_k[:, k]) - matrix_k[k, k])
        
        # 4. Both Incorrect: everything else in this slice
        total_in_slice = np.sum(matrix_k)
        # Sum of all correct-related cells
        accounted_for = (matrix_k[k, k] + 
                         (np.sum(matrix_k[k, :]) - matrix_k[k, k]) + 
                         (np.sum(matrix_k[:, k]) - matrix_k[k, k]))
        both_incorrect += (total_in_slice - accounted_for)

    # Construct the 2x2 matrix in the specified orientation
    geirhos_matrix = np.array([
        [both_incorrect, ann1_inc_ann2_corr],
        [ann1_corr_ann2_inc, both_correct]
    ])
    
    return geirhos_matrix

def calculate_geirhos_metrics(geirhos_mtx):
    """
    Calculates c_exp, c_obs, and Cohen's Kappa from the Geirhos error matrix.
    
    Args:
        geirhos_mtx (np.array): 2x2 matrix where:
                                [0,0] = Both Incorrect
                                [0,1] = Ann1 Inc, Ann2 Corr
                                [1,0] = Ann1 Corr, Ann2 Inc
                                [1,1] = Both Correct
    
    Returns:
        dict: A dictionary containing 'c_exp', 'c_obs', and 'kappa'.
    """
    n = np.sum(geirhos_mtx)
    if n == 0:
        return {'c_exp': 0.0, 'c_obs': 0.0, 'kappa': 0.0}

    # e12 is the trace of the Geirhos error matrix (Both Inc + Both Corr)
    e12 = np.trace(geirhos_mtx)
    
    # c_obs: Observed error overlap
    c_obs = e12 / n
    
    # Calculate individual accuracies (p1 and p2)
    # p1: Accuracy of Annotator 1 (Bottom row sum)
    # p2: Accuracy of Annotator 2 (Right column sum)
    p1 = np.sum(geirhos_mtx[1, :]) / n
    p2 = np.sum(geirhos_mtx[:, 1]) / n
    
    # c_exp: Expected overlap due to chance
    c_exp = (p1 * p2) + ((1 - p1) * (1 - p2))
    
    # kappa: Cohen's Kappa of the Geirhos error matrix
    if c_exp == 1: # Avoid division by zero if both are 100% accurate
        kappa = 1.0
    else:
        kappa = (c_obs - c_exp) / (1 - c_exp)
        
    return {
        'c_exp': c_exp,
        'c_obs': c_obs,
        'kappa': kappa
    }
