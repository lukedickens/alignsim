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
    
    
def generate_annotations(gt, confusion_matrix):
    """
    Generates a synthetic set of annotations based on a fixed ground truth
    and a defined confusion matrix.
    
    Args:
        gt (np.array): Array of ground truth labels.
        confusion_matrix (np.array): K x K matrix where rows represent ground 
                                     truth and columns represent predictions.
    
    Returns:
        np.array: Synthetic annotations array of the same shape as gt.
    """
    K = confusion_matrix.shape[0]
    anno = np.zeros_like(gt)
    
    # Calculate transition probabilities (normalise rows)
    # Adding a tiny epsilon prevents division by zero for empty rows
    row_sums = confusion_matrix.sum(axis=1, keepdims=True)
    probs = confusion_matrix / np.maximum(row_sums, 1e-9)
    
    for k in range(K):
        # Identify indices where ground truth is class k
        indices = np.where(gt == k)[0]
        n_samples = len(indices)
        
        if n_samples > 0:
            # Sample new labels based on the probability distribution for class k
            sampled_labels = np.random.choice(K, size=n_samples, p=probs[k])
            anno[indices] = sampled_labels
            
    return anno
