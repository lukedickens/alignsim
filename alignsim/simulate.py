import numpy as np
from scipy.stats import multinomial
from scipy.stats import dirichlet_multinomial
from scipy.special import gammaln


    
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
    

def generate_fixed_accuracy_annotations(gt, c):
    """
    Generates annotations with an exact number of correct classifications.
    
    Exactly 'c' instances will match the ground truth. For the remaining 
    instances, an incorrect label is chosen uniformly from the other K-1 classes.

    Args:
        gt (np.array): Ground truth labels.
        c (int): The exact number of correct classifications required.

    Returns:
        np.array: Synthetic annotations array.
    """
    n = len(gt)
    if not (0 <= c <= n):
        raise ValueError(f"Number of correct samples 'c' must be between 0 and {n}.")

    K = np.max(gt) + 1
    anno = np.zeros_like(gt)
    
    # 1. Determine which indices will be correct
    indices = np.arange(n)
    np.random.shuffle(indices)
    correct_indices = indices[:c]
    incorrect_indices = indices[c:]
    
    # 2. Assign correct labels
    anno[correct_indices] = gt[correct_indices]
    
    # 3. Assign incorrect labels uniformly
    for idx in incorrect_indices:
        true_label = gt[idx]
        # Create a list of all classes except the true one
        possible_errors = [k for k in range(K) if k != true_label]
        anno[idx] = np.random.choice(possible_errors)
        
    return anno

def generate_uniform_confusion_matrix(K, accuracy):
    """
    Creates a K x K confusion matrix with a fixed diagonal accuracy 
    and errors distributed uniformly across off-diagonal elements.

    Args:
        K (int): Number of classes.
        accuracy (float): The probability of correct classification (0 <= a <= 1).

    Returns:
        np.array: A K x K matrix where rows sum to 1.0.
    """
    if K < 1:
        raise ValueError("Number of classes K must be at least 1.")
    if not (0 <= accuracy <= 1):
        raise ValueError("Accuracy must be between 0 and 1.")

    if K == 1:
        return np.array([[1.0]])

    # Calculate the probability for each error slot
    off_diag_prob = (1.0 - accuracy) / (K - 1)
    
    # Create a matrix filled with the off-diagonal probability
    cm = np.full((K, K), off_diag_prob)
    
    # Fill the diagonal with the target accuracy
    np.fill_diagonal(cm, accuracy)
    
    return cm
    
def generate_latent_bernoulli_results(n, distribution):
    """
    Generates binary labels (1 for correct, 0 for incorrect) based on 
    item-specific probabilities sampled from a given distribution.

    Args:
        n (int): Number of labels to generate.
        distribution: A scipy.stats distribution object (e.g., beta).

    Returns:
        np.array: Array of shape (n,) containing 0s and 1s.
    """
    # 1. Sample p_i for each item from the distribution support [0, 1]
    # Ensure the distribution is bounded or clipped to [0, 1]
    p_samples = distribution.rvs(size=n)
    p_samples = np.clip(p_samples, 0, 1)
    
    # 2. Sample from Bernoulli(p_i) for each item
    # np.random.random generates values in [0, 1); 
    # if the value is less than p_i, the outcome is 'Correct' (1)
    results1 = (np.random.random(n) < p_samples).astype(int)
    results2 = (np.random.random(n) < p_samples).astype(int)
    
    return p_samples, results1, results2


