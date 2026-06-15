from itertools import combinations_with_replacement
import os
import pickle
import numpy as np
from scipy.special import factorial
from math import gcd
from functools import reduce
from collections import Counter


# Precomputed generalized Klyshko / majorization pairs for consecutive
# indices indx = [0, 1, ..., N_P - 1]. File genKlyshko_up_toM.pkl contains
# the pairs for N_P = M + 1, so list_of_files[N_P - 4] is the matching file.
# Paths are resolved relative to the repository root (this file lives in
# src/analytical/), so loading works regardless of the current directory.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_GENK_DIR = os.path.join(_REPO_ROOT, "data", "all_genK")
list_of_files = [os.path.join(_GENK_DIR, "genKlyshko_up_to3.pkl"),
                 os.path.join(_GENK_DIR, "genKlyshko_up_to4.pkl"),
                 os.path.join(_GENK_DIR, "genKlyshko_up_to5.pkl"),
                 os.path.join(_GENK_DIR, "genKlyshko_up_to6.pkl"),
                 os.path.join(_GENK_DIR, "genKlyshko_up_to7.pkl"),
                 os.path.join(_GENK_DIR, "genKlyshko_up_to8.pkl")]


def load_precomputed_pairs(indx, lof=list_of_files):
    """
    Return the precomputed majorization pairs for the given index set, or
    None if no precomputed file is applicable.

    Precomputed pairs are only available for consecutive indices
    indx = [0, 1, ..., N_P - 1] within the range covered by ``lof``.
    """
    indx = np.asarray(indx, dtype=int).reshape(-1)
    NP = len(indx)

    # Precomputed pairs assume consecutive indices starting from 0.
    if not np.array_equal(indx, np.arange(NP, dtype=int)):
        return None

    file_idx = NP - 4
    if file_idx < 0 or file_idx >= len(lof):
        return None

    filename = lof[file_idx]
    if not os.path.exists(filename):
        return None

    with open(filename, "rb") as f:
        return pickle.load(f)


def generate_arrays(N,M,indx=np.array([0,1,2,3],dtype=int)):
    result = []
    for comb in combinations_with_replacement(indx, N):
        if sum(comb) == M:
            sorted_comb = tuple(sorted(comb, reverse=True))
            result.append(sorted_comb)
    return np.array(result)

def majorization_rule(array1, array2):
    return np.all(np.cumsum(array1) >= np.cumsum(array2))

def generate_majorization_pairs(arrays):
    pairs = []
    for i in range(len(arrays)):
        for j in range(len(arrays)):
            if majorization_rule(arrays[i], arrays[j]):
                pairs.append((arrays[i], arrays[j]))
    return pairs

def filter_non_overlapping_pairs(pairs):
    filtered_pairs = []
    
    for pair in pairs:
        array1, array2 = set(pair[0]), set(pair[1])
        if array1.isdisjoint(array2):
            filtered_pairs.append(pair)
    
    return filtered_pairs

def gcd_of_list(lst):
    return reduce(gcd, lst)

def multiplicities_gcd(array):
    counts = Counter(array)
    gcd_value = gcd_of_list(list(counts.values()))
    return gcd_value

def filter_pairs_with_common_factors(pairs):
    final_pairs = []
    
    for pair in pairs:
        aux = np.append(pair[0], pair[1])
        gcd = multiplicities_gcd(aux)
        
        if gcd == 1:
            final_pairs.append(pair)
    
    return final_pairs


def all_pairs_for_list_of_P(indx=np.array([0,1,2,3])):
    
    M=np.sum(range(np.max(indx)+1),dtype=int)
    N=M
    
    all_p=[]
    for i in range(N+1):
        for j in range(1,M+1):
            arrays = generate_arrays(i, j,indx=indx)
            pairs = generate_majorization_pairs(arrays)
            pairs_aux = filter_non_overlapping_pairs(pairs)
            pairs_aux2 = filter_pairs_with_common_factors(pairs_aux)
            for pair in pairs_aux2:
                all_p.append(pair)              
    return all_p

def analytical_m(Pall=np.ones((10,10,3)),indx=np.array([0,1,2],dtype=int),lof=list_of_files):


    Psh = Pall.shape

    NP = Psh[-1]
    if (NP != len(indx)):
        print("The array of probabilities and the index of probailities are different!")
        exit(0)

    rev_index=np.zeros((np.max(indx)+1),dtype=int)
    for i in range(NP):
        rev_index[indx[i]]=i

    nonclass = np.zeros((Psh[0],Psh[1]), dtype=bool)

    # Use precomputed majorization pairs when available, otherwise generate
    # them on the fly.
    all_pairs = load_precomputed_pairs(indx, lof=lof)
    if all_pairs is None:
        all_pairs = all_pairs_for_list_of_P(indx=indx)


    for pair in all_pairs:
        print(pair)
        fact0 = factorial(pair[0])
        fact1 = factorial(pair[1])
        fact = np.prod(fact0/fact1)

        
        Paux0 = fact*np.ones((Psh[0],Psh[1]))
        Paux1 = np.ones((Psh[0],Psh[1]))
         
        for i in pair[1]:
            Paux1 *= Pall[:,:,rev_index[i]]
            
        for i in pair[0]:
            Paux0 *= Pall[:,:,rev_index[i]]
            
        Paux = Paux1 - Paux0
    
        nonclass = np.where(Paux > 0,True,nonclass)
        
    ncaf = np.sum(nonclass.astype(int))/(Psh[0]*Psh[1])
    return nonclass,ncaf,len(all_pairs)