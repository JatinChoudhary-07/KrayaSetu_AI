from sklearn.cluster import DBSCAN
import numpy as np

def cluster_defects_into_blocks(defects: list, eps: float = 2.0, min_samples: int = 1) -> dict:
    """
    Cluster defects along a 1-D corridor chainage using DBSCAN.
    The spec mandates scoping this to one NetworkX-connected-component at a time.
    
    Args:
        defects: List of defect dictionaries, expected to contain "work_id" and "chainage_km".
        eps: Operational parameter (~2km) defining max distance for neighborhood.
        min_samples: Set to 1 so isolated defects remain valid singleton blocks.
    
    Returns:
        dict: Mapping of cluster_id to a list of defect work_ids.
    """
    if not defects:
        return {}
        
    # Extract 1-D chainage distances for DBSCAN
    X = np.array([d.get("chainage_km", 0.0) for d in defects]).reshape(-1, 1)
    
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(X)
    
    clusters = {}
    for idx, label in enumerate(labels):
        # With min_samples=1, noise (-1) is impossible, but handle it just in case.
        cluster_id = f"CLUSTER-{label}" if label != -1 else f"CLUSTER-NOISE-{idx}"
        work_id = defects[idx]["work_id"]
        
        if cluster_id not in clusters:
            clusters[cluster_id] = []
        clusters[cluster_id].append(work_id)
        
    return clusters
