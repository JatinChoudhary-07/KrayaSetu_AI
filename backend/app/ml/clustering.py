from sklearn.cluster import DBSCAN
import numpy as np

EPS_KM = 2.0
MIN_SAMPLES = 1
SETUP_MINUTES_DEFAULT = 20

def cluster_defects_into_blocks(defects: list[dict]) -> list[dict]:
    """
    defects: already restricted to ONE NetworkX-adjacent connected component.
    Each dict needs: id, chainage_km, block_section_id,
    earliest_start_min, latest_start_min, dur_i, priority_score.
    """
    if not defects:
        return []
        
    coords = np.array([[d["chainage_km"]] for d in defects])
    labels = DBSCAN(eps=EPS_KM, min_samples=MIN_SAMPLES).fit_predict(coords)

    blocks = []
    for cluster_id in set(labels):
        members = [d for d, lbl in zip(defects, labels) if lbl == cluster_id]
        for group in _split_by_window_overlap(members):
            # Form candidate block
            blocks.append({
                "block_id": f"CLUSTER-{group[0]['block_section_id']}-{cluster_id}-{len(blocks)}",
                "job_ids": [m["id"] for m in group],
                "earliest_start": max(m["earliest_start_min"] for m in group),
                "latest_start": min(m["latest_start_min"] for m in group),
                "total_duration": sum(m["dur_i"] for m in group) + SETUP_MINUTES_DEFAULT,
            })
    return blocks

def _split_by_window_overlap(members: list[dict]) -> list[list[dict]]:
    members = sorted(members, key=lambda m: m["earliest_start_min"])
    groups, current, window_end = [], [], None
    for m in members:
        if not current or m["earliest_start_min"] <= window_end:
            current.append(m)
            window_end = min(window_end, m["latest_start_min"]) if window_end else m["latest_start_min"]
        else:
            groups.append(current)
            current, window_end = [m], m["latest_start_min"]
    if current:
        groups.append(current)
    return groups
