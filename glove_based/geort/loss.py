import torch

def chamfer_distance(input_points, target_points):#, right, keypoint_ith, debug=False):
    """
    Args:
    - input_points (torch.Tensor): Input point cloud tensor of shape [B, N, 3].
    - target_points (torch.Tensor): Target point cloud tensor of shape [B, M, 3].

    Returns:
    - chamfer_dist (torch.Tensor): Chamfer distance.
    """
    B, N, _ = input_points.size()  # 1, 2048, 3 (embedded_point: human keypoint -[ik]-> qpos -[fk]-> output)
    _, M, _ = target_points.size() # 2048, 4, 3

    # keep originals for debug
    # inp_orig = input_points.detach().cpu()
    # tgt_orig = target_points.detach().cpu()

    input_points = input_points.clone()
    target_points = target_points.clone()
    input_points[..., 1] = input_points[..., 1]
    target_points[..., 1] = target_points[..., 1]

    # broadcasting for dimension extension
    input_points = input_points.unsqueeze(2)    # [B, N, 1, 3]
    target_points = target_points.unsqueeze(1)  # [B, 1, M, 3]

    # Replicate into NxM pairs
    input_points_repeat = input_points.repeat(1, 1, M, 1)    # [B, N, M, 3]
    target_points_repeat = target_points.repeat(1, N, 1, 1)  # [B, N, M, 3]

    # Sum of squared Euclidean distance for each pair
    dist_matrix = torch.sum((input_points_repeat - target_points_repeat)**2, dim=-1)  # [B, N, M]

    # Distance from one side to the closest point on the other side
    min_dist_a, _ = torch.min(dist_matrix, dim=2)  # [B, N]
    min_dist_b, _ = torch.min(dist_matrix, dim=1)  # [B, M]

    # Calculate Chamfer distance by averaging
    chamfer_dist = torch.mean(min_dist_a, dim=1) + torch.mean(min_dist_b, dim=1)

    return chamfer_dist.mean()

