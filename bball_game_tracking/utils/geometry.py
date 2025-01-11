import math

def is_within_ellipse(x, y, ellipse):
    """Check if a point is within an ellipse"""
    center_x, center_y, width, height = ellipse
    return ((x - center_x) ** 2) / (width / 2) ** 2 + ((y - center_y) ** 2) / (height / 2) ** 2 <= 1

def distance_to_ellipse_edge(x, y, ellipse):
    """Calculate distance from point to ellipse edge"""
    center_x, center_y, width, height = ellipse
    return math.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)

def calculate_player_bounds(frame_positions):
    """Optimized function to calculate player bounds for a single frame."""
    if not frame_positions or len(frame_positions) < 2:
        return None

    # Convert to numpy array for faster computation
    positions = np.array(frame_positions)
    x_coords = positions[:, 0]
    y_coords = positions[:, 1]

    # Quick outlier removal using percentiles
    x_25, x_75 = np.percentile(x_coords, [25, 75])
    y_25, y_75 = np.percentile(y_coords, [25, 75])

    x_iqr = x_75 - x_25
    y_iqr = y_75 - y_25

    x_mask = (x_coords >= x_25 - 1.5 * x_iqr) & (x_coords <= x_75 + 1.5 * x_iqr)
    y_mask = (y_coords >= y_25 - 1.5 * y_iqr) & (y_coords <= y_75 + 1.5 * y_iqr)
    mask = x_mask & y_mask

    if not np.any(mask):
        return None

    filtered_x = x_coords[mask]
    filtered_y = y_coords[mask]

    padding = 200
    center_x = int(np.mean(filtered_x))
    center_y = int(np.mean(filtered_y))
    width = max(400, int(np.max(filtered_x) - np.min(filtered_x))) + padding
    height = max(300, int(np.max(filtered_y) - np.min(filtered_y)))

    return center_x, center_y, width, height
