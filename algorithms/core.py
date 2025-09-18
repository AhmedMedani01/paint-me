"""
Core animation algorithms for Hexa Paint Animation
Contains all the computer vision and animation processing functions
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
import shutil
import random
import skimage.morphology as morph
from skimage.measure import regionprops, label
import scipy.ndimage as sci_morph
from scipy.spatial import distance_matrix, distance
import pandas as pd
from tqdm import tqdm
from sklearn.cluster import KMeans
from skimage.segmentation import active_contour
import imageio
import matplotlib.pyplot as plt
from scipy.ndimage import label as cc_label
from skimage.morphology import dilation, skeletonize, rectangle
from scipy.spatial import cKDTree
from random import choice
import time
import threading


def group_colors(image, ngroups):
    """
    Get the actual boundary of the internal object using color k-means clustering segmentation.
    Performs image processing operations to get the boundary of the main object.
    
    Returns:
        list: Binary images used later in the code
    """
    image_1 = image
    image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    img2 = image.reshape((image.shape[0] * image.shape[1], -1))
    
    clt = KMeans(n_clusters=ngroups)
    clt.fit(img2)
    
    clu_final = np.zeros((image.shape[0], image.shape[1]))
    output_img = []
    for i in range(ngroups):
        clu_img = np.zeros((img2.shape[0]))
        clu_img[clt.labels_ == i] = 1
        clu_img = clu_img.reshape((image.shape[0], image.shape[1], 1))
        clu_img = np.concatenate((clu_img, clu_img, clu_img), 2)
        output_img.append(clu_img)
    
    return output_img


def get_points_inds(in_img):
    """
    Continuous DFS ordering of 1-pixels.
    
    Parameters:
    -----------
    in_img : 2-D ndarray, {0,1}
        Binary image whose foreground (value==1) pixels need indexing.
    
    Returns:
    --------
    out_img : 2-D ndarray, uint32
        Background is 0. Foreground pixels hold 1…N in the exact
        order a virtual "pen" should visit them; each connected
        component is completed before the next one begins.
    """
    h, w = in_img.shape
    out_img = np.zeros((h, w), dtype=np.uint32)
    
    # Connected-components labelling (8-neighbour)
    lbl, n_comp = cc_label(in_img, structure=np.ones((3, 3), dtype=np.uint8))
    if n_comp == 0:
        return out_img
    
    # Pre-compute neighbour offsets for 8-connectivity
    neigh = np.array([
        (-1, -1), (-1, 0), (-1, 1),
        ( 0, -1),          ( 0, 1),
        ( 1, -1), ( 1, 0), ( 1, 1),
    ], dtype=np.int16)
    
    rank = 1
    visited = np.zeros_like(in_img, dtype=bool)
    
    # Process each component separately
    for comp_id in range(1, n_comp + 1):
        xs, ys = np.where(lbl == comp_id)
        if xs.size == 0:
            continue
        
        # Pick the lexicographically smallest pixel as start (top-left-most)
        start_idx = xs.argmin() if xs.size == 1 else np.lexsort((ys, xs))[0]
        stack = [(xs[start_idx], ys[start_idx])]
        
        while stack:
            x, y = stack.pop()
            if visited[x, y]:
                continue
            
            visited[x, y] = True
            out_img[x, y] = rank
            rank += 1
            
            # Push unvisited 8-neighbours belonging to the same component
            for dx, dy in neigh:
                nx, ny = x + dx, y + dy
                if (
                    0 <= nx < h and 0 <= ny < w
                    and not visited[nx, ny]
                    and lbl[nx, ny] == comp_id
                ):
                    stack.append((nx, ny))
    
    return out_img


def index_edges_by_hulls(
    edge_img: np.ndarray,
    hull_img: np.ndarray,
    *,
    reverse_hull_order: bool = False,
    reverse_within_hull: bool = False,
) -> np.ndarray:
    """
    Index edges by convex hulls with customizable ordering.
    
    Parameters:
    -----------
    edge_img : np.ndarray
        Binary edge image
    hull_img : np.ndarray
        Binary convex hull image
    reverse_hull_order : bool, default False
        Process convex-hull components bottom-to-top if True
    reverse_within_hull : bool, default False
        Invert indices INSIDE each hull so drawing runs backward
    
    Returns:
    --------
    out_img : np.ndarray
        Indexed edge image
    """
    h, w = edge_img.shape
    out_img = np.zeros((h, w), dtype=np.uint32)
    current_rank = 1
    
    # Label convex-hull segments
    lbl, n_comp = ndi.label(hull_img, structure=np.ones((3, 3), np.uint8))
    
    # Sort hull IDs by topmost row (ascending) then flip if requested
    seg_ids = []
    for cid in range(1, n_comp + 1):
        ys = np.where(lbl == cid)[0]
        if ys.size:
            seg_ids.append((ys.min(), cid))
    seg_ids.sort(key=lambda x: x[0], reverse=reverse_hull_order)
    seg_ids = [cid for _, cid in seg_ids]
    
    # Process each hull
    for cid in seg_ids:
        seg_mask = (lbl == cid)
        seg_edge = edge_img & seg_mask
        if not seg_edge.any():
            continue
        
        seg_idx = get_points_inds(seg_edge)
        if seg_idx.max() == 0:
            continue
        
        if reverse_within_hull:
            max_v = seg_idx.max()
            seg_idx[seg_idx > 0] = max_v + 1 - seg_idx[seg_idx > 0]
        
        seg_idx[seg_idx > 0] += current_rank - 1
        out_img[seg_idx > 0] = seg_idx[seg_idx > 0]
        current_rank = out_img.max() + 1
    
    # Background edges (outside all hulls)
    bg_edge = edge_img & (~hull_img.astype(bool))
    bg_idx = get_points_inds(bg_edge)
    if bg_idx.max() > 0:
        if reverse_within_hull:
            max_v = bg_idx.max()
            bg_idx[bg_idx > 0] = max_v + 1 - bg_idx[bg_idx > 0]
        
        bg_idx[bg_idx > 0] += current_rank - 1
        out_img[bg_idx > 0] = bg_idx[bg_idx > 0]
    
    return out_img


def load_and_resize_image(filename: str, max_size: int = 512):
    """
    Load an image, resize it to max_size width while maintaining aspect ratio,
    and ensure both spatial dimensions are even for FFmpeg compatibility.
    
    Parameters:
    -----------
    filename : str
        Path to the image file
    max_size : int
        Maximum width size (height will be calculated to maintain aspect ratio)
    
    Returns:
    --------
    image : ndarray
        RGB image with even dimensions, resized to max_size width
    fn : str
        Original file name + extension
    """
    fn_base, ext = os.path.splitext(filename)
    fn = os.path.basename(fn_base) + ext
    
    img_bgr = cv2.imread(filename, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(filename)
    
    image = cv2.imread(filename, cv2.COLOR_BGR2RGB)
    
    # Get original dimensions
    h, w = image.shape[:2]
    
    # Calculate new dimensions maintaining aspect ratio
    if w > max_size:
        # Resize based on width
        new_w = max_size
        new_h = int((h * max_size) / w)
    else:
        # Keep original size if smaller than max_size
        new_w = w
        new_h = h
    
    # Resize the image
    if new_w != w or new_h != h:
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Ensure height is even (round up to next even number for FFmpeg compatibility)
    h, w = image.shape[:2]
    if h % 2 != 0:
        # Round height to next even number
        new_h = h + 1
        # Resize to new height while maintaining width
        image = cv2.resize(image, (w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Ensure width is even (round up to next even number for FFmpeg compatibility)
    h, w = image.shape[:2]
    if w % 2 != 0:
        # Round width to next even number
        new_w = w + 1
        # Resize to new width while maintaining height
        image = cv2.resize(image, (new_w, h), interpolation=cv2.INTER_LINEAR)
    
    return image, fn


def detect_edges_thick(
    rgb: np.ndarray,
    *,
    dilate_iters: int = 1,
    kernel_size: int = 3
) -> np.ndarray:
    """
    Detect thick edges in an RGB image.
    
    Parameters:
    -----------
    rgb : np.ndarray
        Input RGB image
    dilate_iters : int
        Number of dilation iterations
    kernel_size : int
        Kernel size for morphological operations
    
    Returns:
    --------
    edges : np.ndarray
        Binary edge image
    """
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.createCLAHE(2.0, (8, 8)).apply(gray)
    gray = cv2.bilateralFilter(gray, 7, 30, 7)
    
    med = np.median(gray)
    lo, hi = int(max(0, 0.66 * med)), int(min(255, 1.33 * med))
    edges = cv2.Canny(gray, lo, hi, L2gradient=True)
    
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    
    if dilate_iters > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        edges = cv2.dilate(edges, k, iterations=dilate_iters)
    
    return (edges > 0).astype(np.uint8)


def create_binary_mask(
    image: np.ndarray,
    *,
    gray_path: str = "",
    dilate_iters: int = 1,
    kernel_size: int = 3,
    min_size: int = 1500
):
    """
    Create binary mask from image.
    
    Parameters:
    -----------
    image : np.ndarray
        Input image
    gray_path : str
        Path to gray image (unused)
    dilate_iters : int
        Dilation iterations
    kernel_size : int
        Kernel size
    min_size : int
        Minimum object size
    
    Returns:
    --------
    bin_img : np.ndarray
        Binary edge image
    hull_image : np.ndarray
        Convex hull image
    """
    bin_img = detect_edges_thick(
        cv2.cvtColor(image, cv2.COLOR_BGR2RGB), 
        dilate_iters=dilate_iters,
        kernel_size=kernel_size
    )
    
    mask_closed = ndi.binary_fill_holes(
        morph.binary_closing(bin_img, morph.disk(3))
    )
    mask_no_small = morph.remove_small_objects(mask_closed, min_size=min_size)
    
    lbl_img = label(mask_no_small)
    hull_image = np.zeros_like(mask_no_small, bool)
    for lab in np.unique(lbl_img)[1:]:
        hull_image |= morph.convex_hull_object(lbl_img == lab)
    
    return bin_img, hull_image


def make_img_objs(
    gray_img: np.ndarray,
    *,
    line_size: int = 11,
    blur_value: int = 3,
    min_size: int = 25,
    closing_size: int = 27,
) -> np.ndarray:
    """
    Return the thick, gap-free binary object mask used by paint_image.
    
    Parameters:
    -----------
    gray_img : 2-D ndarray, uint8
        Input grayscale image
    line_size : int
        Block size for adaptive threshold
    blur_value : int
        Kernel for median blur
    min_size : int
        Remove objects smaller than this
    closing_size : int
        Diameter of disk footprint for final binary closing
    
    Returns:
    --------
    img_objs : uint8 {0,1}
        Binary object mask
    """
    # Enforce odd kernel sizes ≥3
    line_size = max(3, line_size | 1)
    blur_value = max(3, blur_value | 1)
    closing_size = max(3, closing_size | 1)
    
    # Noise suppression
    gray_blur = cv2.medianBlur(gray_img, blur_value)
    
    # Adaptive threshold (white background)
    edges = cv2.adaptiveThreshold(
        gray_blur, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        line_size, blur_value
    )
    
    # Invert & remove tiny specks
    obj_mask = morph.remove_small_objects((~edges).astype(bool), min_size=min_size)
    
    # Morphological closing to seal gaps
    radius = closing_size // 2
    obj_mask = morph.binary_closing(obj_mask, morph.disk(radius))
    
    return obj_mask.astype(np.uint8)


def setup_video_writer(output_path, width, height):
    """
    Setup video writer for output.
    
    Parameters:
    -----------
    output_path : str
        Path to output video file
    width : int
        Video width
    height : int
        Video height
    
    Returns:
    --------
    video_writer : imageio writer
        Video writer object
    """
    return imageio.get_writer(
        output_path,
        format="ffmpeg",
        fps=30,
        codec="libx264",
        pixelformat="yuv420p",
        macro_block_size=1,
    )


def create_mask_rgb(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Return a white-background version of image that keeps the true pixel
    values only where mask == 1.
    
    Parameters:
    -----------
    image : ndarray, RGB
        The original image (H×W×3)
    mask : ndarray, bool / {0,1}
        Binary foreground mask – same H×W size as image
    
    Returns:
    --------
    mask_rgb : ndarray, RGB
        White everywhere except at foreground-mask locations
    """
    mask3 = mask.astype(bool)[:, :, None]
    return np.where(mask3, image, 255)


def overlay_rgba(canvas: np.ndarray, sprite: np.ndarray, y: int, x: int) -> None:
    """
    Alpha-blend sprite so its top-left corner touches (y, x) in canvas.
    Ideal for pen tip at the sprite's corner.
    
    Parameters:
    -----------
    canvas : np.ndarray
        Canvas to draw on
    sprite : np.ndarray
        Sprite to overlay
    y : int
        Y coordinate
    x : int
        X coordinate
    """
    h, w = sprite.shape[:2]
    y1, x1 = y, x
    y2, x2 = y1 + h, x1 + w
    
    cy1, cx1 = max(0, y1), max(0, x1)
    cy2, cx2 = min(canvas.shape[0], y2), min(canvas.shape[1], x2)
    if cy2 <= cy1 or cx2 <= cx1:
        return
    
    sy1, sx1 = cy1 - y1, cx1 - x1
    spr = sprite[sy1:sy1 + (cy2 - cy1), sx1:sx1 + (cx2 - cx1)]
    roi = canvas[cy1:cy2, cx1:cx2]
    
    if spr.shape[2] == 4:
        alpha = spr[..., 3:4].astype(float) / 255.0
        rgb = spr[..., :3]
    else:
        alpha = 1.0
        rgb = spr
    
    canvas[cy1:cy2, cx1:cx2] = (roi * (1 - alpha) + rgb * alpha).astype(canvas.dtype)


def prepare_sprite(
    path: str,
    base_shape: tuple[int, int],
    *,
    scale_frac: float = 2 / 3,
    tol: int = 8
) -> np.ndarray:
    """
    Load an image (pen/brush), make its uniform background transparent,
    and resize it so the longer side equals scale_frac × the corresponding
    side of the animation image.
    
    Parameters:
    -----------
    path : str
        File path of the sprite (PNG/JPEG)
    base_shape : (H, W)
        Shape of the animation frame
    scale_frac : float
        Fraction of the frame dimension to occupy
    tol : int
        Background-detection tolerance
    
    Returns:
    --------
    sprite_rgba : (h, w, 4) uint8
        RGBA sprite resized with aspect ratio preserved
    """
    if not os.path.exists(path):
        # Create a simple default sprite if the file doesn't exist
        h, w = base_shape
        sprite_size = int(min(h, w) * scale_frac)
        sprite = np.ones((sprite_size, sprite_size, 4), dtype=np.uint8) * 255
        sprite[:, :, 3] = 128  # Semi-transparent
        return sprite
    
    img = cv2.imread(path, cv2.COLOR_BGR2RGB)
    
    # Detect background colour
    if img.shape[2] == 4:
        rgb = img[..., :3]
        alpha_existing = img[..., 3]
        rgb_flat = rgb[alpha_existing > 0].reshape(-1, 3)
    else:
        rgb = img
        rgb_flat = rgb.reshape(-1, 3)
    
    if rgb_flat.size > 0:
        colours, counts = np.unique(rgb_flat, axis=0, return_counts=True)
        bg_colour = colours[counts.argmax()]
        bg_mask = (np.abs(rgb - bg_colour.reshape(1, 1, 3)) <= tol).all(2)
        alpha = np.where(bg_mask, 0, 255).astype(np.uint8)
    else:
        alpha = np.ones(rgb.shape[:2], dtype=np.uint8) * 255
    
    sprite_rgba = np.dstack([rgb, alpha])
    
    # Resize to scale_frac of frame while keeping aspect
    Hf, Wf = base_shape
    h0, w0 = sprite_rgba.shape[:2]
    
    max_w = int(Wf * scale_frac)
    max_h = int(Hf * scale_frac)
    scale = min(max_w / w0, max_h / h0)
    
    new_size = (max(1, int(w0 * scale)), max(1, int(h0 * scale)))
    sprite_rgba = cv2.resize(sprite_rgba, new_size, interpolation=cv2.INTER_AREA)
    
    return sprite_rgba


def animate_binary_mask(
    mask_rgb: np.ndarray,
    edge_map: np.ndarray,
    hull_img: np.ndarray,
    video_writer,
    *,
    pen_png,
    skip: int = 10,
    reverse: bool = True,
):
    """
    Coloured-edge sketch animation with a MOVING pen sprite
    that disappears once the sketch is done.
    
    Parameters:
    -----------
    mask_rgb : np.ndarray
        RGB mask image
    edge_map : np.ndarray
        Edge map
    hull_img : np.ndarray
        Convex hull image
    video_writer : imageio writer
        Video writer object
    pen_png : np.ndarray
        Pen sprite
    skip : int
        Skip frames
    reverse : bool
        Reverse animation order
    
    Returns:
    --------
    canvas : np.ndarray
        Final canvas
    """
    # Prepare white canvas & traversal order
    canvas = np.full_like(mask_rgb, 255, dtype=np.uint8)
    
    mult = index_edges_by_hulls(edge_map, hull_img, reverse_hull_order=True, reverse_within_hull=False)
    if mult.max() == 0:
        video_writer.append_data(cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
        return canvas
    
    inds = np.unique(mult)[1:]  # skip background 0
    if reverse:
        inds = inds[::-1]
    
    prev_drawn = np.zeros_like(mult, bool)
    
    # Send initial blank frame
    video_writer.append_data(cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    
    # Reveal batches & overlay pen
    for start in range(0, len(inds), skip):
        lo = inds[start]
        hi = inds[min(start + skip, len(inds)) - 1]
        if reverse:
            batch = (mult >= hi) & (mult <= lo) & (~prev_drawn)
        else:
            batch = (mult >= lo) & (mult <= hi) & (~prev_drawn)
        
        if not batch.any():
            continue
        
        prev_drawn |= batch
        canvas[batch] = mask_rgb[batch]
        
        # Create a TEMPORARY frame to show the pen
        frame = canvas.copy()
        y_tip, x_tip = np.array(np.nonzero(batch))[:, -1]  # newest pixel
        overlay_rgba(frame, pen_png, int(y_tip), int(x_tip))
        
        video_writer.append_data(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    
    # Hold final sketch WITHOUT the pen
    for _ in range(8):
        video_writer.append_data(cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    
    return canvas


def hexgrid_mesh_indices_2d(
    image_shape,
    region_mask,
    *,
    hex_radius=20,
    jitter=0.15,
    rng_seed=None,
):
    """
    Return two maps:
        idx_col : integer column index of each hex tile
        idx_row : integer row index of each hex tile
    
    Pixels outside region_mask are zero in both maps.
    
    Parameters:
    -----------
    image_shape : tuple
        Shape of the image
    region_mask : np.ndarray
        Binary region mask
    hex_radius : int
        Hexagon radius
    jitter : float
        Jitter amount
    rng_seed : int
        Random seed
    
    Returns:
    --------
    idx_col : np.ndarray
        Column indices
    idx_row : np.ndarray
        Row indices
    """
    H, W = image_shape
    idx_col = np.zeros((H, W), dtype=np.uint32)
    idx_row = np.zeros((H, W), dtype=np.uint32)
    if not region_mask.any():
        return idx_col, idx_row
    
    rng = np.random.default_rng(rng_seed)
    
    # Hex-grid constants
    h_step = hex_radius * 3 / 2
    w_step = np.sqrt(3) * hex_radius
    
    # Generate centres with integer (row, col) tags
    centres = []
    row_ids = []
    col_ids = []
    row, y = 0, 0.0
    while y < H:
        x_off = 0 if row % 2 == 0 else w_step / 2
        col, x = 0, x_off
        while x < W:
            centres.append((y, x))
            row_ids.append(row + 1)
            col_ids.append(col + 1)
            col += 1
            x += w_step
        row += 1
        y += h_step
    
    centres = np.asarray(centres, dtype=np.float32)
    centres += rng.uniform(-jitter, jitter, centres.shape) * hex_radius * 2
    centres = np.clip(centres, [[0, 0]], [[H - 1, W - 1]])
    row_ids = np.asarray(row_ids, dtype=np.uint32)
    col_ids = np.asarray(col_ids, dtype=np.uint32)
    
    # Assign each region pixel to nearest centre
    ys, xs = np.where(region_mask)
    pts = np.stack((ys, xs), 1)
    d2 = ((pts[:, None, :] - centres[None, :, :]) ** 2).sum(2)
    nearest = d2.argmin(1)
    
    # Tile orderings
    order_x = np.argsort(centres[:, 1])  # left → right
    order_y = np.argsort(centres[:, 0])  # top → bottom
    rank_x = {tile: r + 1 for r, tile in enumerate(order_x)}
    rank_y = {tile: r + 1 for r, tile in enumerate(order_y)}
    
    idx_col[ys, xs] = np.vectorize(rank_x.get)(nearest)
    idx_row[ys, xs] = np.vectorize(rank_y.get)(nearest)
    
    # Extra masking to ensure NO stray indices outside this colour band
    idx_col *= region_mask
    idx_row *= region_mask
    
    return idx_col, idx_row


def random_path_order(tile_ids: np.ndarray, centres: np.ndarray, *, rng: np.random.Generator, k_neigh: int = 6) -> list[int]:
    """
    Generate random path order for tiles.
    
    Parameters:
    -----------
    tile_ids : np.ndarray
        Tile IDs
    centres : np.ndarray
        Tile centres
    rng : np.random.Generator
        Random number generator
    k_neigh : int
        Number of neighbors to consider
    
    Returns:
    --------
    order : list
        Ordered tile IDs
    """
    tree = cKDTree(centres)
    id2idx = {int(t): i for i, t in enumerate(tile_ids)}
    unvisited = {int(t) for t in tile_ids}
    order = []
    
    while unvisited:
        rows = np.array([centres[id2idx[t], 0] for t in unvisited])
        start_ids = [t for t in unvisited if centres[id2idx[t], 0] == rows.max()]
        current = rng.choice(start_ids)
        
        while True:
            order.append(current)
            unvisited.discard(current)
            if not unvisited:
                break
            
            centre = centres[id2idx[current]]
            _, neigh_idx = tree.query(
                centre, k=min(k_neigh + 1, len(tile_ids)), workers=-1
            )
            
            next_tile = None
            for ni in neigh_idx[1:]:
                if ni >= tile_ids.size:
                    continue
                cand = int(tile_ids[ni])
                if cand in unvisited:
                    next_tile = cand
                    break
            if next_tile is None:
                break
            current = next_tile
    
    return order


def animate_colors_kmeans_random_paths(
    image,
    base_canvas,
    video_writer,
    *,
    brush_png,
    ngroups: int = 3,
    hex_radius: int = 30,
    jitter: float = 0.1,
    rng_seed=None,
):
    """
    Colour-fill animation that follows random tile paths and shows a
    moving brush sprite that disappears when colouring is complete.
    
    Parameters:
    -----------
    image : np.ndarray
        Input image
    base_canvas : np.ndarray
        Base canvas
    video_writer : imageio writer
        Video writer object
    brush_png : np.ndarray
        Brush sprite
    ngroups : int
        Number of color groups
    hex_radius : int
        Hexagon radius
    jitter : float
        Jitter amount
    rng_seed : int
        Random seed
    
    Returns:
    --------
    colour_canvas : np.ndarray
        Final colored canvas
    """
    rng = np.random.default_rng(rng_seed)
    colour_canvas = base_canvas.copy()
    H, W = image.shape[:2]
    
    # K-means colour masks
    cluster_masks = [
        (clus[:, :, 0] > 0) for clus in group_colors(image, ngroups)
    ]
    
    for cl_mask in cluster_masks:
        idx_col, idx_row = hexgrid_mesh_indices_2d(
            image_shape=(H, W),
            region_mask=cl_mask.astype(bool),
            hex_radius=hex_radius,
            jitter=jitter,
            rng_seed=rng_seed,
        )
        idx_col = idx_col.astype(np.int64)
        idx_row = idx_row.astype(np.int64)
        num_cols = idx_col.max()
        tile_id_map = (idx_row - 1) * num_cols + idx_col
        
        tile_ids_all = np.unique(tile_id_map)[1:]  # skip 0
        tiles = {int(t): np.where(tile_id_map == t) for t in tile_ids_all}
        
        tile_ids = np.unique(tile_id_map[cl_mask])
        tile_ids = tile_ids[tile_ids > 0].astype(np.int64)
        if tile_ids.size == 0:
            continue
        
        # Centres for this cluster (row/col int grid)
        centres = np.stack(
            (((tile_ids - 1) // num_cols) + 1,
             ((tile_ids - 1) % num_cols) + 1),
            axis=1
        ).astype(np.float32)
        
        path = random_path_order(tile_ids, centres, rng=rng)
        
        for tid in path:
            pts = tiles.get(int(tid))
            if pts is None or pts[0].size == 0:
                continue
            
            # Paint tile onto persistent canvas
            colour_canvas[pts] = image[pts]
            
            # Make a temp frame with brush overlay
            frame = colour_canvas.copy()
            y_br, x_br = pts[0][-1], pts[1][-1]  # newest pixel
            overlay_rgba(frame, brush_png, int(y_br), int(x_br))
            
            video_writer.append_data(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    
    # Hold final coloured frame WITHOUT the brush
    for _ in range(32):
        video_writer.append_data(cv2.cvtColor(colour_canvas, cv2.COLOR_RGB2BGR))
    
    return colour_canvas


def paint_image(img_path, video_path):
    """
    Main animation function that processes an image and creates an animated video.
    
    Parameters:
    -----------
    img_path : str
        Path to input image
    video_path : str
        Path to output video
    
    Returns:
    --------
    bool
        True if successful, False otherwise
    """
    try:
        image, _ = load_and_resize_image(img_path)
        h, w = image.shape[:2]
        
        # Create default sprites if hands folder doesn't exist
        pen_png = prepare_sprite('hands/pen1.png', base_shape=(h, w), scale_frac=1.5)
        brush_png = prepare_sprite('hands/brush1.png', base_shape=(h, w), scale_frac=1.5)
        
        _, hull_img = create_binary_mask(
            image,
            dilate_iters=1,
            kernel_size=3,
            min_size=1500
        )
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img_objs = make_img_objs(gray, closing_size=5)
        
        mask_rgb = create_mask_rgb(image, img_objs)
        video_writer = setup_video_writer(video_path, w, h)
        
        print("Animating binary mask")
        I_edge = animate_binary_mask(
            mask_rgb=mask_rgb,
            edge_map=img_objs,
            hull_img=hull_img,
            video_writer=video_writer,
            pen_png=pen_png,
            skip=100,
            reverse=False,
        )
        
        print('Animating colors')
        I_rgb = animate_colors_kmeans_random_paths(
            image=image,
            base_canvas=mask_rgb,
            video_writer=video_writer,
            brush_png=brush_png,
            ngroups=4,
            hex_radius=30,
            jitter=0.3, 
            rng_seed=42,
        )
        
        video_writer.close()
        return True
    except Exception as e:
        print(f"Error in paint_image: {e}")
        return False
