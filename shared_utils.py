
import os
import itertools
import requests
import shutil
import time
import asyncio
import aiohttp
import numpy as np
import math
import torch
from PIL import Image

# Initialize device for projection utils
device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')

# Constants
IMGX = 4
IMGY = 2

def _panoids_url(lat, lon):
    url = "https://maps.googleapis.com/maps/api/js/GeoPhotoService.SingleImageSearch?pb=!1m5!1sapiv3!5sUS!11m2!1m1!1b0!2m4!1m2!3d{0:}!4d{1:}!2d50!3m10!2m2!1sen!2sGB!9m1!1e2!11m4!1m3!1e2!2b1!3e2!4m10!1e1!1e2!1e3!1e4!1e8!1e6!5m1!1e2!6m1!1e2&callback=_xdc_._v2mub5"
    return url.format(lat, lon)

def tiles_info(panoid):
    # OPTIMIZED: Reduced zoom to 2 for native 2048px resolution (Fastest)
    image_url = "http://cbk0.google.com/cbk?output=tile&panoid={0:}&zoom=2&x={1:}&y={2:}"
    coord = list(itertools.product(range(IMGX), range(IMGY)))
    tiles = [(x, y, "%s_%dx%d.jpg" % (panoid, x, y), image_url.format(panoid, x, y)) for x, y in coord]
    return tiles

async def download_tile_aiohttp(session, x, y, fname, url):
    for attempt in range(2):
        try:
            async with session.get(url.replace("http://", "https://"), timeout=10) as response:
                if response.status == 200:
                    data = await response.read()
                    return x, y, data
        except Exception:
            await asyncio.sleep(2)
    return x, y, None

def download_tiles(tiles, status_callback=None, max_workers=64):
    total = len(tiles)
    results = {}

    async def main():
        connector = aiohttp.TCPConnector(limit=max_workers)
        async with aiohttp.ClientSession(connector=connector) as session:
            tasks = []
            for i, (x, y, fname, url) in enumerate(tiles):
                tasks.append(download_tile_aiohttp(session, x, y, fname, url))
            for idx, coro in enumerate(asyncio.as_completed(tasks), 1):
                x, y, data = await coro
                if data:
                    results[(x, y)] = data
                if status_callback:
                    status_callback(idx, total)

    asyncio.run(main())
    return results

def stitch_tiles(tiles_data):
    # [OFFICIAL SPEED OPTIMIZATION] Use NumPy block assignment for near-instant assembly
    # tiles_data: dict of {(x,y): binary_data}
    tile_w, tile_h = 512, 512
    import io
    
    # Pre-allocate large NumPy array
    pano_np = np.zeros((IMGY * tile_h, IMGX * tile_w, 3), dtype=np.uint8)
    
    for (x, y), data in tiles_data.items():
        try:
            tile = Image.open(io.BytesIO(data))
            tile_np = np.array(tile)
            # Assign block in NumPy (fast)
            th, tw, _ = tile_np.shape
            pano_np[y*tile_h:y*tile_h+th, x*tile_w:x*tile_w+tw] = tile_np
            tile.close()
        except Exception:
            continue
            
    return Image.fromarray(pano_np)

def pil_to_tensor(im):
    """LightGlue/DISK expect RGB [1,3,H,W] normalized to [0,1]"""
    return torch.from_numpy(np.array(im.convert('RGB'))).float().permute(2, 0, 1).unsqueeze(0).div(255.0).to(device)

def tensor_to_pil(t):
    """Converts a [1, C, H, W] device tensor to a PIL image."""
    t = t.squeeze(0).cpu().clamp(0, 1).mul(255).add_(0.5).to(torch.uint8).permute(1, 2, 0).numpy()
    if t.shape[2] == 1:
        t = t.squeeze(2) # [H, W, 1] -> [H, W] for grayscale
    return Image.fromarray(t)

def get_projection_base_dirs(fov_deg, out_hw):
    """Pre-calculates camera-space pixel directions."""
    fov = math.radians(fov_deg)
    out_h, out_w = out_hw
    cx, cy = out_w / 2.0, out_h / 2.0
    fx = fy = (out_w / 2.0) / math.tan(fov / 2.0)
    
    xx, yy = torch.meshgrid(
        torch.arange(out_w, device=device, dtype=torch.float32),
        torch.arange(out_h, device=device, dtype=torch.float32),
        indexing='xy'
    )
    x = (xx - cx) / fx
    y = (yy - cy) / fy
    z = torch.ones_like(x)
    
    dirs = torch.stack([x, -y, z], dim=-1)
    dirs = dirs / torch.norm(dirs, dim=-1, keepdim=True)
    return dirs.reshape(-1, 3).T # [3, N]

def equirectangular_to_rectilinear_torch(pano_tensor, fov_deg=90, out_hw=(400, 400), yaw_deg=0, pitch_deg=0, base_dirs=None):
    """
    GPU-accelerated projection using grid_sample. 
    Supports BATChED yaw_deg input (float, list, or tensor).
    """
    _, _, h, w = pano_tensor.shape
    out_h, out_w = out_hw
    
    # Handle scalar vs batch yaw
    if isinstance(yaw_deg, (float, int)):
        yaws = torch.tensor([yaw_deg], device=device, dtype=torch.float32)
    elif isinstance(yaw_deg, list):
        yaws = torch.tensor(yaw_deg, device=device, dtype=torch.float32)
    else: # Tensor
        yaws = yaw_deg.to(device).float()
        
    B = len(yaws)
    pitch = math.radians(-pitch_deg)

    # Convert yaws to radians
    yaws_rad = torch.deg2rad(yaws) # [B]

    # Pre-compute Rotation Matrices
    cos_vals = torch.cos(yaws_rad)
    sin_vals = torch.sin(yaws_rad)
    zeros = torch.zeros_like(cos_vals)
    ones = torch.ones_like(cos_vals)
    
    # R_yaw rows: [B, 3]
    row1 = torch.stack([cos_vals, zeros, sin_vals], dim=1)
    row2 = torch.stack([zeros, ones, zeros], dim=1)
    row3 = torch.stack([-sin_vals, zeros, cos_vals], dim=1)
    
    R = torch.stack([row1, row2, row3], dim=1) # [B, 3, 3]

    if base_dirs is None:
        base_dirs = get_projection_base_dirs(fov_deg, out_hw) # [3, N]
    
    # Batch Matrix Multiplication: [B, 3, 3] @ [1, 3, N] -> [B, 3, N]
    dirs = torch.matmul(R, base_dirs.unsqueeze(0))
    dirs = dirs.permute(0, 2, 1) # [B, N, 3]
    
    x = dirs[:, :, 0]
    y = dirs[:, :, 1]
    z = dirs[:, :, 2]
    
    lon = torch.atan2(x, z)
    lat = torch.asin(y.clamp(-1+1e-7, 1-1e-7))
    
    grid_x = lon / math.pi
    grid_y = -lat / (math.pi / 2.0)
    
    # [B, H, W, 2]
    grid = torch.stack([grid_x, grid_y], dim=-1).reshape(B, out_h, out_w, 2)
    
    # Expand pano to batch size [B, 3, H, W]
    pano_batch = pano_tensor.expand(B, -1, -1, -1)
    
    # Single Kernel Launch
    out = torch.nn.functional.grid_sample(pano_batch, grid, mode='bilinear', align_corners=True)
    return out

def equirectangular_to_rectilinear(pano_img, fov_deg=90, out_hw=(400, 400), yaw_deg=0, pitch_deg=0):
    """
    CPU wrapper for backward compatibility, but using the torch engine for speed.
    """
    # Convert PIL to tensor
    pano_tensor = pil_to_tensor(pano_img) # [1, 3, H, W]
    
    # Run torch projection
    out_tensor = equirectangular_to_rectilinear_torch(pano_tensor, fov_deg, out_hw, yaw_deg, pitch_deg)
    
    # Convert tensor back to PIL
    out_img = tensor_to_pil(out_tensor)
    return out_img
