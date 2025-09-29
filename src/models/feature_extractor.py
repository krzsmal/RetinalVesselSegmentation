"""Feature extraction module for vessel segmentation."""

import numpy as np
import cv2
import hashlib
from typing import Dict, Any, Tuple, List
from pathlib import Path
from skimage.filters import frangi, sato
from scipy.ndimage import gaussian_filter

from src.utils import get_logger
logger = get_logger(__name__)

from src.preprocessing.image_processor import ImageProcessor


class FeatureExtractor:
    """Feature extraction for vessel segmentation."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize feature extractor."""
        
        # Classifier-specific config
        self.clf_config = config.get('classifier', {})
        self.random_state = int(self.clf_config.get('random_state', 42))
        self.neg_sample_rate = float(self.clf_config.get('neg_sample_rate', 0.10))
        self.pos_sample_rate = float(self.clf_config.get('pos_sample_rate', 1.0))
        self.max_samples_per_image = int(self.clf_config.get('max_samples_per_image', 1_000_000))
        self.resize_factor = int(self.clf_config.get('resize_factor', 2))
        self.denoise_kernel = int(self.clf_config.get('denoise_kernel', 5))
        
        # Feature extraction
        fcfg = self.clf_config.get('features', {})
        self.cache_enabled = bool(fcfg.get('use_cache', True))
        self.cache_dir = Path(fcfg.get('cache_dir', '.cache/features'))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.window_size = int(fcfg.get('window_size', 5))
        self.frangi_sigmas = fcfg.get('frangi_sigmas', [1.0, 2.0, 3.0])
        self.sato_sigmas = fcfg.get('sato_sigmas', [1.0, 2.0, 3.0])
        self.st_sigma_grad = float(fcfg.get('st_sigma_grad', 1.0))
        self.st_sigma_smooth = float(fcfg.get('st_sigma_smooth', 2.0))
        self.log_sigma = float(fcfg.get('log_sigma', 1.2))
        self.canny_low = float(fcfg.get('canny_low', 0.05))
        self.canny_high = float(fcfg.get('canny_high', 0.15))
        self.hess_sigma = float(fcfg.get('hessian_sigma', 1.0))
        self.blackhat_rs = fcfg.get('blackhat_rs', [3, 5, 7])
        self.median_k = int(fcfg.get('median_k', 5))
        self.gabor_thetas = [0.0, np.pi/4, np.pi/2, 3*np.pi/4] if 'gabor_thetas' not in fcfg else fcfg['gabor_thetas']
        self.gabor_lambdas = [4.0, 8.0] if 'gabor_lambdas' not in fcfg else fcfg['gabor_lambdas']
        self.gabor_gamma = float(fcfg.get('gabor_gamma', 0.5))
        self.gabor_psi = float(fcfg.get('gabor_psi', 0.0))

        self.processor = ImageProcessor()
        self.half_window = self.window_size // 2

        logger.info(f"Initalized FeatureExtractor")

    def _cache_key_common(self) -> str:
        """Common part of cache key based on feature extraction settings."""
        
        return (
            f"w{self.window_size}_rf{self.resize_factor}_dk{self.denoise_kernel}"
            f"_grad1_color1_vess1_pos{self.pos_sample_rate}_neg{self.neg_sample_rate}"
            f"_max{self.max_samples_per_image}_rs{self.random_state}"
        )

    def _precompute_feature_maps(self, image: np.ndarray):
        """Precompute feature maps for the entire image."""

        color_denoised, green = self.preprocess(image)

        # green channel normalization
        g = green.astype(np.float32)
        if g.max() > 1.0 or g.min() < 0.0:
            g = (g - g.min()) / (g.max() - g.min() + 1e-8)

        # gradient magnitude (Sobel)
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = cv2.magnitude(gx, gy)

        # multi-scale vesselness (Frangi, Sato)
        from skimage.filters import frangi, sato
        frangi_sigmas = getattr(self, "frangi_sigmas", [1.0, 2.0, 3.0])
        sato_sigmas = getattr(self, "sato_sigmas",   [1.0, 2.0, 3.0])

        frangi_maps = [frangi(g, sigmas=[float(s)]).astype(np.float32) for s in frangi_sigmas]
        sato_maps = [sato(g, sigmas=[float(s)]).astype(np.float32) for s in sato_sigmas]
        frangi_maps2 = [m*m for m in frangi_maps]
        sato_maps2 = [m*m for m in sato_maps]

        # Gabor filters
        gabor_thetas = getattr(self, "gabor_thetas",  [0.0, np.pi/4, np.pi/2, 3*np.pi/4])
        gabor_lambdas = getattr(self, "gabor_lambdas", [4.0, 8.0])
        gabor_gamma = float(getattr(self, "gabor_gamma", 0.5))
        gabor_psi = float(getattr(self, "gabor_psi",   0.0))
        gabor_list = []
        for lam in gabor_lambdas:
            sigma = 0.56 * float(lam)
            ksize = int(max(7, (int(round(6*sigma)) | 1)))
            for th in gabor_thetas:
                kern = cv2.getGaborKernel((ksize, ksize), sigma, float(th), float(lam), gabor_gamma, gabor_psi, ktype=cv2.CV_32F)
                resp = cv2.filter2D(g, cv2.CV_32F, kern)
                resp = cv2.absdiff(resp, 0)
                gabor_list.append(resp.astype(np.float32))
        gabor2_list = [m*m for m in gabor_list]

        # Hessian eigenvalues and orientation
        hess_sigma = float(getattr(self, "hess_sigma", 1.0))
        kH = int(max(1, round(hess_sigma*3))*2+1)
        gH = cv2.GaussianBlur(g, (kH, kH), hess_sigma)
        Lxx = gaussian_filter(g, sigma=hess_sigma, order=(2,0))
        Lyy = gaussian_filter(g, sigma=hess_sigma, order=(0,2))
        Lxy = gaussian_filter(g, sigma=hess_sigma, order=(1,1))
        tr = 0.5*(Lxx + Lyy)
        df = 0.5*(Lxx - Lyy)
        rad = cv2.sqrt(df*df + Lxy*Lxy)
        lam1 = tr + rad
        lam2 = tr - rad
        theta = 0.5 * np.arctan2(2.0*Lxy, (Lxx - Lyy + 1e-8)).astype(np.float32)
        cos_t = np.cos(theta).astype(np.float32)
        sin_t = np.sin(theta).astype(np.float32)

        # Black-hat (multi-radius)
        bh_rs = getattr(self, "blackhat_rs", [3, 5, 7])
        bh_list = []
        for r in bh_rs:
            k = 2*int(r)+1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            bh = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, kernel).astype(np.float32)
            bh_list.append(bh)
        bh2_list = [m*m for m in bh_list]

        # MAD (mean absolute deviation)
        median_k = int(getattr(self, "median_k", 5)) | 1
        med = cv2.medianBlur((g*255).astype(np.uint8), max(3, median_k)).astype(np.float32) / 255.0
        mad = cv2.absdiff(g, med).astype(np.float32)
        mad2 = mad*mad

        # final maps
        col = color_denoised.astype(np.float32)
        col2 = col * col

        return {
            "color": col,  "color2": col2,
            "grad":  grad_mag, "grad2": grad_mag*grad_mag,
            "frangi_list": frangi_maps, "frangi2_list": frangi_maps2,
            "sato_list":   sato_maps,   "sato2_list":   sato_maps2,
            "gabor_list":  gabor_list,  "gabor2_list":  gabor2_list,
            "hess_l1": lam1, "hess_l2": lam2,
            "cos_t": cos_t,  "sin_t": sin_t,
            "bh_list": bh_list, "bh2_list": bh2_list,
            "mad": mad, "mad2": mad2,
            "green": g
        }

    def _local_mean(self, img: np.ndarray, k: int) -> np.ndarray:
        """Calculate local mean using box filter."""

        return cv2.boxFilter(img, ddepth=-1, ksize=(k, k), normalize=True, borderType=cv2.BORDER_REFLECT101)

    def _local_std(self, img: np.ndarray, img2: np.ndarray, k: int) -> np.ndarray:
        """Calculate local standard deviation using box filter."""

        m  = self._local_mean(img, k)
        m2 = self._local_mean(img2, k)
        v = cv2.max(m2 - m*m, 0)
        return cv2.sqrt(v)

    def _local_max(self, img: np.ndarray, k: int) -> np.ndarray:
        """Calculate local maximum using dilation."""

        return cv2.dilate(img, kernel=cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))

    def _build_feature_stack(self, fmap: dict) -> Tuple[np.ndarray, List[Tuple[int,int]]]:
        """ Build feature stack from precomputed feature maps."""

        k = int(self.window_size)
        half = self.half_window

        # color: mean/std for each channel
        col, col2 = fmap["color"], fmap["color2"]
        mu_r = self._local_mean(col[...,0], k); sd_r = self._local_std(col[...,0], col2[...,0], k)
        mu_g = self._local_mean(col[...,1], k); sd_g = self._local_std(col[...,1], col2[...,1], k)
        mu_b = self._local_mean(col[...,2], k); sd_b = self._local_std(col[...,2], col2[...,2], k)

        # gradient magnitude: mean/std
        mu_grad = self._local_mean(fmap["grad"], k)
        sd_grad = self._local_std (fmap["grad"], fmap["grad2"], k)

        # vesselness (Frangi)
        mu_fr_list, sd_fr_list, mx_fr_list = [], [], []
        for m, m2 in zip(fmap["frangi_list"], fmap["frangi2_list"]):
            mu_fr_list.append(self._local_mean(m, k))
            sd_fr_list.append(self._local_std (m, m2, k))
            mx_fr_list.append(self._local_max (m, k))

        # vesselness (Sato)
        mu_sa_list, sd_sa_list, mx_sa_list = [], [], []
        for m, m2 in zip(fmap["sato_list"], fmap["sato2_list"]):
            mu_sa_list.append(self._local_mean(m, k))
            sd_sa_list.append(self._local_std (m, m2, k))
            mx_sa_list.append(self._local_max (m, k))

        # Gabor: mean/std for each filter
        gabor_mu, gabor_sd = [], []
        for m, m2 in zip(fmap["gabor_list"], fmap["gabor2_list"]):
            gabor_mu.append(self._local_mean(m, k))
            gabor_sd.append(self._local_std (m, m2, k))

        # Hessian eigenvalues and orientation: mean/std
        l1 = cv2.absdiff(fmap["hess_l1"], 0)
        l2 = cv2.absdiff(fmap["hess_l2"], 0)
        mu_l1 = self._local_mean(l1, k); sd_l1 = self._local_std(l1, l1*l1, k)
        mu_l2 = self._local_mean(l2, k); sd_l2 = self._local_std(l2, l2*l2, k)
        mu_cost = self._local_mean(fmap["cos_t"], k)
        mu_sint = self._local_mean(fmap["sin_t"], k)

        # Black-hat: mean/std/max for each radius
        bh_mu, bh_sd, bh_mx = [], [], []
        for m, m2 in zip(fmap["bh_list"], fmap["bh2_list"]):
            bh_mu.append(self._local_mean(m, k))
            bh_sd.append(self._local_std (m, m2, k))
            bh_mx.append(self._local_max (m, k))

        # MAD: mean/std
        mu_mad = self._local_mean(fmap["mad"], k)
        sd_mad = self._local_std (fmap["mad"], fmap["mad2"], k)

        # crop to valid region
        def crop(a: np.ndarray) -> np.ndarray:
            return a[half:-half, half:-half]

        maps = [
            # color
            crop(sd_r), crop(sd_g), crop(sd_b),
            crop(mu_r), crop(mu_g), crop(mu_b),
            # gradient
            crop(mu_grad), crop(sd_grad),
            # Hessian and orientation
            crop(mu_l1), crop(sd_l1),
            crop(mu_l2), crop(sd_l2),
            crop(mu_cost), crop(mu_sint),
            # MAD
            crop(mu_mad), crop(sd_mad),
        ]

        # Frangi/Sato multi-scale
        for trio in (mu_fr_list, sd_fr_list, mx_fr_list, mu_sa_list, sd_sa_list, mx_sa_list):
            for m in trio:
                maps.append(crop(m))

        # Gabor
        for m in gabor_mu: maps.append(crop(m))
        for m in gabor_sd: maps.append(crop(m))

        # Black-hat multi-radius
        for trio in (bh_mu, bh_sd, bh_mx):
            for m in trio:
                maps.append(crop(m))

        # final green channel
        Hc, Wc = maps[0].shape
        X = np.stack([m.reshape(-1) for m in maps], axis=1).astype(np.float32)
        yy, xx = np.mgrid[half:half+Hc, half:half+Wc]
        coords = list(zip(xx.reshape(-1).tolist(), yy.reshape(-1).tolist()))
        return X, coords

    def _hash_array(self, arr: np.ndarray) -> str:
        """Hash numpy array content for caching."""

        h = hashlib.sha1()
        h.update(str(arr.shape).encode()); h.update(str(arr.dtype).encode()); h.update(arr.tobytes())
        return h.hexdigest()

    def _train_cache_path(self, image_arr: np.ndarray, mask_arr: np.ndarray) -> Path:
        """Cache path for training features based on image and mask."""

        _, green = self.preprocess(image_arr)
        mhash = self._hash_array((mask_arr if self.resize_factor == 1 else self.processor.resize_image(mask_arr, scale=1/self.resize_factor)))
        ihash = self._hash_array(green)
        key = f"train_{self._cache_key_common()}_{ihash[:8]}_{mhash[:8]}"
        return (self.cache_dir / f"{key}.npz")

    def _pred_cache_path(self, image_arr: np.ndarray) -> Path:
        """Cache path for prediction features based on image only."""

        _, green = self.preprocess(image_arr)
        ihash = self._hash_array(green)
        key = f"pred_{self._cache_key_common()}_{ihash[:12]}"
        return (self.cache_dir / f"{key}.npz")

    def preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess the input image for feature extraction."""

        # Resize image
        if self.resize_factor > 1:
            image_resized = self.processor.resize_image(image, scale=1 / self.resize_factor)
        else:
            image_resized = image.copy()

        # Denoise and normalize green channel
        green_channel = self.processor.extract_green_channel(image_resized)
        if self.denoise_kernel >= 3:
            green_denoised = self.processor.denoise(green_channel, self.denoise_kernel)
        else:
            green_denoised = green_channel.copy()
        green_normalized = self.processor.normalize(green_denoised)

        if self.denoise_kernel >= 3:
            color_denoised = self.processor.denoise(image_resized, self.denoise_kernel)
        else:
            color_denoised = image_resized.copy()

        return color_denoised, green_normalized

    def extract_features_from_image(self, image: np.ndarray):
        """Convolutional feature extraction with caching."""

        # try to load from cache
        cache_path = self._pred_cache_path(image)
        if self.cache_enabled and cache_path.exists():
            try:
                data = np.load(cache_path, allow_pickle=False)
                feats = data["X"].astype(np.float32)
                coords_arr = data["coords"].astype(np.int32)
                coords = [tuple(c) for c in coords_arr]
                logger.info(f"Features loaded from cache: {cache_path.name} (X={len(feats):,})")
                return feats, coords
            except Exception as e:
                logger.warning(f"Pred cache read failed ({e}), recomputing...")

        # compute features
        fmap = self._precompute_feature_maps(image)
        feats, coords = self._build_feature_stack(fmap)

        # save to cache
        if self.cache_enabled:
            try:
                np.savez_compressed(cache_path, X=feats.astype(np.float32), coords=np.asarray(coords, dtype=np.int32))
                logger.info(f"Pred features cached: {cache_path.name} (X={len(feats):,})")
            except Exception as e:
                logger.warning(f"Pred cache write failed ({e})")

        return feats, coords