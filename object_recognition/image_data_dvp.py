import time
import numpy as np
import cv2


def local_maxima_numpy(img8, intensity_threshold=0):
    """
    Fast 3x3 local maxima using only NumPy.
    - img8: 2D uint8 image
    - intensity_threshold: minimum intensity (0-255) for a peak
    Returns: mask uint8 (0 or 1) of local maxima positions.
    """
    img = img8
    H, W = img.shape

    # Extract center region
    center = img[1:-1, 1:-1]

    # Initialize max_neigh with center values
    max_neigh = center.copy()

    max_neigh = np.maximum(max_neigh, img[0:H-2, 1:W-1])   # up
    max_neigh = np.maximum(max_neigh, img[2:H,   1:W-1])   # down
    max_neigh = np.maximum(max_neigh, img[1:H-1, 0:W-2])   # left
    max_neigh = np.maximum(max_neigh, img[1:H-1, 2:W])     # right

    max_neigh = np.maximum(max_neigh, img[0:H-2, 0:W-2])   # up-left
    max_neigh = np.maximum(max_neigh, img[0:H-2, 2:W])     # up-right
    max_neigh = np.maximum(max_neigh, img[2:H,   0:W-2])   # down-left
    max_neigh = np.maximum(max_neigh, img[2:H,   2:W])     # down-right


    local = (center == max_neigh)

    if intensity_threshold > 0:
        local &= (center >= intensity_threshold)

    mask = np.zeros_like(img, dtype=np.uint8)
    mask[1:-1, 1:-1] = local.astype(np.uint8)

    return mask

def draw_markers_on_image(image8, cx, cy, color=(0, 255, 0), size=1):
    """
    Fast drawing of EV markers using NumPy only.
    - image8: 2D uint8 grayscale image
    - cx, cy: lists/arrays of detected EV positions (x,y)
    - color: BGR tuple
    - size: half-size of marker (1 => 3x3 square)
    """
    H, W = image8.shape
    disp = cv2.cvtColor(image8, cv2.COLOR_GRAY2BGR)

    xs = np.asarray(cx, dtype=np.int32)
    ys = np.asarray(cy, dtype=np.int32)

    if xs.size == 0:
        return disp

    # clip to avoid boundary errors
    xs = np.clip(xs, size, W - size - 1)
    ys = np.clip(ys, size, H - size - 1)

    # draw small square around each EV
    for dy in range(-size, size + 1):
        for dx in range(-size, size + 1):
            disp[ys + dy, xs + dx] = color

    return disp


def nms_points(xs, ys, img8, d_min=3):
    """
    Non-maximum suppression on point coordinates.
    - xs, ys: 1D arrays of peak coordinates
    - img8: 2D uint8 image used for intensities
    - d_min: minimum separation between kept peaks (in pixels)
    Returns: xs_keep, ys_keep (1D arrays)
    """
    xs = np.asarray(xs, dtype=np.int32)
    ys = np.asarray(ys, dtype=np.int32)
    N = xs.size
    if N == 0:
        return xs, ys

    vals = img8[ys, xs].astype(np.float32)   # intensity at each peak
    order = np.argsort(-vals)                # indices sorted by intensity (desc)

    keep = np.ones(N, dtype=bool)

    for idx in order:
        if not keep[idx]:
            continue
        dx = xs - xs[idx]
        dy = ys - ys[idx]
        dist2 = dx*dx + dy*dy
        mask = (dist2 <= d_min*d_min)
        mask[idx] = False          
        keep[mask] = False

    return xs[keep], ys[keep]


def classify_aggregates(xs, ys, R_cluster=10, N_cluster=8):
    """
    Density-based aggregate classifier on point coordinates.
    - xs, ys: 1D arrays of peak coordinates after NMS
    - R_cluster: radius (px) to look for neighbors around each point
    - N_cluster: if a point has >= N_cluster neighbors within R_cluster,
                 it is labeled as 'aggregate'
    Returns:
      single_mask: boolean mask for single EVs
      aggregate_mask: boolean mask for aggregate points
      neighbor_counts: number of neighbors per point
    """
    xs = np.asarray(xs, dtype=np.int32)
    ys = np.asarray(ys, dtype=np.int32)
    N = xs.size
    if N == 0:
        return np.zeros(0, dtype=bool), np.zeros(0, dtype=bool), np.zeros(0, dtype=int)

    coords = np.stack([xs, ys], axis=1)       # shape (N, 2)
    dx = coords[:, 0][:, None] - coords[:, 0][None, :]
    dy = coords[:, 1][:, None] - coords[:, 1][None, :]
    dist2 = dx*dx + dy*dy

    # neighbors within radius (including self)
    neighbor_counts = (dist2 <= R_cluster*R_cluster).sum(axis=1) - 1   # exclude self

    aggregate_mask = neighbor_counts >= N_cluster
    single_mask = ~aggregate_mask

    return single_mask, aggregate_mask, neighbor_counts


def nms_points_v2(xs, ys, img8, d_min=6):
    """
    Non-maximum suppression on (xs, ys).
    d_min: min allowed distance between kept peaks (in pixels).
    Keeps the brightest point and removes neighbors within d_min.
    """
    xs = np.asarray(xs, dtype=np.int32)
    ys = np.asarray(ys, dtype=np.int32)
    N = xs.size
    if N == 0:
        return xs, ys

    vals = img8[ys, xs].astype(np.float32)    # intensity at each peak
    order = np.argsort(-vals)                 # indices sorted by intensity desc

    keep = np.ones(N, dtype=bool)

    for idx in order:
        if not keep[idx]:
            continue
        dx = xs - xs[idx]
        dy = ys - ys[idx]
        dist2 = dx*dx + dy*dy
        mask = (dist2 <= d_min*d_min)
        mask[idx] = False     # don't suppress itself
        keep[mask] = False

    return xs[keep], ys[keep]


def classify_aggregates_density(xs, ys, R_cluster=18, N_cluster=12):
    """
    Density-based aggregate classifier in coordinate space.
    - xs, ys: positions AFTER NMS
    - R_cluster: radius in pixels to search neighbors
    - N_cluster: if a point has >= N_cluster neighbors within R_cluster,
                 it is considered part of an aggregate.
    Returns:
      single_mask: bool mask for single EVs
      aggregate_mask: bool mask for aggregate peaks
      neighbor_counts: number of neighbors per peak
    """
    xs = np.asarray(xs, dtype=np.int32)
    ys = np.asarray(ys, dtype=np.int32)
    N = xs.size
    if N == 0:
        return np.zeros(0, bool), np.zeros(0, bool), np.zeros(0, int)

    coords = np.stack([xs, ys], axis=1)  # (N,2)
    dx = coords[:, 0][:, None] - coords[:, 0][None, :]
    dy = coords[:, 1][:, None] - coords[:, 1][None, :]
    dist2 = dx*dx + dy*dy

    neighbor_counts = (dist2 <= R_cluster*R_cluster).sum(axis=1) - 1  # exclude self
    aggregate_mask = neighbor_counts >= N_cluster
    single_mask = ~aggregate_mask

    return single_mask, aggregate_mask, neighbor_counts


def build_aggregate_mask(img8, bright_thresh=40, min_area=400, close_radius=3):
    """
    Image-based aggregate mask.
    - img8: 2D uint8 image used for detection
    - bright_thresh: threshold for 'bright mass'
    - min_area: minimum connected area in pixels to be considered an aggregate
    - close_radius: radius (px) for morphological closing
    Returns:
      aggregate_mask: bool array, True where aggregate regions are present.
    """
    # threshold bright regions
    _, bright = cv2.threshold(img8, int(bright_thresh), 255, cv2.THRESH_BINARY)

    # morphological closing to connect blobs
    ksize = 2 * close_radius + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    bright_closed = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, kernel, iterations=2)

    # connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bright_closed, connectivity=8)

    aggregate_mask = np.zeros_like(img8, dtype=bool)
    for i in range(1, num_labels):  # skip background
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            aggregate_mask[labels == i] = True

    return aggregate_mask


def draw_EVs_fast(image8, cx, cy, color=(0, 255, 0), size=1):
    """
    Fast drawing of EV markers using NumPy.
    - image8: 2D uint8 grayscale image
    - cx, cy: lists/arrays of EV centers (x, y)
    - color: BGR tuple
    - size: half-size of square marker (1 → 3x3)
    """
    H, W = image8.shape
    disp = cv2.cvtColor(image8, cv2.COLOR_GRAY2BGR)

    xs = np.asarray(cx, dtype=np.int32)
    ys = np.asarray(cy, dtype=np.int32)

    if xs.size == 0:
        return disp

    xs = np.clip(xs, size, W - size - 1)
    ys = np.clip(ys, size, H - size - 1)

    for dy in range(-size, size + 1):
        for dx in range(-size, size + 1):
            disp[ys + dy, xs + dx] = color

    return disp

class ImageManager:
    '''
    Class to be used to store the acquired images split in N channels
    and methods useful for object identification and ROI creation.
    '''

    def __init__(self, dim_h, dim_v,
                 roisize,
                 Nchannels=2, dtype=np.uint16, debug=False):

        # original images from the N channels
        self.image = np.zeros((Nchannels, dim_v, dim_h), dtype)

        self.dim_h = dim_h
        self.dim_v = dim_v

        self.contours = []   # contours of detected objects
        self.cx = []         # x coordinates of centroids
        self.cy = []         # y coordinates of centroids

        self.roisize = roisize
        self.debug = debug

    def clear_countours(self):
        self.contours = []
        self.cx = []
        self.cy = []

   
    def find_object_localmax_NMS(self, channel=0,
                                 bitdepth=12,
                                 norm_factor=None,
                                 intensity_threshold=10,
                                 nms_d_min=3,
                                 R_cluster=10,
                                 N_cluster=8):
        """
        Detect EVs using local maxima + NMS + aggregate filtering (density only).
        """
        im_ch = self.image[channel]

        # ---- 1) 16-bit -> 8-bit for detection ----
        if norm_factor is None:
            norm_factor = (2**bitdepth - 1) / 255.0
        img8 = (im_ch / norm_factor).astype('uint8')

        # ---- 2) Local maxima (NumPy) ----
        maxima_mask = local_maxima_numpy(img8, intensity_threshold=intensity_threshold)
        ys, xs = np.nonzero(maxima_mask)   # row, col

        # ---- 3) Non-maximum suppression ----
        xs_nms, ys_nms = nms_points(xs, ys, img8, d_min=nms_d_min)

        # ---- 4) Density-based aggregate classification ----
        single_mask, aggregate_mask, _ = classify_aggregates(
            xs_nms, ys_nms, R_cluster=R_cluster, N_cluster=N_cluster
        )

        xs_single = xs_nms[single_mask]
        ys_single = ys_nms[single_mask]

        # store aggregates separately
        self.aggregate_x = xs_nms[aggregate_mask].tolist() if np.any(aggregate_mask) else []
        self.aggregate_y = ys_nms[aggregate_mask].tolist() if np.any(aggregate_mask) else []

        # ---- 5) Build contours ONLY for single EVs ----
        contours = []
        cx = []
        cy = []

        roisize = self.roisize
        H, W = img8.shape

        for x0, y0 in zip(xs_single, ys_single):
            x = int(x0 - roisize // 2)
            y = int(y0 - roisize // 2)
            w = h = roisize

            if x > 0 and y > 0 and x + w < W - 1 and y + h < H - 1:
                cnt = np.array([
                    [x,       y      ],
                    [x + w-1, y      ],
                    [x + w-1, y + h-1],
                    [x,       y + h-1]
                ], dtype=np.int32).reshape((-1, 1, 2))
                contours.append(cnt)
                cx.append(int(x0))
                cy.append(int(y0))

        self.cx = cx
        self.cy = cy
        self.contours = contours

        if self.debug:
            self.image8bit = img8

        print(f"[NMS] Total peaks (raw): {len(xs)}")
        print(f"[NMS] After NMS:          {len(xs_nms)}")
        print(f"[NMS] Singles kept:       {len(self.cx)}")
        print(f"[NMS] Aggregates ignored: {np.count_nonzero(aggregate_mask)}")

    
    def find_object_localmax_NMS_v2(self,
                                    channel=0,
                                    bitdepth=12,
                                    norm_factor=None,
                                    intensity_threshold=10,
                                    nms_d_min=6,
                                    R_cluster=18,
                                    N_cluster=12,
                                    use_image_agg_mask=True,
                                    bright_thresh=40,
                                    aggregate_min_area=400):
        """
        EV detection using:
          - local maxima (NumPy)
          - non-maximum suppression (NMS, stronger)
          - density-based aggregate filtering
          - optional image-based aggregate mask
        """
        im_ch = self.image[channel]

        # ---- 1) 16-bit -> 8-bit for DETECTION ----
        if norm_factor is None:
            norm_factor = (2**bitdepth - 1) / 255.0
        img8 = (im_ch / norm_factor).astype('uint8')

        # ---- 2) Local maxima ----
        maxima_mask = local_maxima_numpy(img8, intensity_threshold=intensity_threshold)

        # ---- 3) Image-based aggregate mask (optional) ----
        if use_image_agg_mask:
            agg_mask_img = build_aggregate_mask(
                img8,
                bright_thresh=bright_thresh,
                min_area=aggregate_min_area,
                close_radius=3
            )
            maxima_mask[agg_mask_img] = 0

        ys, xs = np.nonzero(maxima_mask)   # row, col
        total_raw = len(xs)

        # ---- 4) NMS to remove duplicates near each EV ----
        xs_nms, ys_nms = nms_points_v2(xs, ys, img8, d_min=nms_d_min)

        # ---- 5) Density-based aggregate classification ----
        single_mask, aggregate_mask_pts, _ = classify_aggregates_density(
            xs_nms, ys_nms,
            R_cluster=R_cluster,
            N_cluster=N_cluster
        )

        xs_single = xs_nms[single_mask]
        ys_single = ys_nms[single_mask]
        xs_agg = xs_nms[aggregate_mask_pts]
        ys_agg = ys_nms[aggregate_mask_pts]

        # store aggregates separately (for debugging / later use)
        self.aggregate_x = xs_agg.tolist()
        self.aggregate_y = ys_agg.tolist()

        # ---- 6) Build contours ONLY for single EVs ----
        contours = []
        cx = []
        cy = []

        roisize = self.roisize
        H, W = img8.shape

        for x0, y0 in zip(xs_single, ys_single):
            x = int(x0 - roisize // 2)
            y = int(y0 - roisize // 2)
            w = h = roisize

            if x > 0 and y > 0 and x + w < W - 1 and y + h < H - 1:
                cnt = np.array([
                    [x,       y      ],
                    [x + w-1, y      ],
                    [x + w-1, y + h-1],
                    [x,       y + h-1]
                ], dtype=np.int32).reshape((-1, 1, 2))
                contours.append(cnt)
                cx.append(int(x0))
                cy.append(int(y0))

        self.cx = cx
        self.cy = cy
        self.contours = contours

        if self.debug:
            self.image8bit = img8

        print(f"[NMS_v2] Total peaks (raw): {total_raw}")
        print(f"[NMS_v2] After NMS:         {len(xs_nms)}")
        print(f"[NMS_v2] Singles kept:      {len(self.cx)}")
        print(f"[NMS_v2] Aggregates (coords): {len(self.aggregate_x)}")

   
    def find_object_localmax(self, channel=0,
                             bitdepth=12,
                             norm_factor=None,
                             intensity_threshold=10):
       
        im_ch = self.image[channel]

        # Convert to 8-bit for processing
        if norm_factor is None:
            norm_factor = (2**bitdepth - 1) / 255.0
        img8 = (im_ch / norm_factor).astype('uint8')

        # 1) Find local maxima mask with NumPy
        maxima_mask = local_maxima_numpy(
            img8,
            intensity_threshold=intensity_threshold
        )

        # 2) Extract coordinates of maxima
        ys, xs = np.nonzero(maxima_mask)   # row, col
        cx = xs.tolist()
        cy = ys.tolist()

        # 3) Build rectangles around each peak
        contours = []
        roisize = self.roisize
        H, W = img8.shape

        for x0, y0 in zip(xs, ys):
            x = int(x0 - roisize // 2)
            y = int(y0 - roisize // 2)
            w = h = roisize

            # Only keep ROIs fully inside image
            if x > 0 and y > 0 and x + w < W - 1 and y + h < H - 1:
                cnt = np.array([
                    [x,       y      ],
                    [x + w-1, y      ],
                    [x + w-1, y + h-1],
                    [x,       y + h-1]
                ], dtype=np.int32).reshape((-1, 1, 2))
                contours.append(cnt)

        self.cx = [c for c, _ in zip(cx, contours)]  # ensure only kept ones
        self.cy = [c for c, _ in zip(cy, contours)]
        self.contours = contours

        if self.debug:
            self.image8bit = maxima_mask * 255

    
    def draw_markers(self, image8, size=1, color=(0, 255, 0)):
        xs = np.asarray(self.cx, dtype=np.int32)
        ys = np.asarray(self.cy, dtype=np.int32)
        return draw_markers_on_image(image8, xs, ys, color=color, size=size)

    def draw_EVs(self, image8, size=1, color=(0, 255, 0)):
        return draw_EVs_fast(image8, self.cx, self.cy, color=color, size=size)

    
    def find_object(self, channel=0, min_object_area=100, max_object_area=1000,
                    bitdepth=12, norm_factor=None):
        """Legacy Otsu-based object detector."""
        im = self.image[channel]
        if norm_factor is None:
            norm_factor = (2**bitdepth - 1) / 255
        image8bit = (im / norm_factor).astype('uint8')

        _ret, thresh_pre = cv2.threshold(image8bit, 0, 255,
                                         cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh_pre, cv2.MORPH_OPEN, kernel, iterations=1)

        if self.debug:
            self.image8bit = thresh

        cnts, _hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                            cv2.CHAIN_APPROX_SIMPLE)
        cx = []
        cy = []
        contours = []
        roisize = self.roisize
        l = image8bit.shape

        for cnt in cnts:
            M = cv2.moments(cnt)
            if M['m00'] > int(min_object_area) and M['m00'] < int(max_object_area):
                x0 = int(M['m10']/M['m00'])
                y0 = int(M['m01']/M['m00'])
                x = int(x0 - roisize//2)
                y = int(y0 - roisize//2)
                w = h = roisize

                if x > 0 and y > 0 and x+w < l[1]-1 and y+h < l[0]-1:
                    cx.append(x0)
                    cy.append(y0)
                    contours.append(cnt)

        self.cx = cx
        self.cy = cy
        self.contours = contours

    
    def copy(self):
        """
        Returns a deep copy of the ImageManager instance.
        """
        new_im = ImageManager(
            self.dim_h,
            self.dim_v,
            self.roisize,
            Nchannels=self.image.shape[0],
            dtype=self.image.dtype
        )
        new_im.image = self.image.copy()
        new_im.contours = [cnt.copy() for cnt in self.contours]
        new_im.cx = self.cx.copy()
        new_im.cy = self.cy.copy()
        return new_im

    def draw_contours_on_image(self, image8bit):
        """
        Draw legacy rectangle annotations around self.contours.
        """
        cx = self.cx
        cy = self.cy
        roisize = self.roisize
        contours = self.contours

        displayed_image = cv2.cvtColor(image8bit, cv2.COLOR_GRAY2RGB)

        for indx, _val in enumerate(cx):
            x = int(cx[indx] - roisize//2)
            y = int(cy[indx] - roisize//2)
            w = h = roisize

            displayed_image = cv2.drawContours(displayed_image,
                                               [contours[indx]], 0, (0, 256, 0), 2)

            if indx == 0:
                color = (256, 0, 0)
            else:
                color = (0, 0, 256)

            cv2.rectangle(displayed_image, (x, y), (x+w, y+h), color, 1)

        return displayed_image

    def extract_rois(self, ch, cx, cy):
        """
        Extract ROIs given centroids cx, cy in channel ch.
        """
        image16bit = self.image[ch]

        roisize = self.roisize
        rois = []

        for indx, _val in enumerate(cx):
            x = int(cx[indx] - roisize//2)
            y = int(cy[indx] - roisize//2)
            w = h = roisize
            detail = image16bit[y:y+w, x:x+h]
            rois.append(detail)

        return rois

    def highlight_channel(self, displayed_image):
        cv2.rectangle(displayed_image,
                      (0, 0),
                      (self.dim_h-1, self.dim_v-1),
                      (255, 255, 0), 3)
