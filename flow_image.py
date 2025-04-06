import cv2
import numpy as np
import argparse

"""örnek run kodu"""
"""python flow_image.py img1.jpg img2.jpg --smooth 15 --stride 8"""
"""python -c "from flow_image import optical_flow_webcam; optical_flow_webcam()"""


def draw_line(im, x, y, dx, dy, color):
    angle = np.arctan2(dy, dx)
    color = (int(255 * (np.cos(angle) + 1)/2), 
             int(255 * (np.sin(angle) + 1)/2), 
             128)
    cv2.line(im, (int(x), int(y)), (int(x + dx), int(y + dy)), color, 1)
    cv2.circle(im, (int(x), int(y)), 2, color, -1)

def make_integral_image(im):
    """rbg imah"""
    if len(im.shape) == 3:
        
        integral = np.zeros_like(im, dtype=np.float32)
        
        for c in range(im.shape[2]):
            temp = np.cumsum(np.cumsum(im[:, :, c].astype(np.float32), axis=0), axis=1)
            integral[:, :, c] = temp
            """print(integral)"""
        return integral
    else:
        
        return np.cumsum(np.cumsum(im.astype(np.float32), axis=0), axis=1)

"""benzer kod smooting"""
def box_filter_image(im, s):

    if s <= 1:
        return im.copy()
    
    integral = make_integral_image(im)
    h, w = im.shape[:2]
    out = np.zeros_like(im, dtype=np.float32)
    s2 = s // 2
    
    if len(im.shape) == 3:
        for y in range(h):
            for x in range(w):
                y1 = max(0, y - s2)
                x1 = max(0, x - s2)
                y2 = min(h-1, y + s2)
                x2 = min(w-1, x + s2)
                
                area = (y2 - y1) * (x2 - x1)
                for c in range(im.shape[2]):
                    out[y,x,c] = (integral[y2,x2,c] - integral[y1,x2,c] - 
                                 integral[y2,x1,c] + integral[y1,x1,c]) / area
    else:
        for y in range(h):
            for x in range(w):
                y1 = max(0, y - s2)
                x1 = max(0, x - s2)
                y2 = min(h-1, y + s2)
                x2 = min(w-1, x + s2)
                
                area = (y2 - y1) * (x2 - x1)
                out[y,x] = (integral[y2,x2] - integral[y1,x2] - 
                           integral[y2,x1] + integral[y1,x1]) / area
    return out


''' Calculate the time-structure matrix of an image pair.
 image im: the input image.
 image prev: the previous image in sequence.
 int s: window size for smoothing.
 returns: structure matrix. 1st channel is Ix^2, 2nd channel is Iy^2,
        3rd channel is IxIy, 4th channel is IxIt, 5th channel is IyIt.
'''
def time_structure_matrix(im, prev, s):
    
    if im.shape != prev.shape:
        prev = cv2.resize(prev, (im.shape[1], im.shape[0]))
    
    if len(im.shape) == 3:
        im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    else:
        im_gray = im
        prev_gray = prev
    """önceki ödevde yazılşdı"""
    Ix = cv2.Sobel(im_gray, cv2.CV_32F, 1, 0, ksize=3)
    Iy = cv2.Sobel(im_gray, cv2.CV_32F, 0, 1, ksize=3)
    """32 bit"""
    It = im_gray.astype(np.float32) - prev_gray.astype(np.float32)
    
    Ix2 = Ix * Ix
    Iy2 = Iy * Iy
    IxIy = Ix * Iy
    IxIt = Ix * It
    IyIt = Iy * It
    
    Ix2_smooth = box_filter_image(Ix2, s)
    Iy2_smooth = box_filter_image(Iy2, s)

    IxIy_smooth = box_filter_image(IxIy, s)
    IxIt_smooth = box_filter_image(IxIt, s)
    IyIt_smooth = box_filter_image(IyIt, s)
    structure = np.zeros((im.shape[0], im.shape[1], 5), dtype=np.float32)
    structure[:,:,0] = Ix2_smooth
    structure[:,:,1] = Iy2_smooth
    structure[:,:,2] = IxIy_smooth
    structure[:,:,3] = IxIt_smooth
    structure[:,:,4] = IyIt_smooth
    return structure

def velocity_image(S, stride):
    
    h, w = S.shape[:2]
    vx = np.zeros((h, w), dtype=np.float32)
    vy = np.zeros((h, w), dtype=np.float32)
    
    for y in range(0, h, stride):
        for x in range(0, w, stride):
            a = S[y,x,0]
            b = S[y,x,2]
            c = S[y,x,1]
            d = -S[y,x,3]
            e = -S[y,x,4]
            
            M = np.array([[a, b], [b, c]])
            det = a * c - b * b
            if det > 1e-6:
                inv_M = np.array([[c, -b], [-b, a]]) / det
                v = inv_M @ np.array([d, e])
                vx[y,x] = v[0]
                vy[y,x] = v[1]
    
    return vx, vy
'''
Calculate the optical flow between two images
image im: current image
image prev: previous image
int smooth: amount to smooth structure matrix by
int stride: downsampling for velocity matrix
returns: velocity matrix
'''
def draw_flow(im, v, scale):
    
    vx, vy = v
    h, w = im.shape[:2]
    
    for y in range(0, h, 8):
        for x in range(0, w, 8):
            dx = vx[y,x] * scale
            dy = vy[y,x] * scale
            if dx*dx + dy*dy > 0.1:
                draw_line(im, x, y, dx, dy, (0, 255, 0))

def optical_flow_images(im, prev, smooth=15, stride=8):
    S = time_structure_matrix(im, prev, smooth)
    vx, vy = velocity_image(S, stride)
    return vx, vy

"""Kapatma fonksiyonu ekle
"""
def optical_flow_webcam(smooth=15, stride=4, div=8):
    
    cap = cv2.VideoCapture(0)
    
    ret, prev = cap.read()
    if not ret:
        print("caamera not found")
        return
    
    prev = cv2.resize(prev, (prev.shape[1]//div, prev.shape[0]//div))
    
    while True:
        ret, im = cap.read()
 
        
        im = cv2.resize(im, (im.shape[1]//div, im.shape[0]//div))
        vx, vy = optical_flow_images(im, prev, smooth, stride)
        
        draw_flow(im, (vx, vy), 5)
        cv2.imshow('VİDEO', im)
        
        prev = im.copy()
        
        """Yanıt vermiyor"""
        cv2.waitKey(1)
    
    cap.release()
    cv2.destroyAllWindows()

def __main__():
    # Parse command-line arguments for input image files
    parser = argparse.ArgumentParser(description="Run optical flow between two frames.")
    parser.add_argument('img1', type=str, help="Path to the first image")
    parser.add_argument('img2', type=str, help="Path to the second image")
    parser.add_argument('--smooth', type=int, default=15)
    parser.add_argument('--stride', type=int, default=8)
    args = parser.parse_args()

    # Read the images
    img1 = cv2.imread(args.img1)
    img2 = cv2.imread(args.img2)



    # Run optical flow
    vx, vy = optical_flow_images(img2, img1, args.smooth, args.stride)

    # Visualize flow on the first image
    draw_flow(img1, (vx, vy), 5)

    # Display the result
    cv2.imshow('Result', img1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite("output_result.jpg", img1)

if __name__ == "__main__":
    __main__()