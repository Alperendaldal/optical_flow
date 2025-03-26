import cv2
import numpy as np
import argparse

''' Draws a line on an image with color corresponding to the direction of line
 image im: image to draw line on
 float x, y: starting point of line
 float dx, dy: vector corresponding to line angle and magnitude
'''
def draw_line():
  return None


''' Make an integral image or summed area table from an image
 image im: image to process
 returns: image I such that I[x,y] = sum{i<=x, j<=y}(im[i,j])
'''
def make_integral_image(img):
  height, width, channels = img.shape
  new_im = np.zeros((height, width, channels), dtype=np.uint32)

  for y in range(height):
      for x in range(width):
          new_im[y, x] = img[y, x]

          if x > 0:
              new_im[y, x] += new_im[y, x - 1]
          if y > 0:
              new_im[y, x] += new_im[y - 1, x]
          if x > 0 and y > 0:
              new_im[y, x] -= new_im[y - 1, x - 1]

  
  return new_im

''' Apply a box filter to an image using an integral image for speed
 image im: image to smooth
 int s: window size for box filter
 returns: smoothed image
'''
def box_filter_image(img, s):
  img = make_integral_image(img)
  h, w, c = img.shape
  new_img = np.zeros((h, w, 3), dtype=np.uint32)
  kernel = np.ones((s, s), dtype=np.float32) / 9
  pad_size = s//2
  padded = np.pad(img, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), mode='constant')
    
  for ch in range(c):
      for i in range(h):
          for j in range(w):
              region = padded[i:i+s, j:j+s, ch]
              new_img[i, j, ch] = np.sum(region * kernel)  

  cv2.imwrite("box_filter.jpg",new_img)
  return new_img

"""Ek method"""
def conv2d(im, kernel):
    h, w, c = im.shape
    k_size = kernel.shape[0]
    pad = k_size // 2
    padded = np.pad(im, ((pad, pad), (pad, pad), (0, 0)), mode='constant')
    new_img = np.zeros((h, w, 3), dtype=np.uint8)
    
    for ch in range(c):
        for i in range(h):
            for j in range(w):
                region = padded[i:i+k_size, j:j+k_size, ch]
                new_img[i, j, ch] =  np.clip(np.sum(region * kernel).astype(np.float32), 0, 255).astype(np.uint8)
    
    return new_img

''' Calculate the time-structure matrix of an image pair.
 image im: the input image.
 image prev: the previous image in sequence.
 int s: window size for smoothing.
 returns: structure matrix. 1st channel is Ix^2, 2nd channel is Iy^2,
          3rd channel is IxIy, 4th channel is IxIt, 5th channel is IyIt.
'''



"""RETURN FONKSŞİYONU DÜZENLENECEK GRAY SCALE OLARAK İŞLENEBİLİR """
def time_structure_matrix(im,image_prev):
  weight, height , colochanel = im.shape
  It = np.zeros((weight,height,colochanel),np.float32)
  structure_matrix = np.zeros((weight,height,colochanel),np.float32)
  gx_kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
  gy_kernel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    
  grad_x = conv2d(im, gx_kernel)
  grad_y = conv2d(im, gy_kernel)


  for c in range(colochanel):
    for w in range(weight):
      for h in range(height):
         It[w,h,c]= im[w,h,c]-image_prev[w,h,c]
        
  IX = grad_x*grad_x
  IY = grad_y*grad_y
  XY = grad_x*grad_y  
  TX = grad_x*It
  TY = grad_y*It

  return  IX,IY,XY,TX,TY
'''
Calculate the velocity given a structure image
image S: time-structure image
int stride: 
''' 
def velocity_image(im,image_prev,S):
  weight, height , colochanel = im.shape
  structure_matrix = np.zeros((weight,height,5),np.float32)
  structure_matrix =  time_structure_matrix(im,image_prev)
  out_matrix = structure_matrix = np.zeros((weight,height,2),np.float32)
  M = np.zeros((2,2),np.float32)
  Right_Side_vector = np.zeros((1,2),np.float32)
  for w in range(weight):
     for h in range(height):
        for x in range(2):
           for y in range(2):
              M[x,y] = structure_matrix[w,h(x+y-2)]

  return None
'''
Draw lines on an image given the velocity
image im: image to draw on
image v: velocity of each pixel
float scale: scalar to multiply velocity by for drawing
'''
def draw_flow():
  return None
'''
Constrain the absolute value of each image pixel
image im: image to constrain
float v: each pixel will be in range [-v, v]
'''
def constrain_image():
  return None
'''
Calculate the optical flow between two images
image im: current image
image prev: previous image
int smooth: amount to smooth structure matrix by
int stride: downsampling for velocity matrix
returns: velocity matrix
'''
def optical_flow_images():
  return None
'''
Run optical flow demo on webcam
int smooth: amount to smooth structure matrix by
int stride: downsampling for velocity matrix
int div: downsampling factor for images from webcam
'''
def optical_flow_webcam():
  return None

def __main__():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run image resizing.")
    parser.add_argument('img_name', type=str, help="Path to the input image")
    # Required argument for the image filename
    args = parser.parse_args()

    # Load the image
    img = cv2.imread("resim.jpg")
    make_integral_image(img)
    box_filter_image(img,5)
    out = optical_flow_images(img)
    
if __name__ == "__main__":
    
    __main__()
