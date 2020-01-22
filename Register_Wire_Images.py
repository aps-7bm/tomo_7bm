'''Script to process wire images.
Script will normalize both images, flip one of the images,
and perform a cross-correlation to match the features.

Alan Kastengren, XSD, APS

Started April 4, 2017
'''
import tifffile
import numpy as np
import matplotlib.pyplot as plt
import glob
import epics
from skimage.feature import match_template
import scipy.signal

filename_root = 'Alignment'
bright_filename = 'Bright_000.tif'
dark_filename = 'Dark_000.tif'

#Get a list of the filenames that match the current pattern
filename_list = sorted(glob.glob('./'+filename_root+'*.tif'))

#Read in the bright image and the last two wire images
bright_image = np.array(tifffile.imread(bright_filename),dtype=np.float64)[2:,:]
dark_image = np.array(tifffile.imread(dark_filename),dtype=np.float64)[2:,:]
image_0deg = np.array(tifffile.imread(filename_list[-2]),dtype=np.float64)[2:,:]
image_180deg = np.array(tifffile.imread(filename_list[-1]),dtype=np.float64)[2:,:]


def norm_image(im, bright, dark):
    trans = (im - dark) / (bright - dark)
    trans[trans < 0] = 1e-6
    trans[trans > 0.75] = 1
    return -np.log(trans)

#Normalize the wire images by the bright image and flip one of them
norm_0deg = norm_image(image_0deg, bright_image, dark_image)
norm_180deg = norm_image(image_180deg[:,::-1], bright_image[:,::-1], dark_image[:,::-1])

plt.figure()
plt.imshow(norm_0deg - norm_180deg, vmin=-1, vmax=1)
plt.title('Raw 0 deg - 180 deg')
plt.colorbar()

#Make template sizes and shapes
padded_0deg = np.zeros((norm_0deg.shape[0]*2, norm_0deg.shape[1]*2))
row_shift = norm_0deg.shape[0]//2
col_shift = norm_0deg.shape[1]//2
padded_0deg[row_shift:norm_0deg.shape[0]+row_shift,col_shift:col_shift+norm_0deg.shape[1]] = norm_0deg
correlation_matrix = match_template(padded_0deg,norm_180deg)
print("Maximum correlation = " + str(np.max(correlation_matrix)))
plt.figure(2)
plt.imshow(correlation_matrix)
plt.colorbar()
plt.title('Correlation Matrix')
shift_tuple = np.unravel_index(np.argmax(correlation_matrix),correlation_matrix.shape)
row_shift = shift_tuple[0] - row_shift
col_shift = shift_tuple[1] - col_shift
norm_0deg_shifted = None
print("Row shift = " + str(row_shift))
print("Column shift = " + str(col_shift))
plt.figure()

vmin=-0.1
vmax=0.1

roll_0deg = np.roll(norm_0deg,-row_shift,0)
roll_0deg = np.roll(roll_0deg,-col_shift,1)
plt.imshow(roll_0deg-norm_180deg,vmin=vmin,vmax=vmax)
plt.axis('tight')
plt.grid('off')
plt.colorbar()
plt.title('Shifted 0 deg - 180 deg')


def extract_column(image,ratio,width=5):
    start_col = int(image.shape[1] * ratio)
    return np.sum(image[:,start_col:start_col+width],axis=1)


def fit_parabola(trace):
    '''Find the peak position of trace by a parabolic fit to peak.
    '''
    peak_int = np.argmax(trace)
    peak_region = trace[peak_int-1:peak_int+2]
    return peak_int + (peak_region[2] - peak_region[0]) / 2 / (2 * peak_region[1] - peak_region[0] - peak_region[2])


#Show the traces at column 3/4 of image size
slice_0_34 = extract_column(roll_0deg, 0.75)
slice_180_34 = extract_column(norm_180deg, 0.75)
slice_34_corr = scipy.signal.correlate(slice_0_34,slice_180_34) / np.sum(slice_0_34**2)
print(fit_parabola(slice_34_corr))
slice_0_14 = extract_column(roll_0deg, 0.25)
slice_180_14 = extract_column(norm_180deg, 0.25)
slice_14_corr = scipy.signal.correlate(slice_0_14,slice_180_14) / np.sum(slice_0_14**2)
print(fit_parabola(slice_14_corr))
corr_row_shift = fit_parabola(slice_34_corr) - fit_parabola(slice_14_corr)
print('Measured shift of {0:6.4e} pixels over {1:d} pixels'.format(corr_row_shift, roll_0deg.shape[1]//2))
print('Implies an angle between images of {0:7.5f} deg'.format(np.degrees(corr_row_shift/roll_0deg.shape[1]/2.0)))

plt.figure(5)
plt.plot(slice_34_corr / np.max(slice_34_corr),'r.')
plt.plot(slice_14_corr / np.max(slice_14_corr),'g.')
plt.figure(4)
plt.plot(slice_0_34, 'r.')
plt.plot(slice_180_34, 'b.')
plt.plot(slice_0_14, 'g.')
plt.plot(slice_180_14, 'k.')
plt.show()
