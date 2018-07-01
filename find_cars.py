import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from scipy.ndimage.measurements import label
import os
from moviepy.editor import VideoFileClip
from multiprocessing import Manager, Process, cpu_count, Queue, Lock
from functions import *


loop_num = 1 # use to find_cars function , indicate the first loop
window_slid = [] # slid windows result
labels = None # label function result
last_labels = None # last labels
work_nums = cpu_count() # Get number of CPU to process img paralleling
images = glob.glob('./train_data/*/*/*.png') # Get train images' filepath

color_space = 'RGB' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = 0 # Can be 0, 1, 2, or "ALL"
spatial_size = (32, 32) # Spatial binning dimensions
hist_bins = 32    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off


if os.path.exists('model.joblib') and os.path.exists('X_scaler.joblib'):
    print("The model files have already existed! ")
    print("If you want to retrain the model, delete 'model.joblib' and 'X_scaler.joblib' ")
    print()

else:
    print("Model is training ...")
    # cars and notcars use to save train data
    cars = []
    notcars = []

    for image in images:
        if 'non-vehicles' in image:
            notcars.append(image)
        else:
            cars.append(image)

    print(' In the train data, has %s cars and %s notcars, total number %s'%(len(cars), len(notcars), len(cars) + len(notcars)))

    t=time.time()
    car_features = extract_features(cars, color_space = color_space , spatial_size = spatial_size,
                                  hist_bins = hist_bins, orient = orient,
                                  pix_per_cell = pix_per_cell, cell_per_block = cell_per_block, hog_channel = hog_channel,
                                  spatial_feat = spatial_feat, hist_feat = hist_feat, hog_feat = hog_feat)

    notcar_features = extract_features(notcars, color_space = color_space , spatial_size = spatial_size,
                                  hist_bins = hist_bins, orient = orient,
                                  pix_per_cell = pix_per_cell, cell_per_block = cell_per_block, hog_channel = hog_channel,
                                  spatial_feat = spatial_feat, hist_feat = hist_feat, hog_feat = hog_feat)


    # print(np.array(car_features).shape)
    # print(np.array(car_features[0]).shape)
    # print(np.array(notcar_features).shape)


    t2 = time.time()
    print(' ', round(t2-t, 2), 'Seconds to extract HOG features...')

    # Create an array stack of feature vectors
    X = np.vstack((car_features, notcar_features)).astype(np.float64)

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=rand_state)

    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X_train)

    # Apply the scaler to X
    X_train = X_scaler.transform(X_train)
    X_test = X_scaler.transform(X_test)
    print(' ', 'Using:',orient,'orientations',pix_per_cell,'pixels per cell and', cell_per_block,'cells per block')
    print(' ', 'Feature vector length:', len(X_train[0]))

    # Use a linear SVC
    parameters = {'kernel':('linear', 'rbf'), 'C':[0.1, 1, 10]}

    # Define a model
    clf = SVC(kernel = 'rbf', C = 1)

    # Check the training time for the SVC
    t=time.time()
    clf.fit(X_train, y_train)
    t2 = time.time()
    print(' ', round(t2-t, 2), 'Seconds to train SVC...')

    # Check the score of the SVC
    print(' ', 'Test Accuracy of SVC = ', round(clf.score(X_test, y_test), 4))

    #Save svm and scaler models
    print(' ', "Saving model ...")
    joblib.dump(clf,'model.joblib')
    joblib.dump(X_scaler,'X_scaler.joblib')
    print(' ', "Model saved complete ! ")
    print("Model train complete ! ")




# multiprocessing function , to make multiprocessing data queue
def make_data(queue_01, queue_02, imgs, windows, work_nums):
    for i in range(len(windows)):
        queue_01.put(imgs[i])
        queue_02.put(windows[i])

    for i in range(work_nums):
        queue_01.put(None)
        queue_02.put(None)


# multiprocessing function , to handle multiprocessing data
def handle_data(queue_01, queue_02, clf, scaler, lock, rtn):
    #print('process id:', os.getpid())
    while True:
        lock.acquire()
        img = queue_01.get()
        window = queue_02.get()
        lock.release()

        #if window is None or feature is None:
        if window is None:
            break

        feature = single_img_features(img, color_space=color_space,
                            spatial_size=spatial_size, hist_bins=hist_bins,
                            orient=orient, pix_per_cell=pix_per_cell,
                            cell_per_block=cell_per_block,
                            hog_channel=hog_channel, spatial_feat=spatial_feat,
                            hist_feat=hist_feat, hog_feat=hog_feat)

        trans_feature = scaler.transform(np.array(feature).reshape(1, -1))
        pred = clf.predict(trans_feature)

        if pred[0] == 1:
            lock.acquire()
            rtn.append(window)
            lock.release()




# Define a function you will pass an image
# and the list of windows to be searched (output of slide_windows())
def search_windows(img, windows, clf = None,  scaler = None, color_space='RGB', 
                    spatial_size=(32, 32), hist_bins=32, 
                    hist_range=(0, 256), orient=9, 
                    pix_per_cell=8, cell_per_block=2, 
                    hog_channel=0, spatial_feat=True, 
                    hist_feat=True, hog_feat=True):

    manager = Manager() # multiprocessing Manager
    return_list = manager.list() # multiprocessing manager.list
    sub_imgs = []  # the sub img from image by windows

    queue_01 = Queue()  # multiprocessing queue
    queue_02 = Queue()  # multiprocessing queue
    lock = Lock()  # multiprocessing lock
    sub_process = []  # multiprocessing sub process list

    for window in windows:
        # get sub imgs by window
        sub_imgs.append(cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))   )


    # use multiprocessing to predict the sub imgs
    # generate data queue
    master_process = Process(target=make_data, args=(queue_01, queue_02, sub_imgs, windows, work_nums, ))

    # generate sub process
    for i in range(work_nums):
        sub_process1 = Process(target=handle_data, args=(queue_01, queue_02, clf, scaler, lock, return_list,))
        sub_process.append(sub_process1)

    # start sub process
    master_process.start()
    for p in sub_process:
        p.start()

    master_process.join()
    for p in sub_process:
        p.join()

    # Return windows for positive detections
    return return_list
    

# Load model
svc = joblib.load('model.joblib')
X_scaler = joblib.load('X_scaler.joblib')



def find_cars(image, skip = 8):

    global loop_num
    global window_slid
    global labels
    global last_labels
  
    heat_num = 2

    if window_slid == []:
        xy_windows = [(64, 64)]

        y_start = image.shape[0] // 2
        y_stop = None

        for xy_window in xy_windows:
        
            if y_start > image.shape[0] - xy_window[0]:
                break
        
            y_start_stop =[y_start ,  y_stop]
            window = slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop, 
                                #xy_window=xy_window, xy_overlap=(0.5, 0.9))
                                #xy_window=xy_window, xy_overlap=(0.1, 0.9))
                                #xy_window=xy_window, xy_overlap=(0.8, 0.8))
                                xy_window=xy_window, xy_overlap=(0.6, 0.8))

            window_slid.extend(window)
            y_start += xy_window[0] // 2

    if (labels is None and loop_num == 1) or (loop_num % skip == skip - 1):

        hot_windows = search_windows(image, window_slid, clf = svc, scaler = X_scaler, color_space=color_space,
                                spatial_size=spatial_size, hist_bins=hist_bins, 
                                orient=orient, pix_per_cell=pix_per_cell, 
                                cell_per_block=cell_per_block, 
                                hog_channel=hog_channel, spatial_feat=spatial_feat, 
                                hist_feat=hist_feat, hog_feat=hog_feat)                       
        

        heat = np.zeros_like(image[:,:,0]).astype(np.float)

        #If last labels position is not None, add it to current search_windows result
        if not (last_labels is None):
            hot_windows.extend(last_labels)
            heat_num = 3 # Because hot_windows extend last labels, so increase heat threshold

        # Add heat to each box in box list
        heat = add_heat(heat, hot_windows)
        
        # Apply threshold to help remove false positives
        heat = apply_threshold(heat,heat_num)
        
        # Visualize the heatmap when displaying
        heatmap = np.clip(heat, 0, 255)
        
        # Find final boxes from heatmap using label function
        labels = label(heatmap)

        # Save label position on the image , use to next image
        last_labels = get_labeled_bboxes_pos(labels)

    # draw label on to image
    draw_img = draw_labeled_bboxes(np.copy(image), labels)

    loop_num += 1

    # return the result
    return draw_img






#Address test_images
 images = os.listdir("test_images/")
 for i in range(len(images)):
     image = mpimg.imread('test_images/' + images[i])
     print(images[i])
     img = find_cars(image, skip = 1)
     mpimg.imsave('output_images/' + images[i], img)



#Address project_video
white_output = 'output_videos/project_video.mp4'
clip1 = VideoFileClip("project_video.mp4")
white_clip = clip1.fl_image(find_cars)
white_clip.write_videofile(white_output, audio=False)



#Debug code
'''

i = 3

if i == 1:
    tempfilename = 'test4.jpg'
    image = mpimg.imread('test_images/' + tempfilename)
    print(tempfilename)
    img = find_cars(image)
    mpimg.imsave('output_images/' + tempfilename, img)


if i == 2:
    images = os.listdir("test_images/")
    for i in range(len(images)):
        image = mpimg.imread('test_images/' + images[i])
        print(images[i])
        img = find_cars(image)
        mpimg.imsave('output_images/' + images[i], img)
        
    
if i == 3:        
    white_output = 'output_videos/test_video.mp4'
    clip1 = VideoFileClip("test_video.mp4")
    white_clip = clip1.fl_image(find_cars) 
    white_clip.write_videofile(white_output, audio=False)


if i == 4: 
    white_output = 'output_videos/project_video.mp4'
    clip1 = VideoFileClip("project_video.mp4")
    white_clip = clip1.fl_image(find_cars) 
    white_clip.write_videofile(white_output, audio=False)


'''
