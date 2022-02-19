import cv2
import pylab
from scipy import ndimage
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from skimage.morphology import convex_hull_image
from skimage.feature import peak_local_max
from skimage.morphology import watershed
import matplotlib as mpl
import numpy as np
import math
import os
import glob
import statistics
from multiprocessing import Process, Manager, Pool

# cv2.ADAPTIVE_THRESH_MEAN_C #cv2.ADAPTIVE_THRESH_GAUSSIAN_C
  # cv2.THRESH_BINARY #cv2.THRESH_BINARY_INV
  # odd number like 3,5,7,9,11 ++++ 11 is best
  # constant to be subtracted
  # particles with smaller size will be removed
  # put your directory with photos here
# directory = r"E:\plasma experiments\20210817 laser scattering on a diffuser 1500 grit 7h"  # put your directory with photos here
extension = '.Bmp'


# USER GUI ===================================================================================

def remove_small_objects(img):
    min_size = 4
    # find all your connected components (white blobs in your image)
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=8)
    # connectedComponentswithStats yields every seperated component with information on each of them, such as size
    # the following part is just taking out the background which is also considered a component, but most of the time we don't want that.
    sizes = stats[1:, -1]
    nb_components = nb_components - 1

    # your answer image
    img2 = img
    # for every component in the image, you keep it only if it's above min_size
    for i in range(0, nb_components):
        if sizes[i] < min_size:
            img2[output == i + 1] = 0

    return img2


def filebrowser():
    directory = r"E:\plasma experiments\20210817 laser scattering on a diffuser 1500 grit 7h"  # put your directory with photos here
    print('extension')
    arr = []
    "Returns files with an extension"
    for name in glob.glob(f"{directory}/*{extension}"):
        arr.append(name)
    return arr


def photo_analysis(img):
    maxValue = 255
    adaptiveMethod = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
    thresholdType = cv2.THRESH_BINARY
    blockSize = 11
    C = -3
    particle_count = 0
    im_thresholded_avg = 0
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    img = cv2.blur(img, (5, 5))
    img = cv2.bilateralFilter(img, 5, 75, 75)
    img = cv2.medianBlur(img, 11)
    im_thresholded = cv2.adaptiveThreshold(img, maxValue, adaptiveMethod, thresholdType, blockSize, C)
    s = ndimage.generate_binary_structure(2, 2)
    labelarray, particles_count = ndimage.measurements.label(im_thresholded, structure=s)
    return im_thresholded, particles_count


def filling_holes(img):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        cv2.drawContours(img, [cnt], 0, 255, -1)
    return img


def image_processing(directory_, process_i, file_names_array_, counter_array_, capturing_param_, frames_frequency_, n_, file_name_):
    total_duration_of_experiment = frames_frequency_ / 1000 * len(file_names_array_) * capturing_param_
    particles_count_avg_array_ = []
    print()
    print("=======================================")
    print(f'capturing_param = {capturing_param_}')
    print(f'frames_frequency = {frames_frequency_}')
    print(f'avg param n = {n_}')
    print(f"Number of avg calculations: {len(file_names_array_) // n_}\n")
    print("=======================================")
    print()
    print(f'total_duration_of_experiment = {total_duration_of_experiment}')

    print(f'PROCESS {process_i} has been started!')
    i_ = 0
    im_photo = [0] * n_
    while i_ <= len(file_names_array_):
        if i_ + n_ > len(file_names_array_):
            break

        print(f"Progress ........ {round((i_ / len(file_names_array_) * 100), 2)} %")
        img_thresholded = [0] * n_
        img_thresholded_avg = 0

        for j in range(n_):
            im_photo[j] = cv2.imread(file_names_array_[i_ + j])
            img_thresholded[j], particles_count = photo_analysis(im_photo[j])
            img_thresholded_avg += img_thresholded[j]

        remove_small_objects(img_thresholded_avg)
        filling_holes(img_thresholded_avg)
        s = ndimage.generate_binary_structure(2, 2)
        labelarray, particles_count_avg = ndimage.measurements.label(img_thresholded_avg, structure=s)
        particles_count_avg_array_.append(particles_count_avg)
        i_ = i_ + n_
        counter_array_[process_i] += 1

    print()
    print("=======================================")
    print("avg array")
    print("=======================================")
    print()

    for y in range(len(particles_count_avg_array_)):
        print('avg[', y * 30, '] = ', particles_count_avg_array_[y])

    t = np.linspace(0, total_duration_of_experiment, len(particles_count_avg_array_))
    #save_data_file_name = directory_ + '\\' + file_name_ + str(process_i) + '.csv'
    save_data_file_name = file_name_ + str(process_i) + '.csv'
    np.savetxt(save_data_file_name, np.vstack((t, particles_count_avg_array_)).T, delimiter=', ')


if __name__ == '__main__':
    manager = Manager()
    file_names_array = filebrowser()

    NUMBER_OF_PARTITION = 2

    file_names_array_LENGTH = len(file_names_array)
    print('file_names_array_LENGTH', file_names_array_LENGTH)
    partition = file_names_array_LENGTH // NUMBER_OF_PARTITION
    partition_remains = file_names_array_LENGTH - (partition * NUMBER_OF_PARTITION)
    if partition_remains:
        NUMBER_OF_PROCESSES = NUMBER_OF_PARTITION + 1
    else:
        NUMBER_OF_PROCESSES = NUMBER_OF_PARTITION

    print('partition & partition_remains', partition, partition_remains)

    #particles_count_avg_array = manager.list()
    i = 0
    counter = 0  # could be converted into time
    counter_arr = [0] * NUMBER_OF_PROCESSES

    counter_arr_proc = manager.list()
    counter_arr_proc = counter_arr_proc + counter_arr

    directory = r"E:\plasma experiments\20210817 laser scattering on a diffuser 1500 grit 7h"  # put your directory with photos here
    print(f"Folder: {directory}")
    print(f"File extensions: {extension}")
    print(f"Number of captured frames: {len(file_names_array)}")
    print()

    while True:
        try:
            capturing_param = int(input('1 of ... frames captured? (1, 2, 3, ..): '))
        except ValueError:
            print("That's not an integer!")
            continue
        else:
            if capturing_param >= 0:
                break
            else:
                print("Wrong input")

    while True:
        try:
            frames_frequency = int(input('time between frames (in ms): '))
        except ValueError:
            print("That's not an integer!")
            continue
        else:
            if frames_frequency >= 0:
                break
            else:
                print("Wrong input")

    while True:
        try:
            n = int(input('averaging param (1-5 recommended): '))
        except ValueError:
            print("That's not an integer!")
            continue
        else:
            if n >= 0:
                break
            else:
                print("Wrong input")

    file_name = input('Save as (file name): ')

    ar = []
    for h in range(NUMBER_OF_PROCESSES):
        ar.append((directory, h, file_names_array[(partition * h):(partition * (h + 1))], counter_arr_proc, capturing_param, frames_frequency, n, file_name))

    with Pool(NUMBER_OF_PROCESSES) as pool:
        pool.starmap(image_processing, ar)
