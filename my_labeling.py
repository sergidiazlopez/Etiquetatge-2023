__authors__ = '1604158, 1599349, 1601959'
__group__ = 'PRACT2_1'

import numpy as np
from matplotlib import pyplot as plt
from collections import deque
from Kmeans import KMeans, get_colors
from KNN import KNN
from utils import colors
from utils_data import read_dataset, visualize_retrieval, read_extended_dataset, crop_images
import time

def get_shape_accuracy(etiquetes, ground_truth):

    valor = np.sum(etiquetes == ground_truth)
    total = etiquetes.shape[0]
    percentatge = round(valor / total * 100, 2)
    return percentatge

def Kmean_statistics(KMax, K):

    x = []
    tiempo = []
    WCD = []
    while (K <= KMax):

        kmeans = KMeans(train_imgs[0], K)
        inicio = time.time()
        kmeans.fit()
        fin = time.time()
        wcd = kmeans.withinClassDistance()
        WCD.append(wcd)
        tiempo.append(fin-inicio)
        x.append(K)
        K = K + 1
    return x,WCD, tiempo

def get_color_accuracy(etiquetes, ground_truth):
    valor = 0
    for colores, result in zip(etiquetes, ground_truth):
        iguales = all(item in colores for item in result)
        if iguales == True:
            valor = valor + 1
    total = etiquetes.shape[0]
    percentatge = round(valor / total * 100, 2)
    return percentatge

def bar_chart(numbers, labels, pos):
    plt.bar(pos, numbers, color='blue')
    plt.xticks(ticks=pos, labels=labels)
    plt.show()


def retrieval_by_color(images, tags, query):
    """
    :param images: List of images
    :param tags: List of colors obtained from Kmeans
    :param query: List of strings containing colors to search
    :return: numpy array all images containing
    """
    ids = []
    results = []
    for i, (img, colors) in enumerate(zip(images, tags)):
        inters = np.intersect1d(query, colors)
        if inters.size > 0:  # If one or more colors of the query appear in the img
            ids.append(i)
            results.append(img)

    return ids, np.array(results)


def retrieval_by_shape(images, tags, query):
    """
    :param images: List of images
    :param tags: List of shapes obtained from KNN
    :param query: List of strings containing colors to search
    :return: numpy array, all images containing
    """
    ids = []
    results = []
    for i, (img, shapes) in enumerate(zip(images, tags)):
        inters = np.intersect1d(query, shapes)
        if inters.size > 0:  # If one or more shapes of the query appear in the img
            ids.append(i)
            results.append(img)

    return ids, np.array(results)


def retrieval_combined(images, tags_color, tags_shape, query):
    """
    :param images: List of images
    :param tags_color: List of colors obtained from Kmeans
    :param tags_shape: List of shapes obtained from KNN
    :param query: List of strings containing colors to search
    :return: numpy array, all images containing
    """
    ids = []
    results = []
    for i, (img, colors, shapes) in enumerate(zip(images, tags_color, tags_shape)):
        intersection_colors = np.intersect1d(query, colors)
        intersection_shapes = np.intersect1d(query, shapes)
        if intersection_colors.size > 0 and intersection_shapes.size > 0:  # If one or more colors and shapes of the query appear
            ids.append(i)
            results.append(img)

    return ids, np.array(results)


if __name__ == '__main__':
    # Load all the images and GT
    train_imgs, train_class_labels, train_color_labels, test_imgs, test_class_labels, \
        test_color_labels = read_dataset(root_folder='./images/', gt_json='./images/gt.json')

    # List with all the existent classes
    classes = list(set(list(train_class_labels) + list(test_class_labels)))

    # Load extended ground truth
    imgs, class_labels, color_labels, upper, lower, background = read_extended_dataset()
    cropped_images = crop_images(imgs, upper, lower)

    # You can start coding your functions here

    # Qualitative analysis - Color
    """
    print('Executing KMeans')
    start_time = time()
    prediction_colors = []  # Execute KMeans for every image to get color tags
    for input in test_imgs:
        model = KMeans(input)
        model.find_bestK(5)  # max k = 7?
        model.fit()
        c = get_colors(model.centroids)
        prediction_colors.append(c)
    end_time = time()
    print('Finished, took', end_time - start_time, ' seconds')

    for c in colors:
        ids, retrieval = retrieval_by_color(test_imgs, prediction_colors, [c])
        truth = []
        truth_labels = []
        for i in ids:
            truth_labels.append(test_color_labels[i])
            truth.append(True if c in test_color_labels[i] else False)
        visualize_retrieval(retrieval, 15, title=c, ok=truth, info=truth_labels)

    
    print('Executing KMeans with Extended GT and cropped images')
    start_time = time.time()
    prediction_colors = []  # Execute KMeans for every image to get color tags
    for input in cropped_images:
        model = KMeans(input, options = {'fitting':'WCD'})
        model.find_bestK(5) # max k = 7?
        model.fit()
        c = get_colors(model.centroids)
        prediction_colors.append(c)
    end_time = time.time()
    print('Finished, took', end_time - start_time, ' seconds')

    for c in colors:
        ids, retrieval = retrieval_by_color(imgs, prediction_colors, [c])
        truth = []
        truth_labels = []
        for i in ids:
            truth_labels.append(color_labels[i])
            truth.append(True if c in color_labels[i] else False)
        visualize_retrieval(retrieval, 20, title = c, ok = truth, info=truth_labels)


    # Qualitative analysis - Shape
    print('Executing KNN')
    start_time = time.time()
    model = KNN(train_imgs, train_class_labels)
    prediction_shapes = model.predict(test_imgs, 3) # k = 3????
    end_time = time.time()
    print('Finished, took', end_time - start_time, ' seconds')

    for c in classes:
        ids, retrieval = retrieval_by_shape(test_imgs, prediction_shapes, [c])
        truth = []
        truth_labels = []
        for i in ids:
            truth_labels.append(test_class_labels[i])
            truth.append(True if c in test_class_labels[i] else False)
        visualize_retrieval(retrieval, 15, title = c, ok = truth, info=truth_labels)

    # Qualitative analysis - Combined
    prediction_shapes = model.predict(imgs, 3)
    ids, retrieval = retrieval_combined(imgs, prediction_colors, prediction_shapes, ["Red", "Shirts"])
    visualize_retrieval(retrieval, 15)
    """
    K = 4
    KMax = 10
    # Quantitative analysis - Shape
    knn = KNN(train_imgs, train_class_labels)
    # Quantitative analysis - WCD
    x, WCD, tiempo = Kmean_statistics(KMax, K)
    pos = list(range(len(x)))
    plt.xlabel("K")
    # WCD
    plt.ylabel("WCD")
    bar_chart(WCD, x, pos)
    # Tiempo
    #plt.ylabel("Tiempo")
    #bar_chart(tiempo,x,pos)

    print("KNN")
    print("K = ", K, " Percentatge = ", get_shape_accuracy(knn.predict(test_imgs, K),
                                                           test_class_labels), "%")
    colores = deque()
    for i in test_imgs:
        kmeans = KMeans(i, K)
        kmeans.fit()
        color = get_colors(kmeans.centroids)
        color = list(color)
        colores.append(color)
    array = np.array(colores)
    porcentaje = get_color_accuracy(array, test_color_labels)
    print("KNN")
    print("K = ", K, " Percentatge = ", porcentaje, "%")