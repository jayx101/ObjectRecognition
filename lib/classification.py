import numpy as np
import pandas as pd
import cv2, os
# from sklearn.svm import LinearSVC
import imutils
import joblib
from sklearn import metrics
import autothresh as improc
# from SimpleCV import (HueHistogramFeatureExtractor, EdgeHistogramFeatureExtractor, HaarLikeFeatureExtractor, ImageClass)

path_prefix = '/home/opencv/ObjectRecognition/'
surf = cv2.xfeatures2d.SURF_create(400)
simple_cv = False

# CONSTANTS
TEST_PATH = path_prefix + 'dataset/test/'
TRAIN_PATH = path_prefix + 'dataset/train/'
OUTLIERS_PATH = path_prefix + 'dataset/outliers/'

params = {}

def getClasses():
    '''
    Gets a list of the test classes and returns as dict
    format: {id:classname}
    '''
    training_names = os.listdir(TRAIN_PATH)
    classes = {}
    class_id = 0

    for training_name in training_names:
        classes[class_id] = training_name
        class_id += 1
    return classes

def class2Name(class_list):
    '''
    Pass it a list of classID's and it will return class names.
    '''
    classes = getClasses()
    ret = [classes[l] for l in class_list]

    return ret

def getImageMetadata(folderPath):
    '''
    Traverses the path specified,
    finds the folders and names them as the classes, and returns all files.
    Params:
        Folder where class folders resice
    Returns:
        image_path: paths of all images
        image_classes: numeric classes
        classes: dict with classes
    '''
    training_names = os.listdir(folderPath)
    image_paths = []
    image_classes = []
    # classes = {}
    class_id = 0

    for training_name in training_names:
        dir = os.path.join(folderPath, training_name)
        class_path = [os.path.join(dir, f) for f in os.listdir(dir)]

        image_paths += class_path
        image_classes += [class_id] * len(class_path)
        # classes[class_id] = training_name
        class_id += 1

    return image_paths, image_classes

def preProcessImages(image_paths):
    descriptors = []
    # loop through images in path and get desciptors
    for image_path in image_paths:
        # im = cv2.imread(image_path)
        im = imgPrePreprocess(image_path)
        kpts = surf.detect(im)
        kpts, des = surf.compute(im, kpts)
        descriptors.append(des)

    return descriptors

def getSimpleFeat(img):

    hhfe = HueHistogramFeatureExtractor(10)
    ehfe = EdgeHistogramFeatureExtractor(10)
    haarfe = HaarLikeFeatureExtractor(fname=path_prefix + 'haar.txt')

    img = ImageClass.Image(img)
    feat = []
    feat.extend(hhfe.extract(img))
    feat.extend(ehfe.extract(img))
    feat.extend(haarfe.extract(img))

    return feat

def createBOWVocab(path_train):
    if params is None:
        raise ValueError('first set classifier parameters')

    bow_train = cv2.BOWKMeansTrainer(params['bow_size'])

    for des in preProcessImages(path_train):
        bow_train.add(des)

    voc = bow_train.cluster()

    return voc

def setClassifierParams(bow_size=100, crop=False, resize=False, resize_width=600,
                       grayscale=False, hessian=400, verbose=False):
    '''
    Set all parameters needed to run this classifier
    '''
    params['crop'] = crop
    params['resize'] = resize
    params['resize_width'] = resize_width
    params['grayscale'] = grayscale
    params['hessian'] = hessian
    params['bow_size'] = bow_size
    params['verbose'] = verbose

    surf.setHessianThreshold(hessian)

def imgPrePreprocess(im):

    if not any(params):
        raise ValueError('first set classifier parameters using setClassifierParams()')

    # check if a path or image is passed
    if isinstance(im, basestring):
        im = cv2.imread(im)
    if params['crop']:
        im = improc.cropRedImage(im, False)
    if params['resize']:
        im = imutils.resize(im, width=params['resize_width'])
    if params['grayscale']:
        im = cv2.cvtColor(im, 6)
    if params['verbose']:
        cv2.imshow('frame', im)
        cv2.waitKey(100)

    return im

def extractX(image_paths, voc):
    X = []

    bow_desc_extractor = createMatcher(voc)

    for imagepath in image_paths:

        # run image transformation pipeline
        im = imgPrePreprocess(imagepath)

        featureset = bow_desc_extractor.compute(im, surf.detect(im))
        if simple_cv:
            simple_feat = getSimpleFeat(im)
            simple_feat = np.array(simple_feat).reshape(-1, len(simple_feat))
            featureset = np.concatenate((featureset, simple_feat), axis=1)
        X.extend(featureset)

    return X

def extractXfromimg(im, voc):
    X = []
    bow_desc_extractor = createMatcher(voc)

    # run image transformation pipeline
    im = imgPrePreprocess(im)

    featureset = bow_desc_extractor.compute(im, surf.detect(im))
    if simple_cv:
        simple_feat = getSimpleFeat(im)
        simple_feat = np.array(simple_feat).reshape(-1, len(simple_feat))
        featureset = np.concatenate((featureset, simple_feat), axis=1)
    X.extend(featureset)

    return X

def createMatcher(voc):
    # extract featues from vocab
    flann_params = dict(algorithm=0, trees=5)
    matcher = cv2.FlannBasedMatcher(flann_params, {})
    bow_desc_extractor = cv2.BOWImgDescriptorExtractor(surf, matcher)
    bow_desc_extractor.setVocabulary(voc)

    return bow_desc_extractor

def getClassWeights():
    _, classes = getImageMetadata(TRAIN_PATH)

    from collections import Counter
    c = Counter(classes)
    print c
    weights = {}
    for i in c:
        weights[i] = (float(sum(c.values())) / c[i])
    return weights

def getTrainData(persist=False):
    if persist:
        train_paths, y = getImageMetadata(TRAIN_PATH)
        voc = createBOWVocab(train_paths)

        X = extractX(train_paths, voc)

        joblib.dump((X, np.array(y), voc), path_prefix + "lib/X.pkl", compress=3)
        train_data = (X, y, voc)
    else:
        try:
            train_data = joblib.load(path_prefix + "lib/X.pkl")
        except:
            getTrainData(True)

    return train_data

def trainGen(classifier, persist=False):
    weights = getClassWeights()

    clf = classifier(class_weight=weights)
    # train_data = X, y, voc
    train_data = getTrainData(persist)

    clf.fit(train_data[0], train_data[1])

    joblib.dump((train_data[2], clf), path_prefix + "lib/imagereco.pkl", compress=3)

    return train_data

def getTestData(persist=False):
    test_data = ()

    if persist:
        train_data = joblib.load(path_prefix + "lib/X.pkl")
        test_paths, test_classes = getImageMetadata(TEST_PATH)
        # use just the extacted vocab to extract the test X data
        X = extractX(test_paths, train_data[2])

        joblib.dump((X, np.array(test_classes), test_paths), path_prefix + "lib/X_test.pkl", compress=3)
        test_data = (X, test_classes, test_paths)
    else:
        try:
            test_data = joblib.load(path_prefix + "lib/X_test.pkl")
        except:
            getTestData(True)
    return test_data

def getOutlierData():

    train_data = joblib.load(path_prefix + "lib/X.pkl")
    out_paths, out_classes = getImageMetadata(OUTLIERS_PATH)

    # use just the extacted vocab to extract the test X data
    X = extractX(out_paths, train_data[2])

    return X

def predict(model, persist=False):

    # get test data tuple 0=X, 1=y
    test_data = getTestData(persist)
    X = test_data[0]
    y = test_data[1]
    test_paths = test_data[2]

    y_pred = model.predict(X)

    # get prediction distande from hyperplane
    # hyp = model.decision_function()

    report = metrics.classification_report(y, y_pred)

    y_pred_rep = class2Name(y_pred)
    test_classes = class2Name(y)

    df = pd.DataFrame(test_classes)
    files = [os.path.split(p)[1] for p in test_paths]
    df['files'] = pd.Series(files)
    # df['actual'] = pd.Series(test_classes)
    df['predicted'] = pd.Series(y_pred_rep)

    return report, df

def predictImg(model, img):

    # get test data tuple 0=X, 1=y, 2=vocab
    train_data = joblib.load(path_prefix + "lib/X.pkl")
    X = extractXfromimg(img, train_data[2])

    y_pred = model.predict(X)

    hyperplane = model.decision_function(X)

    return y_pred, hyperplane

# td = trainGen(LinearSVC, True)
# print td[0]
# print td[1]
# clf = LinearSVC()
# clf.fit(td[0], td[1])

# rep, df = predict(clf, True)

# print rep
# print df

# print getClassWeights()
# ## predict single image
# im = cv2.imread('../dataset/test/tv/foundimg.jpg')
# y_pred, hyperplane = predictImg(im)

# print class2Name(y_pred), hyperplane
