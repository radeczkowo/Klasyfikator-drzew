from rozdanych import *


DIR = os.path.split(os.path.abspath(__file__))[0]
DATADIR = os.path.join(DIR, 'drzewa')
TDATADIR = os.path.join(DIR, 'trainingdata')
drzewa = ["dab", "sosna"]
alldata = []
tdata = []
testdata = []
ptestdata = []
traindata = []
validdata = []
augvaliddata = []
nazwa="400"
IMG_SIZE = 400

X = []
y = []

def preparedata(point, roznica):
    pdata=[]
    countd = 0
    counts = 0
    for img, number in alldata:
        if number == 0 and countd < (point / 2) - roznica:
            pdata.append([img, number])
            countd = countd + 1
        if number == 1 and counts < ((point / 2) + 1 + roznica):
            pdata.append([img, number])
            counts = counts + 1
    for img, number in pdata:
        for index in range(len(alldata)):
            if img.all() == alldata[index][0].all():
                alldata.pop(index)
                break
    print "ilosc debow: ", countd
    print "ilosc sosen: ", counts
    return pdata



def createdata():
    for rodzaje in drzewa:
        path = os.path.join(DATADIR, rodzaje)
        number = drzewa.index(rodzaje)
        for img in os.listdir(path):
            try:
                img = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                H = img.shape[0]
                W = img.shape[1]
                bmg = np.reshape(img, (-1, H, W, 1))
                alldata.append([img, number])
            except:
                print "popsuted"
                continue
    random.shuffle(alldata)
    point1 = int(0.20 * len(alldata))
    point2 = int(0.20 * len(alldata))
    print len(alldata)
    validdata = preparedata(point2, 2)
    random.shuffle(validdata)
    print len(validdata)
    testdata= preparedata(point1, 1)
    random.shuffle(testdata)
    print len(testdata)
    print len(alldata)
    return validdata, testdata, alldata


def augmentation(tdata, traindata, rrange, plus):
    rrangedab = rrange+plus
    for img, number in tdata:
        #print number
        #plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        #plt.imshow(img, cmap="gray")
        #plt.show()
        if (number==0):
            rozszerzaniedanych(img, traindata, number, IMG_SIZE, rrangedab)
        else:
            rozszerzaniedanych(img, traindata, number, IMG_SIZE, rrange)
    random.shuffle(traindata)

def resizevalid(data):
    for number in range(len(data)):
        data[number][0] = cv2.resize(data[number][0], (IMG_SIZE, IMG_SIZE))
        #print data[number][1]
        #plt.imshow(data[number][0], cmap="gray")
        #plt.show()


def editdata(tdata):
    x = []
    y = []
    for features, label in tdata:
        x.append(features)
        y.append(label)
    X = np.array(x).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    X = np.array(X, dtype="float32")
    y = np.array(y, dtype="float32")
    return X, y


def savedata(pathname, data):
    pickdata = open(pathname, "wb")
    pickle.dump(data, pickdata)
    pickdata.close()

validdata, testdata, tdata = createdata()
#augmentation(validdata, augvaliddata, 4, 0)
augmentation(tdata, traindata, 5, 0)
resizevalid(validdata)
#resizevalid(tdata)
X, y = editdata(traindata)
testX, testy = editdata(validdata)
savedata(TDATADIR+"/X"+nazwa, X)
savedata(TDATADIR+"/testX"+nazwa, testX)
savedata(TDATADIR+"/y"+nazwa, y)
savedata(TDATADIR+"/testy"+nazwa, testy)
savedata(TDATADIR+"/test"+nazwa, testdata)


"""
def preparedata(point, roznica):
    pdata=[]
    countd = 0
    counts = 0
    for img, number in alldata:
        if number == 0 and countd < (point / 2) - roznica:
            pdata.append([img, number])
            countd = countd + 1
        if number == 1 and counts < ((point / 2) + 1 + roznica):
            pdata.append([img, number])
            counts = counts + 1
    for img, number in pdata:
        for index in range(len(alldata)):
            if img.all() == alldata[index][0].all():
                alldata.pop(index)
                break
    print "ilosc debow: ", countd
    print "ilosc sosen: ", counts
    return pdata



def createdata():
    for rodzaje in drzewa:
        path = os.path.join(DATADIR, rodzaje)
        number = drzewa.index(rodzaje)
        for img in os.listdir(path):
            try:
                img = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                H = img.shape[0]
                W = img.shape[1]
                bmg = np.reshape(img, (-1, H, W, 1))
                alldata.append([img, number])
            except:
                print "popsuted"
                continue
    random.shuffle(alldata)
    point1 = int(0.15 * len(alldata))
    point2 = int(0.3 * len(alldata))
    print len(alldata)
    validdata = preparedata(point2, 2)
    random.shuffle(validdata)
    print len(validdata)
    testdata= preparedata(point1, 1)
    random.shuffle(testdata)
    print len(testdata)
    print len(alldata)
    return validdata, testdata, alldata


def augmentation(tdata, traindata, rrange, plus):
    rrangedab = rrange+plus
    for img, number in tdata:
        #print number
        #plt.imshow(img, cmap="gray")
        #plt.show()
        if (number==0):
            rozszerzaniedanych(img, traindata, number, IMG_SIZE, rrangedab)
        else:
            rozszerzaniedanych(img, traindata, number, IMG_SIZE, rrange)
    random.shuffle(traindata)

def resizevalid():
    for number in range(len(validdata)):
        validdata[number][0] = cv2.resize(validdata[number][0], (IMG_SIZE, IMG_SIZE))


def editdata(tdata):
    x = []
    y = []
    for features, label in tdata:
        x.append(features)
        y.append(label)
    X = np.array(x).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    X = np.array(X, dtype="float32")
    y = np.array(y, dtype="float32")
    return X, y


def savedata(pathname, data):
    pickdata = open(pathname, "wb")
    pickle.dump(data, pickdata)
    pickdata.close()

validdata, testdata, tdata = createdata()
augmentation(tdata, traindata, 30, 3)
augmentation(validdata, augvaliddata, 10, 0)
X, y = editdata(traindata)
#resizevalid()
testX, testy = editdata(augvaliddata)
savedata(TDATADIR+"/X"+nazwa, X)
savedata(TDATADIR+"/testX"+nazwa, testX)
savedata(TDATADIR+"/y"+nazwa, y)
savedata(TDATADIR+"/testy"+nazwa, testy)
savedata(TDATADIR+"/test"+nazwa, testdata)

"""
