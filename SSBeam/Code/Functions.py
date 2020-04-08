import numpy as np
# import cupy as np

def ShowList(li, opt = "ListType"):
    for value in li:
        print(value)

    if opt == "ListType":
        print("List Type: " + str(type(value)))
    elif opt == "ItemType":
        print("List Item Type: " + str(type(value[0])))

def ShowDict(di, opt = "DictType"):
    for key, value in di.items():
        print(key, value)

    if opt == "DictType":
        print("Key Type: " + str(type(key)) + ", Value Type: " + str(type(value)))
    elif opt == "ItemType":
        print("Key Type: " + str(type(key)) + ", Value Item Type: " + str(type(value[0])))

def ShowMatrix(mat):
    if type(mat) != np.ndarray:
        print("Given matrix is not numpy array.")
    else:
        if len(mat.shape) == 1:
            nRow = mat.shape
            for i in range(nRow[0]):
                print('{0:11.4e}'.format(mat[i]))
        elif len(mat.shape) == 2:
            nRow, nColumn = mat.shape
            for i in range(nRow):
                for j in range(nColumn):
                    print('{0:11.4e}'.format(mat[i][j]), end=' ')
                print()

def WriteMatrix(filename, mat, isNew=True):
    if type(mat) != np.ndarray:
        print("Given matrix is not numpy array.")
    else:
        if isNew == True:
            f = open(filename, 'w')
        else:
            f = open(filename, 'a')
            f.write('\n')

        if len(mat.shape) == 1:
            nRow = mat.shape
            for i in range(nRow[0]):
                f.write('{0:11.4e}'.format(mat[i]))
        elif len(mat.shape) == 2:
            nRow, nColumn = mat.shape
            f.write(filename.split('.')[0]+'\n')
            f.write('      ')
            dataType = type(mat[0][0])
            for j in range(nColumn):
                if dataType == np.complex128:
                    data = '{0:25d}'.format(j+1) + ' '
                    f.write(data)
                else:
                    data = '{0:11d}'.format(j+1) + ' '
                    f.write(data)
            f.write('\n')

            for i in range(nRow):
                f.write('{0:5d}'.format(i+1)+' ')
                for j in range(nColumn):
                    if dataType == np.complex128:
                        data = '(' + '{0:11.4e}'.format(mat[i][j].real) + ',' + '{0:11.4e}'.format(mat[i][j].imag) + ') '
                        f.write(data)
                    else:
                        data = '{0:11.4e}'.format(mat[i][j]) + ' '
                        f.write(data)
                f.write('\n')
        f.close()


# def WriteMatrix(filename, mat):
#         f = open(filename, 'w')
#         if len(np.shape(mat)) == 1:
#             nRow = np.shape(mat)
#             for i in range(nRow[0]):
#                 f.write('{0:11.4e}'.format(mat[i]))
#         elif len(np.shape(mat)) == 2:
#             nRow, nColumn = np.shape(mat)
#             f.write(filename.split('.')[0]+'\n')
#             f.write('      ')
#             for j in range(nColumn):
#                 data = '{0:11d}'.format(j+1) + ' '
#                 f.write(data)
#             f.write('\n')
#             for i in range(nRow):
#                 f.write('{0:5d}'.format(i+1)+' ')
#                 for j in range(nColumn):
#                     data = '{0:11.4e}'.format(mat[i][j]) + ' '
#                     f.write(data)
#                 f.write('\n')
#         f.close()    
