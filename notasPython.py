#### PYTHON VANILLA
import math

### TIPOS DE DATOS

a1 = 'Esto' # String
print(a1,type(a1))

a2 = """ 
Bloque de
texto
"""         # String
print(a2,type(a2))

a3 = f'{a1} no es un {a2}'
print(a3,type(a3))

b = 10       # Number -> Integer
print(b,type(b))

b1 = 10.1    # Number -> Float
print(b1,type(b1))

b2 = 10 + 1j # Number -> Complex
print(b2,type(b2), b2.real, b2.imag)

c = ['david','BU',1,1.0] # List
print(c,type(c))

d = ('Juan','P') # Tuples
print(d,type(d))

e = {'nombre':'David','score':3.5} # Dictionary
print(e,type(e))

f = {1,2,3,4,5} # Set
print(f,type(f))

g = False # Boolean
print(g,type(g))


### NUMEROS
## Operaciones Numericas
print(2.5 + 2.1)
print(4 - 3.5)
print(5 * 4)
print(5 / 6)
print(2 ** 4) # Exponenciación
print(4 % 3)  # Modulo -> Residuo de una division entera
print(5//2)   # Floor division -> Aproximate result of division to prev int

## Functions for numbers
print(int(2.5))       # Convertir a entero
print(float(2.5))     # Convertir a flotante
print(abs(-2.5))      # Valor absoluto
round(math.pi,3)      # round(numberToRound, decimalNumbersToLeave)
math.ceil(3.5)        # math.ceil(numberToCeil) -> Aproxmate to next int
math.floor(3.5)       # math.floor(numberToFloor) -> Aproxmate to prev int
print(divmod(94, 21)) # Devuleve la division entera y el modulo -> divmod(dividendo, divisor)
print(complex(4,3))   # Convertir a imaginario -> complex(intNumber,imagNumber)

### STRINGS

## Function for Strings
len('string')
a1.isnumeric() # Deuelve booleano dependiendo si el string es un numero o no
a1[0]          # Devuelve el numero en el index indicado
a1[1:-2]       # Devuelve el pedazo de codigo entre las posiciones -> string[indexInicioCorte, indexFinCorte]

### 

#### NUMPY
import numpy as np

### ARRAY
## 1) Solo admite un tipo de dato por array (string, bool, int, float, complex)
## 2) Es mucho más rapido que una lista ya que es más pequeño en tamaño porque no tiene metadatos -> Fixed type
## 3) Numpy tiene memoria contigua, no tiene pointers a los espacios de memoria con las variables sino que estan seguido -> Contiguos memory

a = np.array([1,2,3], dtype='int16')              #Create an numpy array -> Vector 1D
b = np.array([[1.0 ,2.5 ,3.2], [1.3, 1.0, 1.2]])  #Create an numpy array -> Matrix 2D
c = np.array([[[1,2],[3,4]],[[5,6],[7,8]]])       #Create an numpy array -> Tensor 3D


### BASICS

b.shape    # Gets a tuple with dimension AXB -> (2, 3)
a.ndim     # Get Dimension of the array -> 1
a.size     # Get the number of elements inside the array -> 3
a.itemsize # Get size of the data elements in array -> 4 (bytes)
a.nbytes   # Gets the size in bytes that occupies the array -> 12
a.dtype    # Gets type of data elements in array -> int16

### ACCESING ELEMENTS -> a[r] - a[r, c] - (...) - a[n, ... , r, c ] -> n is the closer to the first []

b[0,2]      # Gets element from row 1, column 2 -> 3.2
b[0, :]     # Get specific row -> [1.0 ,2.5 ,3.2]
b[: ,-1]    # Get specific column -> [3.2, 1.2]
b[0:3:2, 0] # Get a specific section (from row or column) -> [r, startIndex:endIndex:stepSize] or [startIndex:endIndex:stepSize, c]
b[0,2] = 5  # Change value from an specific cell

### INITIALIZING DIFFERENT TYPE OF ARRAYS

a.copy()                           # Use this method to make a copy of the original array, if b = array, any make made to b will change also a

## VECTOR
np.arange(5, 5.3, 0.1)             # Returns a array with the values specified -> [5. , 5.1, 5.2] ->  np.arange(initialVal, finalVal, step)
np.linspace(5, 5.5, 3)             # Return an array with values between a range -> [5.  , 5.25, 5.5] -> np.linspace(initialVal, finalVal, sizeOfArray)

## MATRIXES or +2D
np.zeros((2,3))                    # Returns a matrix full of zeros in the shape specified -> array([[0., 0., 0.],[0., 0., 0.]])
np.ones((4,2,2), dtype='int32')    # Returns a matrix full of ones in the shape specified
np.full((2,2), 99)                 # Returns a matrix full of _specifiedNumber_ in the shape specified
np.random.rand(4,2)                # Returns a matrix with random values (0<val<1) in the specified shape
np.random.randint(4,7, size=(3,3)) # Returns a matrix with random integers in specified range and shape np.random.randint(minVal,maxVal, size=(r,c))
np.identity(5)                     # Creates de identidy of the matrix ( Diagonal ones - the rest is zero )
np.repeat([[1,2,3]], 3, axis=0)    # Repearts an array -> [[1 2 3], [1 2 3], [1 2 3]]

### MODIFY AN ARRAY

np.insert(a,(0,-1),-1)   # Inserta un valor dentro de el o los index especificados np.insert(array,index(es), valToInsert)
np.append(a, -5)         # Coloca al final de el array "a" el numero -5
np.delete(a, 0)          # Elimina el item que esté en el index indicado



np.where(a < 5, a, 10*a) # Itera sobre un array, si el item cumple la condición se convierte en a1, si no se convierte en a2 -> numpy.where(condition, a1, a2)
np.extract(a > 5, a*2)   # Necesita dos listas, un array que indique si es falso o verdadero y de donde se van a extraer los elementos # Solo para 1D
np.mod(a,3) == 0         # Devuelve los valores divisibles como true y el resto como False

a = np.array([[1,2],[3,4],[5,6],[3,4]])
np.compress([0,1,0,1], a, axis=0) # Devuelve los elementos del axis 0 (Filas) que se seleccionen -> a = Matrix 4x2 (R X C) -> La funcion indicada devuelve la segunda y la cuarta fila

a= np.array([[1,2],[3,4]])
b= np.array([[5,6]])

# Concatenate can only be used with same ndimensions arrays
np.concatenate((a,b),axis=0) # Concatenate a to b by rows -> [[1 2], [3 4], [5 6]]
np.concatenate((a,b.T),axis=1) # Concatenate a to b by columns, the T transpose the array -> [[1 2 5], [3 4 6]]

np.sort(a)    # Ordena un vector, matriz o +2D
np.argsort(a) # Devuelve un array de index con el order correcto del array/matrix/etc "a"

a.flatten(order="C")         # Flat the array in order C -> QUESTION -> Como funciona con 3D o Más
a.flatten(order="F")
a.flatten(order="A")
a.flatten(order="K")

### REORGANIZING ARRAYS

a = np.array([[1,2,3,4],[5,6,7,8]])

a.reshape((2,4))         # This will reshape the array, the number of elements needs to be equal to a*b -> a.reshape((a,b)) -> [[1, 2], [3, 4], [5, 6], [7, 8]]

v1 = np.array([1,2,3,4])
v2 = np.array([5,6,7,8])

np.vstack([v1,v2,v1,v2]) # This will stack verticaly the vectors (1D) -> [[1,2,3,4], [5,6,7,8], [1,2,3,4], [5,6,7,8]]
np.hstack([v1,v2])       # This will stack horizontaly the vectors (1D) -> [1, 2, 3, 4, 5, 6, 7, 8]

np.hsplit(a,2)           # This will divide -> QUESTION -> Cómo funciona
np.vsplit(a,2)           # This will divide -> QUESTION -> Cómo funciona


### MATHS USED IN ARRAYS
## For more https://docs.scipy.org/doc/numpy/reference/routines.math.html

a = np.array([1,2,3,4])
b = np.array([1,0,1,0])

a + 2  # [3, 4, 5, 6]
a - b  # [0, 2, 2, 4]
a * 2  # [2, 4, 6, 8]
a / 2  # [0.5, 1.0, 1.5, 2]
a ** 2 # [1, 4, 9, 16]

a = np.array([1,2,3,4])

a > 2  # [False, False, True, True]
a >= 2 # [False, True, True, True]
a < 9  # [True, True, True, True]
a <= 1 # [True, False, False, False]
a == a # [True True True True]
a == b # [True False False False]
a != b # [False True True True]

a + b  # [2, 2, 4, 4]
# (...)

a = np.array([1,2,3,4])
b = np.array([1,0,1,0])

np.array_equal(a,b)   # False
np.greater(a,b)       # [False,  True,  True,  True]
np.greater_equal(a,b) # [ True,  True,  True,  True]
np.less(a,b)          # [False, False, False, False]
np.less_equal(a,b)    # [ True, False, False, False]
np.equal(a,b)         # [ True, False, False, False]
np.not_equal(a,b)     # [False,  True,  True,  True]


np.sin(a) # [0.84147098,  0.90929743,  0.14112001, -0.7568025]
np.cos(a) # [0.54030231, -0.41614684, -0.9899925 , -0.65364362]

c = np.array([[False,False],[True,True]])


a = np.array([True, False, True, True])
b = np.array([False, False, True, False ])

print(a & b) # [False, False, True, False]
print(~a)    # [False, True, False, False]
print(a | b) # [True, False, True, True]

a = np.arange(4)
b = np.array([0,1,2.2,3.1])
c = np.array([[False,True],[True,True]])

np.allclose(a,b,atol=0.25) # Retuns True if an array has their elements equals in a tolerance range -> True
np.all(c,axis=0)           # Verify if all element on the axis is true or false -> [False,  True]

### LINEAL ALGEBRA
## For more https://docs.scipy.org/doc/numpy/reference/routines.linalg.html

a = np.ones((2,3))
b = np.full((3,2), 2)
c = np.identity(3)

np.matmul(a,b)   # Mulpiply matrixes -> AXB
np.linalg.det(c) # Hallar el determinante de una matriz cuadrada

### IMPORT AND EXPORT DATA

# This will transform the information in the .txt -> It will count the spaces and the commas as delimiters
array = np.genfromtxt('datos.txt', dtype=float, delimiter=',', skiprows=0)

# This will transform the array to file format that we prefer
a.tofile('datos.dat')


np.fromfile('datos.dat')