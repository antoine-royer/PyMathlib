# --------------------------------------------------
# PyMathlib (version 1.0.0)
# by Charlotte Thomas and Sha-chan~
# Last version released on the July 18 2020
#
# code provided whith licence :
# GNU General Public Licence v3.0
# --------------------------------------------------

"""
  PyMathlib is a librairie of vectorial and matricial manipulations.
  This librairie is oriented-object, so you can use Vector and Matrix like list or str objects.
  This librairie was made by Charlotte THOMAS and Sha-chan~.
  Please see the documentation for more information : https://github.com/Shadow15510/PyMathlib/blob/master/README.md
"""
__version__ = "1.0.0"

from math import acos, degrees, sqrt

class Vector:
    """
    The Vector objets was designed for support n dimensions.
    Such as u = Vector(1, 1) creates a two-dimensional vector, or even u = Vector(1, 2, 3, 4, 5) creates a five-dimensional vector.
    Don't forget the dot for use the handlings, for instance :
    
      >>> my_vector1 = Vector(1, 0)
      >>> my_vector2 = Vector(0, 1)
      >>> my_vector1.angle(my_vector2)
      90.0

    The vectors supports the basic operation + and - for the addition and substraction between two vectors and * and / for the multiplication and division between a vector and a real number, they also work for element-wise multiplication between two vectors.
    Vectors are printable and return a string representation of their coordonates.
    Theys also are subscriptable : `my_vector[0]` returns the first coordinate.
    """
    def __init__(self, *coord):
        self.coord = list(coord)
        self.dim = len(coord)

    def __add__(self, vec: "Vector"):
        if vec.dim != self.dim:
            raise ArithmeticError("The dimensions are incompatible")
        return Vector(*map(lambda x,y: x+y, self.coord, vec.coord))

    def __sub__(self, vec: "Vector"):
        if vec.dim != self.dim:
            raise ArithmeticError("The dimensions are incompatible")
        return Vector(*map(lambda x,y: x-y, self.coord, vec.coord))

    def __mul__(self, x):
        if isinstance(x, (float, int)):
            return Vector(*map(lambda y: x*y, self.coord))
        if isinstance(x, Vector):
            return Vector(*map(lambda y,z: y*z, self.coord, x.coord))
        else:
            raise TypeError("The wrong types were given")

    def __truediv__(self, x):
        if isinstance(x, (float, int)):
            return Vector(*map(lambda y: y/x, self.coord))
        if isinstance(x, Vector):
            return Vector(*map(lambda y,z: y/z, self.coord, x.coord))
        else:
            raise TypeError("The wrong types were given")

    def __str__(self):
        return str(tuple(self.coord))

    def __abs__(self):
        return sqrt(sum(list(map(lambda i: i**2, self.coord))))

    def __getitem__(self, index):
        return self.coord[index]

    __radd__ = __add__
    __rmul__ = __mul__

    def unit(self):
        """Returns the unitary vector."""
        a = abs(self)
        return Vector(*map(lambda x: x/a, self.coord))
        
    def dotP(self, vec: "Vector"):
        """Returns the dot product."""
        return sum(list(map(lambda x,y: x*y, self.coord, vec.coord)))
      
    def crossP(self, vec: "Vector"):
        """Returns the cross product. (Only available for two three-dimensions vectors.)"""
        if self.dim == 3 and vec.dim == 3: return Vector(self.z*vec.y - self.y*vec.z, self.x*vec.z - self.z*vec.x, self.y*vec.x - self.x*vec.y)

    def det(self, *vec):
        """Returns the vectors's determinant. You can pass several vectors without any problem. (Please take care to pass n vectors to n dimensions.)"""
        return Matrix([self.coord] + [i.coord for i in vec]).det()
      
    def colinear(self, vec: "Vector"):
        """Tests the colinearty between the vectors. Return a boolean."""
        return not self.det(vec)

    def angle(self, vec: "Vector"):
        """Returns the angle in degrees between the vectors."""
        return round(degrees(acos(self.dotP(vec) / (abs(self)*abs(vec)))), 2)


class Matrix:
    """
    Matrix was designed for support n * m dimensions, so you don't have to take care at dimensions.
    If an error occured, it's because the dimensions doesn't allow to calculate what you want. For exemple, the matrix's determinant is only available with squarred matrix.
    As well as for Vector, don't forget the dot for use these handlings.
    
    For initialise a matrix follow this scheme M = Matrix([1, 2], [3, 4]). You can put everything you want, just take care to have one row per argument :
    Matrix([0]) for a matrix with one row and one column.

    When you use write_row or write_column, please take care, the new_row or new_column argument is a list object, not a matrix.

    The matrices supports basic operation, + and - for addition and substraction between two matrices and * and / for multiplication and division between a matrix and a real number, they also work for multiplication between two matrices.

    The matrices are printable and return a string representation of the column and rows.
    They also are subscriptables.
    """
    def __init__(self, *row):
        self.content = [i for i in row]

    def __getitem__(self, index):
        return self.content[index]

    def __add__(self, matrix: "Matrix"):
        if self.get_dim() != matrix.get_dim():
            raise ArithmeticError("The dimensions are incompatible")
        return Matrix(*[[self[i][j] + matrix[i][j] for j in range(len(self[0]))] for i in range(len(self.content))])
  
    def __sub__(self, matrix: "Matrix"):
        if self.get_dim() != matrix.get_dim():
            raise ArithmeticError("The dimensions are incompatible")
        return Matrix(*[[self[i][j] - matrix[i][j] for j in range(len(self[0]))] for i in range(len(self.content))])
 
    def __mul__(self, x):
        if isinstance(x, (float, int)):
            return Matrix(*[[self[i][j] * x for j in range(len(self[0]))] for i in range(len(self.content))])
        elif isinstance(x, Matrix):
            return Matrix(*[[sum([self[j][i] * x[i][j] for i in range(len(self[0]))]) for k in range(len(x[0]))] for j in range(len(self.content))])
        else:
            raise TypeError("The wrong types were given")

    def __truediv__(self, x):
        if isinstance(x, (float, int)):
            return Matrix(*[[self[i][j] / x for j in range(len(self[0]))] for i in range(len(self.content))])
        elif isinstance(x, Matrix):
            return self * x.inverse()
        else:
            raise TypeError("The wrong types were given")
  
    def __str__(self):
        return "\n".join(map(str, self.content))

    def __len__(self):
        return len(self.content)

    __radd__ = __add__
    __rmul__ = __mul__

    def __copy(self):
        return Matrix(*[[nb for nb in row] for row in self.content])
  
    def __gauss_jordan_elimination(self): # return a tuple (mat ref, determinant)
        def get_pivot(mat, j):
            pivot_row, pivot = j, abs(mat[j][j])
            
            for i in range(j + 1, len(mat)):
                if not pivot:
                    pivot_row = i
                    pivot = abs(mat[i][j])
            
            if not pivot: return -1
            else: return pivot_row

        mat = self.__copy()
        n = len(mat)

        for j in range(n):
            i = get_pivot(mat, j)
            if i == -1: raise MathError("This matrix cannot be inverted")

            mat.switch_row(i, j)
            pivot = mat[j][j]

            for i in range(j + 1, n):
                mat.transvection(i, j, -mat[i][j] / pivot)

        det = 1
        for i in [mat[i][i] for i in range(n)]: det *= i

        return mat, det


    def __s_mat(self, jmp_i, jmp_j):
        return Matrix(*[[self.content[j][i] for i in range(len(self.content)) if i != jmp_i] for j in range(len(self.content[0])) if j != jmp_j])  

    def get_dim(self):
        """Returns the dimension of the matrix. The results is a tuple : (row, column)."""
        return len(self.content), len(self[0])
    
    def augment(self, mat: "Matrix"):
        """Allows to augment the size of the matrix by adding another matrix to the first one."""
        [self[i].append(mat[i][j]) for j in range(len(mat[0])) for i in range(len(mat.content))]
        
    def sub(self, row_st, column_st, row_ed, column_ed):
        """Returns a sub-matrix between the given limits."""
        return Matrix(*[[self[i][j] for j in range(column_st, column_ed+1)] for i in range(row_st, row_ed+1)])
  
    def det(self):
        """Returns the matrix's determinant."""
        return self.__gauss_jordan_elimination()[1]

    def ref(self):
        """Returns the row echelon form of the matrix. (Calculate by the Gauss-Jordan elimination.)"""
        return self.__gauss_jordan_elimination()[0]

    def rref(self):
        """Returns the reduces row echelon form of the matrix."""
        mat = self.__gauss_jordan_elimination()[0]
        for j in range(len(mat)):
            mat.dilatation(j, 1 / mat[j][j])
        return mat

    def trace(self):
        """Returns the matrix's trace."""
        rslt = 1
        for i in range(len(self.content)): rslt *= self[i][i]
        return rslt
        
    def transpose(self):
        """Returns the transpose matrix."""
        return Matrix(*[[self[i][j] for i in range(len(self.content))] for j in range(len(self[0]))])
  
    def comat(self):
        """Returns the co-matrix."""
        return Matrix(*[[(-1) ** (i + j) * self.__s_mat(i, j).det() for i in range(len(self.content))] for j in range(len(self[0]))])

    def inverse(self):
        """Returns the inverse matrix. (Please take care to the determinant.)"""
        return self.comat().transpose() * (1/self.det())

    def transvection(self, i, j, c):
        """Modify the matrix with a transvection."""
        for k in range(len(self[0])): self[i][k] += c * self[j][k]

    def dilatation(self, i, c):
        """Dilates a column."""
        for k in range(len(self[0])): self[i][k] *= c

    def switch_row(self, row_1: "int", row_2: "int"):
        """Reverses the two rows."""
        self[row_1][:], self[row_2][:] = self[row_2][:], self[row_1][:]

    def switch_column(self, column_1: "int", column_2: "int"):
        """Reverses the two columns."""
        for i in range(len(self.content)): self[i][column_1], self[i][column_2] = self[i][column_2], self[i][column_1]
    
    def write_row(self, index: "int", new_row: "list"):
        """Replaces the index-row by the new row."""
        self[index][:] = new_row[:]

    def write_column(self, indef: "int", new_column: "list"):
        """Replaces the index-column by the new column."""
        for i in range(len(self.content)): self[i][index] = new_column[i][0]

    def solve(self, *solution):
        """
        Solves linear systems described by Matrix.
        For exemple, we have this system :

        2x + y = 0
        x - 2y = 10

        we have a matrix :

          >>> system = Matrix([2, 1], [1, -2])

        and to solve them :

          >>> system.solve(0, 10)
          [2.0, -4.0]

        So x = 2 and y = -4
        """
        mat = self.__copy()
        mat.augment(Matrix(*[[i] for i in solution]))
        mat = mat.ref()
        var = [0 for i in range(len(solution))]
        for i in range(1, mat.get_dim()[0] + 1):
            var[-i] = (mat[-i][-1] - sum(map(lambda x,y:x*y, var, mat[-i]))) / mat[-i][-(i+1)]
    
        return var
     

class Polynom:
    """
    An implementation of Polynoms.
    To declare a polynom, you just have to precise its coefficients :
      >>> P = Polynom(1, 2, 3)
    will create a polynome : 1 + 2X + 3X^2

    (with powers in ascending order)
    """
    def __init__(self, *coeff):
        self.coeff = list(coeff)
        self.deg = len(coeff)

    def __str__(self):
        answer = []
        for power, value in enumerate(self.coeff):
            if value:
                if not power: answer.append(f"{value}")
                elif power == 1:answer.append(f"{value}·X")
                else: answer.append(f"{value}·X^{power}") 
        answer.reverse()
        return " + ".join(answer)

    def __call__(self, x):
        result = 0
        for power, value in enumerate(self.coeff):
            result += value * (x ** power)
        return result

    def __add__(self, P):
        new_coeff = []
        for n in range(max(self.deg, P.deg)):
            if self.deg < n: new_coeff.append(P.coeff[n])
            elif P.deg < n: new_coeff.append(self.coeff[n])
            else: new_coeff.append(self.coeff[n] + P.coeff[n])
        return Polynom(*new_coeff)

    __radd__ = __add__

    def __sub__(self, P):
        new_coeff = []
        for n in range(max(self.deg, P.deg)):
            if self.deg < n: new_coeff.append(-P.coeff[n])
            elif P.deg < n: new_coeff.append(self.coeff[n])
            else: new_coeff.append(self.coeff[n] - P.coeff[n])
        return Polynom(*new_coeff)

    def __rsub__(self, P):
        new_coeff = []
        for n in range(max(self.deg, P.deg)):
            if self.deg < n: new_coeff.append(P.coeff[n])
            elif P.deg < n: new_coeff.append(-self.coeff[n])
            else: new_coeff.append(self.coeff[n] + P.coeff[n])
        return Polynom(*new_coeff)

    def derivative(self):
        """Returns the derivative."""
        new_coeff = [0]
        for index, value in enumerate(self.coeff):
            if index:
                new_coeff[index - 1] = value * index
                new_coeff.append(0)
        return Polynom(*new_coeff)


def identity(n: "int"):
  """Returns an identity matrix of order n. (This is a Matrix object.)"""
  return Matrix(*[[int(i == j) for i in range(n)] for j in range(n)])
