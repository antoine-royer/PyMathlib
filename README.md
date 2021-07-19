# PyMathlib

## General presentation

### Presentation

PyMathlib is a librairie of vectorial and matricial manipulations. This librairie is oriented-object, so you can use `Vector` and `Matrix` like `list` or `str` objects. This librairie was made by Charlotte THOMAS and Sha-Chan~.

There is two file : `vecmat.py` is the complete librairie for computer and `vecmat_nw.py` is the same librairie but without some functions like `solve`. This lighter version is specially design for the Numworks calculators but you can execute it on other calculators like Casio Graph 90+E or Graph 35+E II.

### Licence

This code was provided with licence GNU General Public Licence v3.0

## Vector's manipulations

### Generalities

The `Vector` objets was designed for support n dimensions. Such as `u = Vector(1, 1)` creates a two-dimensional vector, or even `u = Vector(1, 2, 3, 4, 5)` creates a five-dimensional vector. (Don't forget the dot for use the handlings, for instance : `my_vector.unitV()`.)

### Vectorial manipulations available

#### Handlings on the vector itself

 - Vectors are printable and return a string representation of their coordonates.
 - Theys also are subscriptable : `my_vector[0]` returns the first coordinate.
 - `Vector.unitV()` : Returns the unitary vector.

#### Basical mathematical operations between two vectors

The vectors supports the basic operation `+` and `-` for the addition and substraction between two vectors and `*` and `/` for the multiplication and division between a vector and a real number, they also work for element-wise multiplication between two vectors.

#### Advanced vector manipulations

 - `Vector.dotP(vec)` : Returns the dot product.
 - `Vector.crossP(vec)` : Returns the cross product. (Only available for two three-dimensions vectors.)
 - `Vector.det(*vec)` : Returns the vectors's determinant. You can pass several vectors without any problem. (Please take care to pass n vectors to n dimensions.)
 - `Vector.colinear(vec)` : Tests the colinearty between the vectors. Return a boolean.
 - `Vector.angle(vec)` : Returns the angle in degrees between the vectors.

## Matrix's manipulations

### Generalities

`Matrix` was designed for support n * m dimensions, so you don't have to take care at dimensions. If an error occured, it's because the dimensions doesn't allow to calculate what you want. For exemple, the matrix's determinant is only available with squarred matrix. As well as for `Vector`, don't forget the dot for use these handlings.

For initialise a matrix follow this scheme `M = Matrix([1, 2], [3, 4])`. You can everything you want, just take care to have one row per argument : `Matrix([0])` for a matrix with one row and one column.

When you use `write_row` or `write_column`, please take care, the `new_row` or `new_column` argument is a `list` object, not a matrix.

### Matricial manipulations available

#### Handlings on the matrix itself

 - The matrices are printable and return a string representation of the column and rows.
 - They also are subscriptable.
 - `Matrix.get_dim()` : Returns the dimension of the matrix. The results is a tuple : `(row, column)`.
 - `Matrix.switch_row(row_1, row_2)` : Reverses the two rows.
 - `Matrix.switch_column(column_1, column_2)` : Reverses the two columns.
 - `Matrix.write_row(index, new_row)` : Replaces the index-row by the new row.
 - `Matrix.write_column(index, new_column)` : Replaces the index-column by the new column.

#### Basical mathematical operations

The matrices supports basic operation, `+` and `-` for addition/substraction between two matrices and `*` and `/` for multiplication and division between a matrix and a real number, they also work for multiplication between two matrices.

#### Advanced matricial manipulations

 - `Matrix.augment(mat)` : Allows to augment the size of the matrix by adding another matrix to the first one.
 - `Matrix.sub(row_st, column_st, row_ed, column_ed)` : Returns a sub-matrix between the given limits.
 - `Matrix.det()` : Returns the matrix's determinant.
 - `Matrix.tranpose()` : Returns the transpose matrix.
 - `Matrix.comat()` : Returns the co-matrix.
 - `Matrix.inverse()` : Returns the inverse matrix. (Please take care to the determinant.)
 - `Matrix.ref()` : Returns the row echelon form of the matrix. (Calculate by the Gauss-Jordan elimination.)
 - `Matrix.rref()` : Returns the reduced row echelon form
 - `Matrix.solve(*solution)` : Solves the linear system describe by Matrix. [Exemple](https://github.com/Shadow15510/PyMathlib#Linear-system-solver)

## Polynoms manipulations

### Generalities

As for Vectors and Matrix, Polynoms was made with OOP. This object support the addition, substraction, and can be print on screen.

### Basic manipulations

You can evaluate a polynom on a value by simply call them. Please see : [Exemple]()

### Advanced manipulations

`Polynom.derivative()` : Returns the derivative of the polynom.

## Other manipulations

### Generalities

These handlings are not concerned by the oriented-object code.

### Available functions

 - `identity(n)` : Returns an identity matrix of order n. (This is a `Matrix` object.)
 - `abs(vec)` : Returns the vec's norm.

## Exemple

### Matrix

#### Linear system solver

If we have a linear system like : 

```
2x + y = 0
x - 2y = 10
```

We can transpose this system into a matrix : 

`>>> system = Matrix([2, 1], [1, -2])`

Then, we enter : 

`>>> system.solve(0, 10)`

Here, 0 and 10 are the equations's solutions. In this case, the solver returns :

`[2.0, -4.0]`

So now, we know that the system's solutions are x = 2 and y = -4.

### Polynom

#### Evaluation

We can create a Polynom object :

`>>> P = Polynom(1, 2, 3)`

And to evaluate it on a value (for exemple, 0) : 

```
>>> P(0)
1
```
