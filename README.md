# PointgroupPy

PointgroupPy is a general-purpose Python library for group-theoretical calculations for symmetry analysis of moleculars and crystals. It is designed to automate the generation of symmetry-adapted basis functions and the computation of Clebsch-Gordan coefficients for any finite group using the following core features:

## Features

### Generality and Automation
- **General for Any Point Group**: The library leverages the Burnside–Dixon–Schneider (BDS) method from computational representation theory to solve character tables, making it applicable to any finite group. 
- **Automatic Symmetry-Adapted Basis Functions**: By using projection operators from the group ring, the symmetry-adapted basis functions are constructed seamlessly. Users only need to specify the generators of the point group in matrix form.
- **Symmetric Clebsch-Gordan Coefficients**: The library calculates the Clebsch-Gordan coefficients for any finite group based on projection operators, making it useful for a variety of applications in physics and materials science.
- **Symmetry-adapted Polynomials in (x, y, z)**: The library supports finding polynomials in (x, y, z) that transform as irreducible representations of the given point group.
- **Symmetry-adapted Atomic Vibrational Modes**: The library supports identifying atomic vibration modes that transform as irreducible representations of the given point group. 

### User-Friendly Design
- Minimal input: Provide the point group generators in matrix form, and everything else is generated automatically.
- Intuitive: Designed to be straightforward and easy to integrate into materials modeling. 

## Applications
PointgroupPy can be used in:
- Group theory-related problems in condensed matter physics and chemistry.
- Automating symmetry analysis in molecular and crystal structures.
- Constructing irreducible representations and symmetry-adapted functions for theoretical and computational studies.

## Getting Started


### Installation
Clone the repository and install the dependencies:
```bash
git clone https://github.com/yaoluo/PointGroupPy
cd PointgroupPy
pip install . 
```
### Dependencies
PointgroupPy requires minimal Python libraries:
- `numpy`
- `matplotlib`
  

### Usage

1. **Input Generators**: Define the generators of the point group in matrix form.
2. **Run Calculations**:
   - Generate the character table.
   - Construct symmetry-adapted basis functions / invariant subspace. 
   - Compute Clebsch-Gordan coefficients for any finite group.

### Example: Automatic Solving of the Character Table
This example demonstrates automatic solving of the character table based solely on the matrix multiplication table (independent of the matrix representation).

```python
# case study of Oh point group 
import numpy as np 
from PointGroupPy import MatrixGroup
from PointGroupPy import R_X, R_Y, R_Z 

#Oh 
Id = np.eye(3)
th = np.pi/2
Rx90 = R_X(np.pi/2)
Ry90 = R_Y(np.pi/2)
Rz90 = R_Z(np.pi/2)
sigma_d = np.array([[0,1,0],[1,0,0],[0,0,1]])
Oh = MatrixGroup(generator = [Id, Rx90, Ry90, Rz90, sigma_d])
print(f'|Oh| = {Oh.nG}')
if Oh.nG!=48:
   raise ValueError('# of elements in Oh is inconsistent')

Oh.constructMultiplicationTable()
Oh.conjugacy_class()
if Oh.nClass!=10:
   raise ValueError('# of conjugacy classes in Oh is inconsistent')
print('Oh ConjClass = ',Oh.ConjClass)
# 
from PointGroupPy import character_solver
ChiSolver = character_solver(Oh.MultilicationTable, Oh.ConjClass)
chi_table = ChiSolver.solve()
print('Character table of Oh = ')
for i in range(Oh.nClass):
   print(" ".join(f"{x:10.2f}" for x in chi_table[:,i]))  # Format numbers to 2 decimal places
```

## Reference
Burnside–Dixon–Schneider (BDS) method follows this [lecture note](http://www.math.rwth-aachen.de/~hiss/Presentations/Galway08_Lec1.pdf). 


## Contributing
Contributions are welcome! Please feel free to fork this repository and submit pull requests.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.
