from distutils.core import setup
from Cython.Build import cythonize
import numpy
setup(name='Hello world app',
      ext_modules=cythonize("test.pyx"), include_dirs=[numpy.get_include()])
