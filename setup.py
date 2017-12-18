import setuptools
from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy
import os
import imp

if os.name =='nt' :
    ext_modules=[
        Extension("keras_retinanet.cython_utils.nms",
            sources=["keras_retinanet/cython_utils/nms.pyx"],
            #libraries=["m"] # Unix-like specific
            include_dirs=[numpy.get_include()]
        ),        
        Extension("keras_retinanet.cython_utils.cy_yolo2_findboxes",
            sources=["keras_retinanet/cython_utils/cy_yolo2_findboxes.pyx"],
            #libraries=["m"] # Unix-like specific
            include_dirs=[numpy.get_include()]
        ),
        Extension("keras_retinanet.cython_utils.cy_yolo_findboxes",
            sources=["keras_retinanet/cython_utils/cy_yolo_findboxes.pyx"],
            #libraries=["m"] # Unix-like specific
            include_dirs=[numpy.get_include()]
        )
    ]

elif os.name =='posix' :
    ext_modules=[
        Extension("keras_retinanet.cython_utils.nms",
            sources=["keras_retinanet/cython_utils/nms.pyx"],
            libraries=["m"], # Unix-like specific
            include_dirs=[numpy.get_include()]
        ),        
        Extension("keras_retinanet.cython_utils.cy_yolo2_findboxes",
            sources=["keras_retinanet/cython_utils/cy_yolo2_findboxes.pyx"],
            libraries=["m"], # Unix-like specific
            include_dirs=[numpy.get_include()]
        ),
        Extension("keras_retinanet.cython_utils.cy_yolo_findboxes",
            sources=["keras_retinanet/cython_utils/cy_yolo_findboxes.pyx"],
            libraries=["m"], # Unix-like specific
            include_dirs=[numpy.get_include()]
        )
    ]

else :
    ext_modules=[
        Extension("keras_retinanet.cython_utils.nms",
            sources=["keras_retinanet/cython_utils/nms.pyx"],
            libraries=["m"] # Unix-like specific
        ),        
        Extension("keras_retinanet.cython_utils.cy_yolo2_findboxes",
            sources=["keras_retinanet/cython_utils/cy_yolo2_findboxes.pyx"],
            libraries=["m"] # Unix-like specific
        ),
        Extension("keras_retinanet.cython_utils.cy_yolo_findboxes",
            sources=["keras_retinanet/cython_utils/cy_yolo_findboxes.pyx"],
            libraries=["m"] # Unix-like specific
        )
    ]

setuptools.setup(
    name='keras-retinanet',
    version='0.0.1',
    url='https://github.com/fizyr/keras-retinanet',
    author='Hans Gaiser',
    author_email='h.gaiser@fizyr.com',
    maintainer='Hans Gaiser',
    maintainer_email='h.gaiser@fizyr.com',
    packages=setuptools.find_packages(),
    install_requires=['keras', 'keras-resnet', 'six'],
    ext_modules = cythonize(ext_modules)
)
