from setuptools import setup, Extension
import pybind11
import os

# Cari path include OpenCV (Sesuaikan jika lu pakai Windows lokal, di Docker ini aman)
# Asumsi di sistem/Docker, OpenCV header ada di /usr/include/opencv4
opencv_include = "/usr/include/opencv4"
opencv_libs = ["opencv_core", "opencv_imgproc"]

ext_modules = [
    Extension(
        "scanner_cpp",
        ["scanner_engine.cpp"],
        include_dirs=[pybind11.get_include(), opencv_include],
        libraries=opencv_libs,
        language="c++"
    ),
]

setup(
    name="scanner_cpp",
    version="1.0",
    ext_modules=ext_modules,
)