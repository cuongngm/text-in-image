import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ultocr",
    version="0.1.0",
    author="cuongngm",
    author_email="cuonghip0908@gmail.com",
    description="text detection + text recognition",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/cuongngm/text-in-image",
    packages=setuptools.find_packages(),
    install_requires=[
        'Polygon3',
        'pyclipper',
        'imgaug',
        'distance'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)