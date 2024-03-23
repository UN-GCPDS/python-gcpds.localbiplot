import os
from setuptools import setup

with open(os.path.join(os.path.dirname(__file__), 'README.md')) as readme:
    README = readme.read()

os.chdir(os.path.normpath(os.path.join(os.path.abspath(__file__), os.pardir)))

setup(
    name="gcpds-localbiplot",
    version='0.1',
    packages=["gcpds", "gcpds.localbiplot"],
    author="Jenniffer Carolina Triana Martinez",
    author_email="jectrianama@unal.edu.co",
    maintainer="Jenniffer Carolina Triana Martinez",
    maintainer_email="jectrianama@unal.edu.co",
    download_url='',
    install_requires=[ 'matplotlib', 'numpy', 'pandas', 'seaborn', 'scikit-learn', 'scipy'],
    scripts=[
    ],
    include_package_data=True,
    license='Simplified BSD License',
    description="",
    zip_safe=False,
    long_description=README,
    long_description_content_type='text/markdown',
    python_requires='>=3.7',

    #https://pypi.org/classifiers/
    classifiers=[
    ],
)
  
