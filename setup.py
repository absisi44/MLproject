import email
from gettext import install
from setuptools import setup, find_packages
setup(
    name='mlproject',
    version='0.0.1',
    author='YASSIR',
    author_email='absisi2009@gmail.com',
    packages=find_packages(),
    install_requires=['numpy', 'pandas', 'scikit-learn', 'matplotlib', 'seaborn'],
    
)