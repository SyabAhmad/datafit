from setuptools import setup, find_packages




setup(
    name='datafit',
    version='0.2023.11',
    description='This is a Python package that automates your data preprocessing',
    # long_description=longDescription,
    long_description = """
## DataFit
DataFit is a python package developed for automating data preprocessing.
#### Note ```This Package is under development and is open source.```

This package is developed by Syed Syab and Hamza Rustam for the purpose of Final Year Project at University of Swat.
our information is given below

```
About Project:
    DataFit is a python package developed for automating data preprocessing.
    Project initilization data: 01/OCT/2023
    Project Finilization Data: 01/Dec/2023 (Expected)

Team Member:
    Syed Syab: Student (Me) [github.com/SyabAhmad] [linkedin.com/SyedSyab]
    Hamza Rustam: Student
    ================================
    Professor Naeem Ullah: Supervisor 
```
This Package is desinged in a user-friendly manner which means every one can use it.

The main functionality of the package is to just automate the data pre-processing step, and make it easy for machine learning engineers or data scientist.

Current Functionality of the package is:
```
    Function:
        displaying information
        Handling Null Value
        Delete Multiple Columns
        Handling Categorical Values
        Normalization
        Standardization
        Extract Numeric Values
        Tokenization
```

### To use the package
```commandline
pip install datafit
```
To use this package it's quit simple, just import it like pandas and then use it.
```python
import datafit as df
# to check information of the data
df.information(data)
```

To categorize the data
```python
import datafit as df

df.handleCategoricalValues(data,["column1","column2"])
```
if you want to not mention the columns name an do it for all columns then simply type **None** inplace of columns names.
```python
import datafit as df

df.handleCategoricalValues(data,None)
```

To Extract numerical values from the columns

```python
import datafit as df

df.extractValues(data,["columns1", "columns2"])
```


**Note Again:** This package is uder development. 
if it touches your heart do share it and follow me on **github** [github.com/SyabAhmad] and **linkedin** [linkedin.com/in/SyedSyab]for mote intersting updates


""",
    long_description_content_type='text/markdown',  # Specify the type of content as Markdown
    author='Syed Syab & Hamza Rustam',
    author_email='syab.se@hotmail.com',
    url='https://github.com/SyabAhmad/datafit',
    license='MIT',

    packages=find_packages(),
    install_requires=[
        'numpy>=1.0',
        'pandas>=1.0',
        'scikit-learn',

    ],

    # Entry points for command-line scripts if applicable
    entry_points={
        'console_scripts': [
            'my_script = datafit.datafit:information',
        ],
    },

    # Other optional metadata
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        # Add more classifiers as appropriate
    ],
)
