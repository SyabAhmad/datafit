```python
from setuptools import setup, find_packages

longDescription = """
## DataFit
DataFit is a Python package developed for automating data preprocessing.

**Note:** This package is under development and is open source.

This package is developed by Syed Syab and Hamza Rustam for the purpose of the Final Year Project at the University of Swat. Our information is given below:

```
About Project:
    DataFit is a Python package developed for automating data preprocessing.
    Project initialization date: 01/OCT/2023
    Project Finalization Date: 01/Dec/2023 (Expected)

Team Member:
    Syed Syab: Student (Me) [GitHub: SyabAhmad] [LinkedIn: SyedSyab]
    Hamza Rustam: Student
    ================================
    Professor Naeem Ullah: Supervisor 
```

This Package is designed in a user-friendly manner, which means everyone can use it.

The main functionality of the package is to automate the data pre-processing step and make it easy for machine learning engineers or data scientists.

Current Functionality of the package includes:
```
    Function:
        Displaying information
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

To use this package, it's quite simple. Just import it like pandas and then use it.

```python
import datafit as df
# To check information of the data
df.information(data)
```

To categorize the data:

```python
import datafit as df

df.handleCategoricalValues(data, ["column1", "column2"])
```

If you want to not mention the columns' names and do it for all columns, then simply type **None** in place of column names.

```python
import datafit as df

df.handleCategoricalValues(data, None)
```

To extract numerical values from the columns:

```python
import datafit as df

df.extractValues(data, ["columns1", "columns2"])
```

**Note Again:** This package is under development. If it touches your heart, do share it and follow me on **GitHub** [GitHub: SyabAhmad] and **LinkedIn** [LinkedIn: SyedSyab] for more interesting updates.
"""

setup(
    name='datafit',
    version='0.2023.15',
    description='This is a Python package that automates your data preprocessing',
    long_description=longDescription,
    long_description_content_type='text/markdown',  # Specify the type of content as Markdown
    author='Syed Syab & Hamza Rustam',
    author_email='syab.se@hotmail.com',
    url='https://github.com/SyabAhmad/datafit',
    keywords=['python', 'machine learning', 'data science', 'data', 'preprocessing', 'AI'],
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
