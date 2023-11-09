from setuptools import setup, find_packages

setup(
    name='datafit',  
    version='0.2023.1',  
    description='This is Python Package that automate your data preprocessing. visit [] to learn more about this package.',
    author='Syed Syab & Hamza Rustam',
    author_email='syab.se@hotmail.com || hs4647213@gmail.com',
    url='https://github.com/SyabAhmad/yourpackage',
    license='MIT',  #license

    packages=find_packages(),  # Automatically include all Python packages
    install_requires=[
        'numpy>=1.0',  # Replace with your package dependencies
        'pandas>=1.0',
        'scikit-learn',
        're',
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
