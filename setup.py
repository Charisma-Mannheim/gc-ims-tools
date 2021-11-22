import setuptools


with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setuptools.setup(
    name="ims",
    version="0.1.0",
    url="https://github.com/Charisma-Mannheim/ims",
    author="Joscha Christmann",
    author_email="j.christmann@hs-mannheim.de",
    description="Analyze GC-IMS data.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    license="LICENSE.txt",
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.19.5",
        "pandas>=1.3.4",
        "scipy>=1.7.1",
        "matplotlib>=3.4.3",
        "seaborn>=0.11.2",
        "h5py>=3.1.0",
        "scikit-learn>=1.0",
        "scikit-image>=0.18.3"
    ],
    classifiers=[
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Researchers"
    ]
)
