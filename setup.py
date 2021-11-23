import setuptools


with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setuptools.setup(
    name="gc-ims-tools",
    version="0.1.0",
    url="https://github.com/Charisma-Mannheim/gc-ims-tools",
    author="Joscha Christmann",
    author_email="j.christmann@hs-mannheim.de",
    description="Analyze GC-IMS data.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    license="BSD 3-clause",
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "pandas",
        "scipy",
        "matplotlib",
        "seaborn",
        "h5py",
        "scikit-learn",
        "scikit-image"
    ],
    classifiers=[
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research"
    ]
)
