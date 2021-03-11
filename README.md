# ims

Python package to handle ion mobility spectrometry data from GAS Dortmund instruments.

## Installation

Optionally create a new conda python environment with:

`conda create --name ims python`

`conda activate ims`

To install the package clone the repository and navigate to the directory
you cloned it to.
Install requirements with:

`pip install -r ims/requirements.txt`

Install the package locally with:

`pip install -e ims`

Verify installation by running:

`conda list ims`

## Usage

`ims.Spectrum` is the main class and represents a single GC-IMS Spectrum
with the data matrix, retention and drift time coordinates.
Includes a name and time stamp for identification.
Contains all methods that can be done with a single spectrum like plotting
or cutting axis.

`ims.Dataset` coordinates a list of ims.Spectrum instances with labels.
Maps ims.Spectrum methods to all Spectra and contains methods
that require multiple Spectra such as alignment tools.

Import the module and access the classes and methods with:

```python
import ims

ims.Spectrum
ims.Dataset
```

Both classes use the read_mea method as a constructor:

```python
# reading a mea file returns an ims.Spectrum instance
sample = ims.Spectrum.read_mea("data/sample_file.mea")

# reads all mea files in a folder and returns an ims.Dataset instance
dataset = ims.Dataset.read_mea("data/folder_with_mea_files")
```

Use preprocessing methods like cutting drift and retention time
by chaining them together and plot the result.

```python
sample = ims.Spectrum.read_mea(path).cut_dt(5, 15).cut_rt(70, 500)
sample.plot()
```

![](sample.svg)


## License

[MIT](https://choosealicense.com/licenses/mit/)
