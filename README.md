# ims

## Installation

Optionally create a new conda python enviroment with:

`conda create --name ims python`

`conda activate ims`

To install the package locally clone the repository and navigate to the directory
you cloned it to. Install requirements with:

`pip install -r ims/requirements.txt`

Install the package with:

`pip install -e ims`

Verify installation by running:

`conda list ims`

## Usage

Import the module and access the classes and methods with:

```
import ims

ims.Spectrum
ims.Dataset
```

When the import could not be resolved, check if you are using the right enviroment.
