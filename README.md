# Analysis of  the New York City Property Prices Dataset

## Goal

The goal of this project is to predict the selling price of real estate properties in New York. For this propose the data on sales (containing the regular features such as total square feet, address, ...) way extended by visual features extracted from areal photos of the city. Further, data about the location of Wifi hotspots is introduced. The predictive power of using these additional features is examined. 

## Data

The housing sales data was acquired from the public NYC Geodatabase. This dataset provides geocoded data on real estate sales in New York City from the year 2015. This Geodatabase is based on annual sales reports collected by the New York City Department of Finance. All records are provided with longitude and latitude coordinates using the NAD83 geodetic datum as their reference point. Next to the location, the data provides information on the properties as well as the selling price. In total, the dataset contains 30 attributes including information on both sale and property level. [(More Information)](https://geo.nyu.edu/catalog/nyu-2451-34678)Explanatory details of this dataset can be found [here](https://www1.nyc.gov/assets/finance/downloads/pdf/07pdf/glossary_rsf071607.pdf). The image data source is from the New York state orthoimagery program. [(More Information)](http://gis.ny.gov/gateway/orthoprogram/ortho_options.htm) The data for the wifi hotspot data can be found on NYC open data [(More Information)](https://data.cityofnewyork.us/Social-Services/NYC-Wi-Fi-Hotspot-Locations/a9we-mtpn).


## Develop

* Clone the repository: `git clone [repo-url]`

* Add the dataset files to your local repository.

### Automatic Output Clearing

Using the git hook from (https://github.com/kynan/nbstripout)

`pip install nbstripout nbconvert`

`nbstripout --install`