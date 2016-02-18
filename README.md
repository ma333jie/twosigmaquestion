# twosigmaquestion
2sigmaDataChallenge


1.rolling sale excel sheet for each Borough in to a folder. For example, the rolling sales for 2015 is in a folder named "2015Detial". 
Excel sheets in 2015 is from http://www1.nyc.gov/site/finance/taxes/property-rolling-sales-data.page. The rest years are from http://www1.nyc.gov/site/finance/taxes/property-annualized-sales-update.page 


2.Pluto data is stored in a folder named "nyc_pluto". 
The data is download form http://www1.nyc.gov/site/planning/data-maps/open-data.page


3.Also, The dynamic map question 6 in the 'NYC_property_market.ipynb' and the interactive 
map in 'NYC_porperty_advance'. Since, the free arcgis onine only allow user to upload up to 1000 points, only some of the zip code contains the configuration pop up. All the area in the staten island has the interactive pop up showing the plot of monthly total sale-value for that points. 


Here are some question I hope to answer:

Suggested research threads:
 
1.       What months are most active for property sales? Does this vary by Borough or by Building-Class?
2.       What are 2014’s hottest residential neighborhoods, as measured by year-over-year growth in total sale-value and total sale-count?
3.       Does the residential market prefer new-construction (buildings built within 2years of the sale date)? How does 2014 compare with previous years in this regard?
4.       Please produce a timeseries plot showing monthly YoY sales-growth at various price tiers.
a.       what has the trend been like for sales of properties priced <$3M?
b.      modify this to show stats by neighborhood & price-tier; how do sales <$3M compare for Brooklyn Heights vs. Bedford-Stuyvesant?
5.       What neighborhoods have seen the highest turnover in 2014 (as measured by time-between sales of the same property)
6.       Can you visualize YoY sales-growth by neighborhood (or ZIP code) on a map? How would you add the element of time to this map?
 
Advanced (requires joining the DOF rolling sales w/ the Dept. of City Planning’s PLUTO dataset)

1.       What does the distribution of sale-price look like for 2014 sales of residential properties having more than 6 floors?
2.       What is the total OfficeArea sold in 2014?
3.       Build an interactive map showing median 2014 sale-price by neighborhood and/or ZIP-code for residential properties < $15M. Clicking on a neighborhood or ZIP will render a timeseries plot of monthly total sale-value for that polygon. It would be great to be able to see a finer-grained density of sales (as points or polygons) for the individual properties sold.
