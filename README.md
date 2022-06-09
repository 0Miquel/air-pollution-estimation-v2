# Air pollution estimation

PM2.5 estimation using image analysis and weather data from Shanghai and Beijing.

## Data

Dataset used for Beijing and Shanghai imagery can be accessed through this [link](https://figshare.com/articles/figure/Particle_pollution_estimation_based_on_image_analysis/1603556).

Ground truth data for PM2.5 measures has been extracted from the [Mission China](http://www.stateair.net/web/historical/) of the U.S Department of State Air Quality Monitoring Program,
such mission provides historical hourly PM2.5 data from Shanghai, Beijing, Chengdu, Guangzhou and Shenyang.

Weather data has been used together with images to achieve better results, as weather data is highly correlated with PM2.5 estimations. Such data has been extracted from the 
[visualcrossing](https://www.visualcrossing.com/weather/weather-data-services) website

## Methods

Images are processed using the [Dark Channel Prior](https://github.com/He-Zhang/image_dehaze) algorithm to extract its transmission, which can be used for 
PM2.5 estimation.

Such transmission is fit into our model, which consists of a convolutional neural network model (ResNet18) together with a last fully connected layer that
considers the image estimation and the weather data and provides the PM2.5 estimation.