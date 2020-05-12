#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 11 13:25:47 2020

@author: sandeepkompella
"""

import turicreate
sales = turicreate.SFrame('home_data.sframe/')
train_data,test_data = sales.random_split(.8,seed=0)
# Let's compute the mean of the House Prices in King County in 2 different ways.
prices = sales['price'] # extract the price column of the sales SFrame -- this is now an SArray

# recall that the arithmetic average (the mean) is the sum of the prices divided by the total number of houses:
sum_prices = prices.sum()
num_houses = len(prices) # when prices is an SArray len() returns its length
avg_price_1 = sum_prices/num_houses
avg_price_2 = prices.mean() # if you just want the average, the .mean() function
print("average price via method 1: " + str(avg_price_1))
print("average price via method 2: " + str(avg_price_2))

# if we want to multiply every price by 0.5 it's a simple as:
half_prices = 0.5*prices
# Let's compute the sum of squares of price. We can multiply two SArrays of the same length elementwise also with *
prices_squared = prices*prices
sum_prices_squared = prices_squared.sum() # price_squared is an SArray of the squares and we want to add them up.
print("the sum of price squared is: " + str(sum_prices_squared))

def simple_linear_regression(input_feature, output):
    # compute the sum of input_feature and output
    sum_y = output.sum()
    sum_x = input_feature.sum()
    sum_yx = (output*input_feature).sum()
    sum_xx = (input_feature**2).sum()
    n = float(len(output))
    slope = (sum_yx - ((sum_y*sum_x)/n))/(sum_xx - ((sum_x*sum_x)/n))
    intercept = (sum_y/n) - slope*(sum_x/n)  
    return (intercept, slope)

test_feature = turicreate.SArray(range(5))   
test_output = turicreate.SArray(1 + 1*test_feature)
(test_intercept, test_slope) =  simple_linear_regression(test_feature, test_output)
print("Intercept: " + str(test_intercept))
print("Slope: " + str(test_slope))

sqft_intercept, sqft_slope = simple_linear_regression(train_data['sqft_living'], 
                                                      train_data['price'])

print("Intercept: " + str(sqft_intercept))
print("Slope: " + str(sqft_slope))

def get_regression_predictions(input_feature, intercept, slope):
    # calculate the predicted values:
    predicted_values = intercept + (input_feature*slope)
    return predicted_values

my_house_sqft = 2650
estimated_price = get_regression_predictions(my_house_sqft, sqft_intercept, sqft_slope)
print("The estimated price for a house with %d squarefeet is $%.2f" % (my_house_sqft, estimated_price))

def get_residual_sum_of_squares(input_feature, output, intercept, slope):
    # First get the predictions
    predictions = get_regression_predictions(input_feature,intercept,slope)
    # then compute the residuals (since we are squaring it doesn't matter 
    #which order you subtract)
    residual = predictions - output 
    # square the residuals and add them up
    resid_sq = residual*residual
    RSS = resid_sq.sum()
    return(RSS)

print(get_residual_sum_of_squares(test_feature, test_output, 
                                  test_intercept, test_slope)) # should be 0.0

rss_prices_on_sqft = get_residual_sum_of_squares(train_data['sqft_living'], train_data['price'], sqft_intercept, sqft_slope)
print('The RSS of predicting Prices based on Square Feet is : ' 
      + str(rss_prices_on_sqft))

def inverse_regression_predictions(output, intercept, slope):
    # solve output = intercept + slope*input_feature for input_feature. Use this equation to compute the inverse predictions:
    estimated_feature = (output - intercept)/slope
    return estimated_feature

my_house_price = 800000
estimated_squarefeet = inverse_regression_predictions(my_house_price, sqft_intercept, sqft_slope)
print("The estimated squarefeet for a house worth $%.2f is %d" % (my_house_price, estimated_squarefeet))

bed_intercept, bed_slope = simple_linear_regression(train_data['bedrooms'], train_data['price'])
print("Intercept: " + str(bed_intercept))
print("Slope: " + str(bed_slope))

rss_prices_on_bedrooms = get_residual_sum_of_squares(test_data['bedrooms'], test_data['price'], bed_intercept, bed_slope)
print('The RSS of predicting Prices based on Square Feet is : ' + str(rss_prices_on_bedrooms))

rss_prices_on_sqft_test = get_residual_sum_of_squares(test_data['sqft_living'], test_data['price'], sqft_intercept, sqft_slope)
print('The RSS of predicting Prices based on Square Feet is : ' + str(rss_prices_on_sqft_test))

print(rss_prices_on_bedrooms-rss_prices_on_sqft_test)