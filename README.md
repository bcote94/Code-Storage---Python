
# Market ML Process

# NOTE: This repo is currently being restructured into a proper PyPi package / to be more pythonic. ML/ folder is old (crappy) code.

![](https://github.com/bcote94/Market-Learning/blob/master/AMD_Performance.png)

## Overview
This package implements a "market learner", generating a classifier that can predict the direction of a stocks movement in an arbitrary forward time window. The default is a 20-day-forward classifier.

## Motivation
Market modeling is one of Statistics most challenging problems. Stocks are generally dynamic, non-parametric, chaotic, and noisy in nature. Even worse, as stocks in a day-to-day basis semi-perfectly account for all past (known) information in their price, any future movement is essentially a random walk. **This means all information relating to the underlying asset itself is likely meaningless, as that information is already accounted for in the price.** This might mean trading is a futile effort. However, Jagadeesh & Titman and Jacobsen & Zhang indicate two theories:

	1. Short term behavior of stock prices exhibit momentum
	2. Stocks tend to follow measurable, seasonal trends
	
Based on this information, **I will build a model irrespective of any information relating to the underlying asset** but purely as a mathematical formulation of momentum and volatility in both short and long-term. 

## Methods
Most market models attempt to predict price or price-differential. This is a foolhardy approach. Forecasting multivariate timeseries is itself an incredibly difficult task under ideal conditions. We are not in ideal conditions. Even worse, these methods violate basic assumptions of an equity. For instance, though not especially likely, it is technically possible for a time series forecast of price to predict an equity price of under $0. This is obviously impossible. 

Classification is a far more robust modeling strategy. It is solid in its assumptions (a stock price at any given point is guaranteed to go up or down), classification of direction irrespective of scale is more robust than predictive forecasting, and it is less susceptible to outliers. The downside being we lose scale (only know if it's going up/down, not by how much), but there are measures to mitigate this, such as deliberately biasing toward Specificity to avoid as many losses as possible. 

## Model Assessment
It's not enough, in my view, to just see how well the model predicts up/down movement. Does it actually make money? 

### Learner Strategy
I use a very naive trading strategy to assess the performance of this model. Every time a buy-signal is detected, purchase a single share at current price. Every sell signal, sell the entire position and wait. 

### Adversarial Strategy
Simply detecting if a model makes money is irrelevant. We need to compare it to another, adversarial trading strategy to see how it performs relative to it. I compare it to Dollar-Cost Averaging. In this method, every 20 day period at the same time, I buy a share no matter what and hold until the end of the trading period. 

## Results
I bring in three stocks with distinct behavior to assess the learner. $AMD, $TSLA, and $AMZN. I trained from 2011-01-01 to 2018-12-31 and testing from 2019-01-01 to 2019-10-15. That is, performance of the model Year-to-Date (YTD). 

### Advanced Micron Devices ($AMD)
AMD is a very fun example. For most of the training period it is essentially 'sideways', and then in 2016 experienced a sharp uptick of rapid volatility. It has consistently increased in price in 2019.

	YTD ML Profits:  $14 per share
	YTD DCA Profits: $41 per share

### Tesla ($TSLA)
Tesla is a very fun case. Its IPO was near the beginning of our training date, so we get to predict on its entire history. It experienced meteoric growth and then settled into a sideways behavior. The fun being, in 2019, it has experienced consistent downward movement, dropping from a high of $347 to a low of $179. Can the ML model do well here?

	YTD ML Profits:  $121 per share
	YTD DCA Profits: $-339 (loss) per share

### Amazon ($AMZN)
Amazon is a good 'sideways' case for the year of 2019.

	YTD ML Profits:  $1330 per share
	YTD DCA Profits: $224 per share
	
## Conclusions
On stocks consistently increasing in price, Dollar-Cost Averaging outperforms an ML model. On sideways or downward moving stock, the ML learner outperforms DCA considerably. I think an even more aggressive trading strategy might help the model outperform in 'positive' stocks as well. For instance, saving money every sell signal to buy more shares next buy, or reinvesting winnings into even more shares per buy signal. However for now, these are very satisfying results!
