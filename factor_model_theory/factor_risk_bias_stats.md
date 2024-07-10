_This post is a deep-dive into the structure of fundamental factor models, and as such will be more interesting for equity quants and investors. By dividing the separating the structural model of returns in a fundamental factor model from the estimation of covariance and risk, it makes clear why average cross-sectional R^2 is considered a critical performance metric for these models. It demonstrates the effect higher R^2 has on both the average and the distribution of risk estimates, and discusses why this isn’t always evident from analyses of delivered risk._

Fundamental factor portfolios dominate equity risk modelling and are ubiquitous across areas of finance where low-dimensional structural models are not available. There are a number of reasons for this – historical, computational, and pedagogical – but the simple factor space which the model gives its users hides significant complexity. Even the most expert users often relying unwittingly on subtle theoretical or practical modelling choices which make meaningful differences to the quality of risk estimate provided.

Over on the site formerly known as twitter, Giuseppe Paleologo – one of the most renowned and expert equity in the world – observes that, while model builders provide cross-sectional R2 as a key model metric, the reasons for this are not always clear:

https://x.com/__paleologo/status/1805600874825498734

An answer to this subtle question leads us to a beautiful illustration of some of the deeper structural characteristics – but starts with a very quick historical digression. Why do fundamental factor models exist at all?

**The Historical Development of Fundamental Factor Models**

Equity securities are intrinsically harder to model than other securities. They are huge in number, but their characteristics change through time and can be hidden from the user. There are so many, in fact, that there isn’t enough data to properly estimate security volatilities and correlations at all. For example, calculating the security covariance matrix of the S&P 500 requires the calculation of 125,250 individual security relationships. Doing so robustly would require 500 trading years of daily data, and inverting the resulting matrix for optimisation was beyond the capabilities of the computers available at their time of development.

The early pioneers of fundamental factor models, Barr Rosenberg and Vinay Marathe, made the following breakthrough: although stocks’ individual exposures to the true macroeconomic drivers of common returns were not available, many of these exposures would have measurable microeconomic drivers which would at least correlate with the desired structural market exposures.

Instead of directly calculating security correlations, they estimated a structural model of the stock market by  mapping returns to observable microeconomic characteristics of individual stocks. The resulting model explain returns, attributing them to factor correlates, and leaving behind a residual return for each stock. This allowed for easy and intuitive explanation of the characteristics of risk to users. It also significantly improved the resulting risk estimates: having much lower dimensionality, it was more statistically robust and had the added advantage of reducing the computational intensity of risk models in a time of much more expensive compute.

**Axioms for building Fundamental Factor Models**

The process of constructing such a model is essentially unchanged from these early days, and involves two steps:

- First, security returns for each day are decomposed into returns for each factor, plus a residual (called the idiosyncratic return) for each security, usually via cross-sectional regression of security returns against their input factor exposures for each period.
  (Those very familiar with non-equity models will note this is essentially true of all models, even where the model can look extremely structural in nature. For example, even for rates models the input return factors are constructed from some function of daily yield curve changes. These are generally derived by splines, which is just another statistical technique)

- Second, these factor and residual returns are used to estimate a factor risk model, usually in the form of a covariance matrix, and a method for estimating idiosyncratic risk for each security is chosen.

This second choice - how to handle the volatility of and correlation between these residual security returns - is what we’re focused on today. To obtain the benefits of robustness and ease of calculation, the standard approach makes a crucial assumption: it assumes any returns not captured by the factor structure used are uncorrelated across stocks.

This choice has a different character to the usual issues around modelling factor risk such as mis-estimated covariance or unlikely but possible out-of-sample outcomes from the expected distribution, deliberately discarding information about security co-movement. This becomes a crucial potential limitation of these models.

**Estimating Fundamental Factor Models in Simulated Markets**

The easiest way to understand the ramifications this modelling choice is to do some simulation. Let’s construct a market characterised by five fundamental factors, and then estimate two models. All bar one of the factors will be contained within both models – one model will be overfit, and thus contain all true factors plus a second spurious factor; the other will be underfit, only containing the four most significant factors in the market. No robustness adjustments will be made to either model.

Running the model once, we are returned the following estimated risk model parameters:



Factor volatilities in both models are similar to each other and to the true distributional parameters; the overfit model correctly captures the risk of the missing factor but captures a small amount of variance from the overfit factor. Estimated residual volatilities are similar in both models, but with telling differences: idiosyncratic risk is slightly higher for the underfit model, particularly for the right tail of securities with non-zero exposure to the missing factor.

Finally, average cross-sectional R-squared is very slightly higher for the “overfit” model. An increase from 0.202 to 0.22 in this model – equivalent to an 10% increase in explained sum of squares – seems trivial. Should we care?

Models Must Work Well Across All Possible Portfolios

A model being constructed for a large number of users has to work across a large number of portfolio contexts. Small average increases in explanatory power can result in much more meaningful improvements in risk estimates for individual portfolios.

Risk model are assessed via bias statistic: the delivered risk of a portfolio, divided by the risk it was estimated to have. In real world contexts, this often yields unclear results, as noted by Paleologo in his original tweets:



This lack of clarity is driven by the impossibility of separating differences in structural quality from the inevitable noise of random return outcomes. As the results show, models of differing quality may deliver indistinguishable outcomes when comparing to ex-post returns. However, in our simulation we can go better, comparing model estimates to the true ex-ante risk of any given portfolios.

Construct a number of random portfolios, we can compare various risk estimates to both delivered returns and the true market structure:



Despite knowing for sure that we are not correctly estimating market structure, both our underfit and overfit model yields essentially the same distribution of bias statistics as the perfect-knowledge ex-ante risk estimate.

However, when we look at the distribution of bias statistics of delivered volatility and our various model estimates relative to the true ex-ante risk of each portfolio, we get a markedly different picture: although the mean bias statistic is indistinguishable across metrics, the distribution of bias statistics is much narrower for the higher r-squared model.

**Decomposing The Portfolio Level Bias**

To understand why these models are performing so differently, we can decompose the model bias by comparing the risk contributions within the model to that from the true known market structure. This allows us to see the drivers of model over- and under-estimation of risk within given portfolios:



Across all portfolios, factor risk is too high and idiosyncratic risk is too low. In general, the factors are on average capturing too much of the stock-level volatility. This is due to the lack of any robustness controls on daily returns.

The effect of the spurious factor in the overfit model is relatively small; by contrast, the effect on risk from missing a true factor is very marked, as in a diversified portfolio, the increase in individual stock volatility that results is diluted by the diversification, meaning the risk estimate is markedly low.

Finally, when we decompose the difference between the ex-post portfolio volatility against ex-ante, we see the majority of the risk difference is driven by what I’ve termed as “residual”. This is the (by construction spurious) observed correlation within idiosyncratic stock returns or between idiosyncratic stock returns and the factor block.

How Can We Generalise This As A Linear Algebra Problem?

The effect above can be generalised by considering the linear algebra problem. A market of securities can be thought of as a set of partially-correlated vectors in a high-dimensional space, the length of which corresponds to the individual security risk. 

Model A is a vectorisation of subset A of this space. R2A is the proportion of the overall length of all the vectors aligned with the subset. Any portfolio p represents a vector within the space, the length of which reflects the which can be expressed as a linear combination of weighted security vectors, wisi. Each security vector can be broken down into fAi, a component contained within A and rAi, a residual component orthogonal to the space A.

The assumption made by a fundamental factor model is that |sum(wifAi)|, the length of the sum of the vectors within A, can be greater than or less than sqrt(sum(|wisi|2). However, the length of the sum of the vectors |sum(wifAi)|. By contrast, the length of the sum of the vectors orthogonal to A |sum(wisiAi)| is exactly equal to sqrt(sum(|wisAi|2).

The claim there is a model with a meaningfully higher R2 is equivalent to the following assumption. There exists a space B which is a superset of A. Define C as the union between B and the complement of A. The vectors fCi, the portion of the vector of security i which is contained within C, are not orthogonal for some subset of securities. This means that there are a number of portfolios for which |sum(wifCi)| are definitionally higher and lower than |sum(wifCi)|.

In this world, the length of the risk vectors in B are sqrt(|sum(wifAi)|2 + |sum(wifCi)|2 + sum((wi|sBi|)2). Depending on the weightings, |sum(wifCi)|2 may be greater than or less than sum((wi|fCi|)2), but because we are taking the square root of sum of squares, the effect where it is larger will be to increase the true risk estimate by more than it is reduced when it is smaller.

As such, for a subset of portfolios, the model B will yield correctly higher risk estimates than model A. This misestimation of risk will be most pronounced when the portfolio vector is most contained within the portion of the space B which is the complement of A.

Unresolved Questions

The above helps us understand why average R^2is a crucial sales metric for the manufacturers of risk models - higher R^2 will always reduce the variability of model quality. This is not without cost. Without careful robustness adjustments, the risk captured by factors will be too high and the idiosyncratic variance too low, and risk for portfolios loaded on any further missing factors will still have their risk underestimated. Analysis of the statistical methods and risk management practices available to mitigate these issues is one for a later date. 

_Benedict Treloar is an investment scientist and risk manager with institutional experience in portfolio construction, asset allocation, behavioural finance and quantitative risk modelling across rates, credit and equities. He is currently taking a brief career break with his new child, having left a previous role in May for a quant research role which fell through. He will be writing on interesting quant finance topics, working on machine learning, and campaigning for regulatory changes to speed the development of cures for rare genetic disease._ 

_All code is available at https://github.com/benedicttreloar/quant_finance_articles. The author can be contacted through substack or at https://www.linkedin.com/in/benedict-treloar-34a62327/_



This is essentially a very strong form regularisation, effectively shrinking the empirical stock-stock and stock-factor correlations to zero

The market has five factors:

three factors with normally distributed exposures and annualised vol of 6%

one industry factor, with 90% of the market having exposure 0 and 10% having exposure 1

one lower-volatility missing factor with normally distributed exposures and 2% annualised vol

The market will contain 1000 securities, with stock specific vol of 20%. Factor exposures and factor returns are assumed to be uncorrelated, and risk models are estimated using 10y of trading data, an unrealistic assumption chosen for illustrative purposes

Daily factor and residual returns are taken directly from an OLS regression – no robustness adjustments have been made whatsoever, either in the daily factor return regression or based on the resulting stream of returns.

In specific, we construct 200 equally weighted stock portfolios containing 60 random stocks

This chart shows the contributions to the difference in bias by factor for all portfolios, sorted by the bias of the risk stats, and then smoothed