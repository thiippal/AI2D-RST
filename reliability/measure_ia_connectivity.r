# Import the irr library
library(irr)

# Read data
data = read.csv("connectivity/connectivity.csv")

# Measure agreement using Fleiss' kappa
kappam.fleiss(data, detail = TRUE)
