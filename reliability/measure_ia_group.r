# Import the irr library
library(irr)

# Read data
data = read.csv("grouping/grouping.csv")

# Measure agreement using Fleiss' kappa
kappam.fleiss(data, detail = TRUE)
