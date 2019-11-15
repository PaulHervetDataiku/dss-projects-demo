# -*- coding: utf-8 -*-
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu

# Read recipe inputs
missing_survived_scored = dataiku.Dataset("missing_survived_scored")
missing_survived_scored_df = missing_survived_scored.get_dataframe()


# Compute recipe outputs from inputs
# TODO: Replace this part by your actual code that computes the output, as a Pandas dataframe
# NB: DSS also supports other kinds of APIs for reading and writing data. Please see doc.

test_df = missing_survived_scored_df # For this sample code, simply copy input to output


# Write recipe outputs
test = dataiku.Dataset("test")
test.write_with_schema(test_df)
