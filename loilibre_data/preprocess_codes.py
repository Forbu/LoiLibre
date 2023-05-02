"""
In this file, we will preprocess the data for the model.
We preprocess the data using unstructure IO.
"""

import os
from unstructured.partition.pdf import partition_pdf

# directory where the pdf files are stored
directory = '/home/loilibre_data/data/'

# we list the pdf files
pdf_files = [f for f in os.listdir(directory) if f.endswith('.pdf')]

# we partition the pdf files
partition_pdf(filename=os.path.join(directory, pdf_files[-2]),)