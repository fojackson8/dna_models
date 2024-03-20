import numpy as np
import pandas as pd
import os
import os.path
import sys
import pysam
from pyfaidx import Fasta
from collections import Counter
import random
import re
import codecs
import collections
import pickle
sys.path.append('/well/ludwig/users/dyp502/read_classifier/code/sample_processing/')

from bam_processing_functions import taps_bam_to_read_nov_22
from datetime import datetime

overwrite = True

# filtered_bam_dir = "/well/ludwig/users/dyp502/read_classifier/dismir_experiments/taps_dmrs/filtered_bams/"
filtered_bam_dir = "/well/ludwig/users/dyp502/read_classifier/read_classifier_2024/bam_intersections/"
filtered_reads_dir = "/well/ludwig/users/dyp502/read_classifier/read_classifier_2024/read_files/"

file_list = os.listdir(filtered_bam_dir)
filename = file_list[int(sys.argv[1])]
# filename = file_list[0]
# filename = "CD564934_Liver-Tumour_md.Liver_tumour_vs_healthy_dmbs.top1000.hyper_hypo.bam"
# filename="CD564146_Liver-Tumour_md.Liver_tumour_vs_healthy_dmbs.top300.hyper_hypo.bam"
savename = filename.replace('.bam','.reads')

log_dir = "/well/ludwig/users/dyp502/read_classifier/read_classifier_2024/logs/"

if (not os.path.exists(filtered_reads_dir + savename)) or overwrite:
    print(f"Converting: {filename}")
    taps_bam_to_read_nov_22(filename,savename,filtered_bam_dir,filtered_reads_dir,log_dir=log_dir,trimmed=False,save=True,read_objects=False)
else:
    print(f"File exists already at: {filtered_reads_dir + savename}")

